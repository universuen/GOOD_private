from __future__ import annotations
import context

from pathlib import Path

import numpy as np
import pandas as pd

from GOOD.kernel.main import *
from GOOD.utils.train import nan2zero_get_mask
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from history import History

CONFIG_NAME = Path(__file__).name.split('.')[0]
CONFIG_PATH = 'configs/GOOD_configs/GOODMotif/basis/covariate/ERM.yaml'
SEEDS = list(range(10))


def analyze_results_by_ratio(ratios: list[int] = None):
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {k: [] for k in ratios}

    for ratio in ratios:
        for seed in range(10):
            try:
                history = History(
                    name=f'test_auc_{seed}',
                    config_name=CONFIG_NAME,
                )
                history.load()
                results[ratio].append(history[int(len(history.values) * ratio) - 1] * 100)
            except (FileNotFoundError, IndexError):
                pass
        mean = sum(results[ratio]) / len(results[ratio])
        std = round(float(np.std(results[ratio])), 1)
        results[ratio] = f'{mean}±{std}'

    pd.options.display.max_columns = None
    results = pd.DataFrame.from_dict(results)
    print(results)
    results.to_excel(Path(__file__).absolute() / 'results' / CONFIG_NAME / 'analyzed_results.xlsx')


def training_bar(epoch: int, total_epochs: int, **kwargs) -> str:
    content = f'epoch {epoch + 1} / {total_epochs}:'
    for k, v in kwargs.items():
        content = ' '.join([content, f'[{k}:{v:.5f}]'])
    return content


def train_batch(data, config, optimizer, model) -> dict:
    r"""
    Train a batch. (Project use only)

    Args:
        data (Batch): Current batch of data.

    Returns:
        Calculated loss.
    """
    data = data.to(config.device)

    optimizer.zero_grad()

    mask, targets = nan2zero_get_mask(data, 'train', config)

    data, targets, mask = data, targets, mask

    model_output = model(data=data)
    raw_pred = model_output
    loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
    loss = loss.sum() / mask.sum()

    loss.backward()
    optimizer.step()

    return {'loss': loss.detach()}


def evaluate(split: str, loader, model, config):
    stat = {'score': None, 'loss': None}
    if loader.get(split) is None:
        return stat
    model.eval()

    loss_all = []
    mask_all = []
    pred_all = []
    target_all = []
    for data in loader[split]:
        data = data.to(config.device)

        mask, targets = nan2zero_get_mask(data, split, config)
        if mask is None:
            return stat
        model_output = model(data=data, edge_weight=None)
        raw_preds = model_output

        # --------------- Loss collection ------------------
        loss = config.metric.loss_func(raw_preds, targets, reduction='none') * mask
        mask_all.append(mask)
        loss_all.append(loss)

        # ------------- Score data collection ------------------
        pred, target = eval_data_preprocess(data.y, raw_preds, mask, config)
        pred_all.append(pred)
        target_all.append(target)

    # ------- Loss calculate -------
    loss_all = torch.cat(loss_all)
    mask_all = torch.cat(mask_all)
    stat['loss'] = loss_all.sum() / mask_all.sum()

    # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
    stat['score'] = eval_score(pred_all, target_all, config)

    print(f'#IN#\n{split.capitalize()} {config.metric.score_name}: {stat["score"]:.4f}\n'
          f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

    model.train()

    return {'score': stat['score'], 'loss': stat['loss']}


def main(seed: int):
    # load config for yml
    root_path = Path(__file__).absolute().parent.parent
    config_path = str(root_path / CONFIG_PATH)
    args = args_parser(
        [
            '--config_path', config_path
        ]
    )
    config = config_summoner(args)
    config.random_seed = seed
    reset_random_seed(config)

    # get model and data loader
    reset_random_seed(config)
    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config)
    model = load_model(config.model.model_name, config)

    # config model
    model.to(config.device)
    model.train()

    # set up optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.train.mile_stones,
        gamma=0.1,
    )

    # set up history
    test_auc_history = History(
        name=f'test_auc_{seed}',
        config_name=CONFIG_NAME,
    )
    # train the model
    print('Started training')
    for epoch in range(config.train.ctn_epoch, config.train.max_epoch):
        config.train.epoch = epoch

        for index, data in enumerate(loader['train']):
            if data.batch is not None and (data.batch[-1] < config.train.train_bs - 1):
                continue

            # train a batch
            train_batch(data, config, optimizer, model)

        # evaluate
        epoch_train_stat = evaluate('eval_train', loader, model, config)
        val_stat = evaluate('val', loader, model, config)
        test_stat = evaluate('test', loader, model, config)

        print(
            training_bar(
                epoch,
                total_epochs=config.train.max_epoch,
                training_loss=epoch_train_stat['loss'],
                training_auc=epoch_train_stat['score'],
                validating_auc=val_stat['score'],
                test_auc=test_stat['score'],
            )
        )
        # adjust lr
        scheduler.step()

        # record
        test_auc_history.append(test_stat['score'])
        test_auc_history.save()

    print('Done!')


if __name__ == '__main__':
    for i in SEEDS:
        main(i)
