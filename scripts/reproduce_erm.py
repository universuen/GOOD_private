from __future__ import annotations
import context

from pathlib import Path

import torch
import numpy as np
import pandas as pd

from GOOD.kernel.main import *
from GOOD.utils.train import nan2zero_get_mask
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from history import History

FILE_NAME = Path(__file__).name.split('.')[0]
SEEDS = list(range(10))

CONFIG_NAME_PATH_PAIRS = {
    f'{FILE_NAME}_motif': 'configs/GOOD_configs/GOODMotif/basis/covariate/ERM.yaml',
    f'{FILE_NAME}_cmnist': 'configs/GOOD_configs/GOODCMNIST/color/covariate/ERM.yaml',
    f'{FILE_NAME}_sst2': 'configs/GOOD_configs/GOODSST2/length/covariate/ERM.yaml',
    f'{FILE_NAME}_pcba': 'configs/GOOD_configs/GOODPCBA/scaffold/covariate/ERM.yaml',
}


def analyze_results_by_ratio(config_name):
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {k: [] for k in ratios}

    for ratio in ratios:
        for seed in range(10):
            try:
                history = History(
                    name=f'test_auc_{seed}',
                    config_name=config_name,
                )
                history.load()
                results[ratio].append(history[int(len(history.values) * ratio) - 1] * 100)
            except (FileNotFoundError, IndexError):
                pass
        mean = round(sum(results[ratio]) / len(results[ratio]), 1)
        std = round(float(np.std(results[ratio])), 1)
        results[ratio] = f'{mean}±{std}'

    pd.options.display.max_columns = None
    results = pd.DataFrame.from_dict(results, orient='index')
    print(results)
    results.to_excel(Path(__file__).absolute().parent / 'results' / config_name / 'analyzed_results.xlsx')


def training_bar(epoch: int, total_epochs: int, **kwargs) -> str:
    content = f'epoch {epoch + 1} / {total_epochs}:'
    for k, v in kwargs.items():
        content = ' '.join([content, f'[{k}:{v:.5f}]'])
    return content


def train_batch(data, config, optimizer, model) -> dict:
    data = data.to(config.device)

    optimizer.zero_grad()

    mask, targets = nan2zero_get_mask(data, 'train', config)

    model_output = model(data=data)
    raw_pred = model_output
    loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
    loss = loss.sum() / mask.sum()

    loss.backward()
    optimizer.step()

    return {'loss': loss.detach()}


@torch.no_grad()
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

    model.train()

    return {'score': stat['score'], 'loss': stat['loss']}


def main(config_name, config_path, seed: int):
    PromptedGINEncoder.enable()
    PromptedVNGINEncoder.enable()
    PromptedVNMolGINEncoder.enable()
    # load config for yml
    args = args_parser(
        [
            '--config_path', config_path
        ]
    )
    config = config_summoner(args)
    config.random_seed = seed

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
    train_auc_history = History(
        name=f'train_auc_{seed}',
        config_name=config_name,
    )
    val_auc_history = History(
        name=f'val_auc_{seed}',
        config_name=config_name,
    )
    test_auc_history = History(
        name=f'test_auc_{seed}',
        config_name=config_name,
    )

    best_val_auc = None
    test_auc_at_best_val = None
    epoch_at_best_val = None
    # train the model
    print(seed, config_name, 'Started training', flush=True)
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
            seed,
            config_name,
            training_bar(
                epoch,
                total_epochs=config.train.max_epoch,
                training_loss=epoch_train_stat['loss'],
                training_auc=epoch_train_stat['score'],
                validating_auc=val_stat['score'],
                test_auc=test_stat['score'],
            ),
            flush=True,
        )
        # adjust lr
        scheduler.step()

        # record
        train_auc_history.append(epoch_train_stat['score'])
        val_auc_history.append(val_stat['score'])
        test_auc_history.append(test_stat['score'])

        train_auc_history.save()
        val_auc_history.save()
        test_auc_history.save()

        # save model
        if best_val_auc is None or val_auc_history.last_one > best_val_auc:
            max_val_auc = val_auc_history.last_one
            test_auc_at_best_val = test_auc_history.last_one
            models_dir = Path(__file__).absolute().parent / 'models'
            models_dir.mkdir(exist_ok=True)
            models_dir = Path(__file__).absolute().parent / 'models' / config_name
            models_dir.mkdir(exist_ok=True)
            path = models_dir / f'{config_name}_{config.random_seed}_best.pt'
            torch.save(
                model.state_dict(),
                path,
            )
            print(f"Saved a new model at {path}")
            print(f'epoch: {epoch + 1} | val_auc: {max_val_auc} | test_auc: {test_auc_at_best_val}')

    print(seed, config_name, 'Done!', flush=True)
    return {
        'epoch': epoch_at_best_val,
        'val': round(best_val_auc, 1),
        'test': round(test_auc_at_best_val, 1),
    }


def test_all_seeds(config_name, relative_path):
    results = dict()
    for i in SEEDS:
        root_path = Path(__file__).absolute().parent.parent
        config_path = str(root_path / relative_path)
        results[i] = main(config_name, config_path, i)

        # analyze results
        mean_epoch = sum(i['epoch'] for i in results.values()) / len(results)
        mean_val = sum([i['val'] for i in results.values()]) / len(results)
        mean_test = sum([i['test'] for i in results.values()]) / len(results)

        pd.options.display.max_columns = None
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    {
                        'mean': {
                            'epoch': round(mean_epoch, 1),
                            'val': round(mean_val, 1),
                            'test': round(mean_test, 1),
                        }
                    }
                ).transpose(),
            ],
        )
        results_df.columns.name = 'seed'
        print(results_df)
        results_dir = Path(__file__).absolute().parent / 'results'
        results_dir.mkdir(exist_ok=True)
        results_dir = Path(__file__).absolute().parent / 'results' / config_name
        results_dir.mkdir(exist_ok=True)
        results_df.to_excel(Path(__file__).absolute().parent / 'results' / config_name / f'results.xlsx')


if __name__ == '__main__':

    from multiprocessing import Process

    torch.multiprocessing.set_start_method('spawn')

    for config_name, relative_path in CONFIG_NAME_PATH_PAIRS.items():
        Process(
            target=test_all_seeds,
            args=(config_name, relative_path),
        ).start()
