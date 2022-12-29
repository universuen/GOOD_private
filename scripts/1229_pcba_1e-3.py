from __future__ import annotations

import context

from pathlib import Path

import torch
from torch import nn
import numpy as np
import pandas as pd
import torch_geometric.nn as gnn

from GOOD.kernel.main import *
from GOOD.utils.train import nan2zero_get_mask
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.networks.models import GINs, GINvirtualnode
from history import History

FILE_NAME = Path(__file__).name.split('.')[0]
SEEDS = list(range(10))
ADV_STEP_SIZE = 1e-3
ADV_NUM_ITER = 3
BATCH_SIZE = 128

CONFIG_NAME_PATH_PAIRS = {
    # f'{FILE_NAME}_motif': 'configs/GOOD_configs/GOODMotif/basis/covariate/ERM.yaml',
    # f'{FILE_NAME}_cmnist': 'configs/GOOD_configs/GOODCMNIST/color/covariate/ERM.yaml',
    # f'{FILE_NAME}_sst2': 'configs/GOOD_configs/GOODSST2/length/covariate/ERM.yaml',
    f'{FILE_NAME}_pcba': 'configs/GOOD_configs/GOODPCBA/scaffold/covariate/ERM.yaml',
}

CONFIG_NAME = None


def logging_print(*args, **kwargs):
    global CONFIG_NAME
    assert CONFIG_NAME is not None
    logs_dir = Path(__file__).absolute().parent / 'logs'
    logs_dir.mkdir(exist_ok=True)
    with open(logs_dir / f'{CONFIG_NAME}.log', 'a') as f:
        args = [str(i) for i in args]
        f.write(f"{' '.join(args)}\n")
    print(*args, **kwargs)


class Prompt(nn.Module):
    def __init__(
            self,
            batch_size: int,
            length: int,
            uniform_init_interval: tuple[float, float] = (
                    -ADV_STEP_SIZE, ADV_STEP_SIZE
            ),
    ):
        super().__init__()

        self.uniform_init_interval = uniform_init_interval
        self.b = torch.nn.Parameter(
            torch.zeros(
                batch_size,
                length,
            )
        )
        nn.init.uniform_(self.b, *self.uniform_init_interval)

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        return x + self.b[batch]


class PromptedGINEncoder(GINs.GINEncoder):
    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.prompts = None
        self.embedding = nn.Linear(config.dataset.dim_node, config.model.dim_hidden, bias=False)
        self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        x = self.embedding(x)
        if self.prompts is None:
            return super().forward(x, edge_index, batch, batch_size, **kwargs)

        x = self.prompts[0](x, batch)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)
        ):
            post_conv = self.prompts[i + 1](post_conv, batch)
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout

    @classmethod
    def enable(cls):
        GINs.GINEncoder = cls


class PromptedVNGINEncoder(GINvirtualnode.vGINEncoder):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super().__init__(config, **kwargs)
        self.prompts = None
        self.embedding = nn.Linear(config.dataset.dim_node, config.model.dim_hidden, bias=False)
        self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        x = self.embedding(x)
        if self.prompts is None:
            return super().forward(x, edge_index, batch, batch_size, **kwargs)

        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch_size, device=self.config.device, dtype=torch.long))

        x = self.prompts[0](x, batch)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)
        ):
            post_conv = self.prompts[i + 1](post_conv, batch)
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(
                    self.virtual_pool(post_conv, batch, batch_size) + virtual_node_feat
                )

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout

    @classmethod
    def enable(cls):
        GINvirtualnode.vGINEncoder = cls


class PromptedVNMolGINEncoder(GINvirtualnode.vGINMolEncoder):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super().__init__(config, **kwargs)
        self.prompts = None

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):

        if self.prompts is None:
            return super().forward(x, edge_index, edge_attr, batch, batch_size, **kwargs)

        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch_size, device=self.config.device, dtype=torch.long)
        )

        x = self.atom_encoder(x)
        x = self.prompts[0](x, batch)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)
        ):
            post_conv = self.prompts[i + 1](post_conv, batch)
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(
                    self.virtual_pool(post_conv, batch, batch_size) + virtual_node_feat
                )

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout

    @classmethod
    def enable(cls):
        GINvirtualnode.vGINMolEncoder = cls


def training_bar(epoch: int, total_epochs: int, **kwargs) -> str:
    content = f'epoch {epoch + 1} / {total_epochs}:'
    for k, v in kwargs.items():
        content = ' '.join([content, f'[{k}:{v:.5f}]'])
    return content


def train_batch(data, config, optimizer, model) -> dict:
    data = data.to(config.device)

    # add prompts
    prompts = nn.ModuleList(
        [
            Prompt(
                batch_size=config.train.train_bs,
                length=config.model.dim_hidden,
            ).to(config.device)
            for _ in range(config.model.model_layer)
        ]
    )
    assert hasattr(model.feat_encoder.encoder, 'prompts')
    model.feat_encoder.encoder.prompts = prompts

    optimizer.zero_grad()

    # calculate loss
    mask, targets = nan2zero_get_mask(data, 'train', config)
    model_output = model(data=data)
    raw_pred = model_output
    loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
    loss = loss.sum() / mask.sum()
    loss /= ADV_NUM_ITER

    # maximize loss by updating prompts
    for _ in range(ADV_NUM_ITER - 1):
        # calculate gradients
        loss.backward()
        # update prompts parameters based on gradients sign
        for i in prompts.parameters():
            assert i.grad is not None
            i_data = i.detach() + ADV_STEP_SIZE * torch.sign(i.grad.detach())
            i.data = i_data.data
            i.grad[:] = 0
        # calculate loss
        mask, targets = nan2zero_get_mask(data, 'train', config)
        model_output = model(data=data)
        raw_pred = model_output
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss = loss.sum() / mask.sum()
        loss /= ADV_NUM_ITER

    # minimize loss by updating others
    loss.backward()
    optimizer.step()

    # remove prompts
    model.feat_encoder.encoder.prompts = None

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
    config.train.train_bs = BATCH_SIZE

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
    logging_print(seed, config_name, 'Started training', flush=True)
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

        logging_print(
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
            epoch_at_best_val = epoch + 1
            best_val_auc = val_auc_history.last_one
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
            logging_print()
            logging_print(seed, config_name, "New model saved:")
            logging_print(f'***** epoch: {epoch + 1} | val_auc: {best_val_auc} | test_auc: {test_auc_at_best_val} *****')
            logging_print()

    logging_print(seed, config_name, 'Done!', flush=True)
    return {
        'epoch': epoch_at_best_val,
        'val': round(best_val_auc * 100, 1),
        'test': round(test_auc_at_best_val * 100, 1),
    }


def test_all_seeds(config_name, relative_path):
    global CONFIG_NAME
    CONFIG_NAME = config_name

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
        logging_print(results_df)
        results_dir = Path(__file__).absolute().parent / 'results'
        results_dir.mkdir(exist_ok=True)
        results_dir = Path(__file__).absolute().parent / 'results' / config_name
        results_dir.mkdir(exist_ok=True)
        results_df.to_excel(Path(__file__).absolute().parent / 'results' / config_name / f'results.xlsx')


if __name__ == '__main__':

    from multiprocessing import Process

    # torch.multiprocessing.set_start_method('spawn')

    for config_name, relative_path in CONFIG_NAME_PATH_PAIRS.items():
        test_all_seeds(config_name, relative_path)
        # Process(
        #     target=test_all_seeds,
        #     args=(config_name, relative_path),
        # ).start()
