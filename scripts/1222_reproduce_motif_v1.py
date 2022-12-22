import context

from pathlib import Path
from multiprocessing import Process

from GOOD.kernel.main import *

CONFIG_PATH = 'configs/GOOD_configs/GOODMotif/basis/covariate/ERM.yaml'
SEEDS = list(range(10))


def train_and_test(seed: int):
    root_path = Path(__file__).absolute().parent.parent
    config_path = str(root_path / CONFIG_PATH)
    args = args_parser(
        [
            '--config_path', config_path
        ]
    )
    config = config_summoner(args)
    config.random_seed = seed
    load_logger(config)

    model, loader = initialize_model_dataset(config)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

    pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
    pipeline.load_task()

    if config.task == 'train':
        pipeline.task = 'test'
        pipeline.load_task()


if __name__ == '__main__':
    for i in SEEDS:
        train_and_test(i)
