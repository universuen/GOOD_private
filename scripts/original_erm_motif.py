import context

from GOOD.kernel.main import *

if __name__ == '__main__':
    args = args_parser(
        [
            '--config_path', 'GOOD_configs/GOODMotif/basis/covariate/ERM.yaml'
        ]
    )
    config = config_summoner(args)
    load_logger(config)

    model, loader = initialize_model_dataset(config)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

    pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
    pipeline.load_task()

    if config.task == 'train':
        pipeline.task = 'test'
        pipeline.load_task()
