from general_files.models.base_files.processor import get_model_processor
from general_files.trainer.processor import get_trainer_processor
from general_files.utils.common_util import (
    print_parameters,
    print_generated_dialogs,
    get_logger,
    seed_everything,
    print_gpu_info,
    set_config_gpus,
    RedisClient
)
from general_files.utils.others.data_processor.processor import get_data_processor
from general_files.utils.data_util import (
    get_tokenized_data,
    concatenate_multi_datasets,
    save_as,
    read_by,
    print_dataset_overview,
    print_sample_data
)
from general_files.utils.model_util import get_eval_metrics, generate_sentences, predict_labels
import os
from omegaconf import DictConfig, OmegaConf

log = get_logger(__name__)


def train(config: DictConfig, experiment):
    # Set seed for random number generators in pytorch, numpy and python.random
    log.info(f"设置 seed 为:  {config.seed}")
    seed_everything(config.seed)

    # Init model and tokenizer
    model_name = config.pretrain_model if config.pretrain_model is not None else config.model_processor
    log.info(f"初始化模型...: {model_name}")
    model, tokenizer = get_model_processor(config)
    print_parameters(model)

    # Init data
    log.info(f"初始化数据集...: {config.dataset}")
    if os.path.exists(config.cache_dataset_path + '_preprocess_dataset.pt') and not config.force_reload_data and not config.fast_run:
        log.info(
            f"发现缓存数据集，准备加载...: {config.cache_dataset_path}_preprocess_dataset.pt")
        train_data_tokenized, valid_data_tokenized, test_data_tokenized, raw_data = read_by(
            config.cache_dataset_path + '_preprocess_dataset.pt', data_name="数据集缓存")
        print_dataset_overview(train_data_tokenized,
                               valid_data_tokenized, test_data_tokenized)
    else:
        train_data_tokenized, valid_data_tokenized, test_data_tokenized, raw_data = get_tokenized_data(
            config=config,
            tokenizer=tokenizer,
            model=model)
        if not config.fast_run:
            log.info(
                f"保存数据集缓存...: {config.cache_dataset_path}_preprocess_dataset.pt")
            save_as((train_data_tokenized, valid_data_tokenized, test_data_tokenized, raw_data),
                    config.cache_dataset_path + '_preprocess_dataset', data_name="数据集缓存")
    print_sample_data(tokenizer, [train_data_tokenized, valid_data_tokenized, test_data_tokenized],
                      ['Train data', 'Valid data', 'Test data'], experiment=experiment, config=config)

    config = set_config_gpus(config)

    ###############################################
    # 更新Comet信息
    ###############################################
    if config.logger and config.logger == "comet":
        for key in config.keys():
            if isinstance(config[key], DictConfig) or isinstance(config[key], OmegaConf):
                for key2 in config[key].keys():
                    experiment.log_other(
                        f"{str(key)}:{str(key2)}", config[key][key2])
            else:
                experiment.log_other(str(key), config[key])

    # Init Trainer
    log.info(f"初始化 Trainer...")
    trainer_processor = get_trainer_processor(config)
    trainer = trainer_processor(config=config,
                                model=model,
                                train_dataset=train_data_tokenized,
                                eval_dataset=valid_data_tokenized,
                                tokenizer=tokenizer,
                                experiment=experiment,
                                )
    # Train the model
    log.info("准备就绪！ 开始训练！")
    log.info('实验标识： ' + config.task_full_name)
    log.info('实验备注： ' + config.run_notes)
    log.info("正在搜集可用GPU信息")
    print_gpu_info(config.visible_cuda)
    trainer.train()
    # log.info(f"保存最优模型...: {config.result_path}")
    # trainer.save_model(config.result_path + '/best_model.ckpt')

    if config.eval_bad_case_analysis:
        test_data_tokenized = valid_data_tokenized

    # Make predictions
    model = trainer.model.to(config.default_device)
    model.eval()
    model.config.stage = 'test'
    if config.eval_bad_case_analysis:
        log.info(f"使用验证集作为测试集，进行Bad case生成分析！")
    log.info(f"使用最优模型进行预测/生成!")
    if config.data_mode == 'classification':
        test_output = test_data_tokenized.map(
            lambda batch: {'generated': predict_labels(
                model,
                batch,
                tokenizer,
                config=config
            )
            },
            batched=True,
            batch_size=config.test_batch_size,
            desc='正在预测分类标签'
        )
    else:
        test_output = test_data_tokenized.map(
            lambda batch: {'generated': generate_sentences(
                model,
                batch,
                tokenizer,
                config=config
            )
            },
            batched=True,
            batch_size=config.test_batch_size,
            desc='正在生成'
        )
    if config.eval_bad_case_analysis:
        test_output = concatenate_multi_datasets(test_output, raw_data[-2])
    else:
        test_output = concatenate_multi_datasets(test_output, raw_data[-1])
    if config.data_mode != 'classification':
        test_output = test_output.map(
            lambda batch: {'generated_seqs': batch['generated']['seqs'],
                           'generated_seqs_with_special_tokens': batch['generated']['seqs_with_special_tokens'],
                           },
            desc='生成语句字典展开映射')
        test_output = test_output.remove_columns(['generated'])

    if not config.fast_run:
        log.info(f"保存测试输出结果...: {config.result_path}")
        save_as(test_output, config.result_path +
                '/test_output', data_name="测试输出")

    # 将所有输出列名标准化以使用统一的评价指标函数
    data_processor = get_data_processor(config, tokenizer)
    test_output = data_processor.map_column(test_output)
    if config.data_mode != 'classification':
        # 保存测试生成语句方便以后测试
        print_generated_dialogs(test_output, experiment,
                                mode=config.data_mode, config=config)
    log.info("评估模型！")
    if config.eval_metrics is not None:
        test_results = get_eval_metrics(test_output, model, config, tokenizer)
    else:
        test_result = {}

    if not config.fast_run:
        log.info(f"如果要使用此次模型，请设置 ckpt_identifier 为: ")
        log.info(f"{config.task_full_name}")
        log.info(f"运行结果保存在：")
        log.info(f"{config.result_path}")

    ###############################################
    # 删除Redis的Gpu占用记录
    ###############################################
    if config.task_id:
        redis_client = RedisClient()
        redis_client.deregister_gpus(config)

    return test_results
