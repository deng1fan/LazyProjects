import os
import torch
from omegaconf import DictConfig, OmegaConf
from general_files.models.base_files.processor import get_model_processor
from general_files.utils.common_util import print_generated_dialogs, RedisClient, print_parameters, get_logger, set_config_gpus
from general_files.utils.others.data_processor.processor import get_data_processor
from general_files.utils.data_util import get_tokenized_data, concatenate_multi_datasets, print_dataset_overview, print_sample_data, read_by
from general_files.utils.model_util import get_eval_metrics, generate_sentences, predict_labels

log = get_logger(__name__)


def test(config: DictConfig, experiment):
    # Init model and tokenizer
    log.info(f"初始化模型...:  {config.pretrain_model}")
    model, tokenizer = get_model_processor(config)
    model = model.to(config.default_device)
    print_parameters(model)

    test_output_path = config.ckpt_path
    if '.ckpt' in test_output_path:
        test_output_path = '/'.join(test_output_path.split('/')[:-1])
    if not os.path.exists(test_output_path + '/test_output.pt') or config.eval_bad_case_analysis:
        # Init data
        log.info(f"初始化数据集...: {config.dataset}")
        if os.path.exists(config.cache_dataset_path + '_preprocess_dataset.pt'):
            log.info(f"发现缓存数据集，准备加载...: {config.cache_dataset_path}_preprocess_dataset.pt")
            train_data_tokenized, valid_data_tokenized, test_data_tokenized, raw_data = read_by(
                config.cache_dataset_path + '_preprocess_dataset.pt', data_name='数据集缓存')
            print_dataset_overview(train_data_tokenized, valid_data_tokenized, test_data_tokenized)
        elif config.get('eval_bad_case_analysis'):
            train_data_tokenized, valid_data_tokenized, test_data_tokenized, raw_data = get_tokenized_data(
                config=config,
                tokenizer=tokenizer,
                model=model)
            test_data_tokenized = valid_data_tokenized
        else:
            _, _, test_data_tokenized, raw_data = get_tokenized_data(config=config,
                                                                     tokenizer=tokenizer,
                                                                     model=model,
                                                                     only_test=True)

        config = set_config_gpus(config)

        ###############################################
        # 更新Comet信息
        ###############################################
        if config.logger and config.logger == "comet":
            for key in config.keys():
                if isinstance(config[key], DictConfig) or isinstance(config[key], OmegaConf):
                    for key2 in config[key].keys():
                        experiment.log_other(f"{str(key)}:{str(key2)}", config[key][key2])
                else:
                    experiment.log_other(str(key), config[key])

        print_sample_data(tokenizer, [test_data_tokenized], ['Test data'], experiment=experiment, config=config)
        # Make predictions
        model.eval()
        model = model.to(config.default_device)
        if 'pl_' in config.trainer_processor:
            model.config.stage = 'test'
        else:
            model.hyparam.stage = 'test'
        if config.eval_bad_case_analysis:
            log.info(f"使用验证集作为测试集，进行Bad case生成分析！")
        log.info(f"使用最优模型进行预测/生成！")
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
        
        log.info(f"保存测试输出结果...: {config.ckpt_path}")
        torch.save(test_output, test_output_path + '/test_output', data_name="测试输出")

        # 将所有输出列名标准化以使用统一的评价指标函数
        data_processor = get_data_processor(config, tokenizer)
        test_output = data_processor.map_column(test_output)
        if config.data_mode != 'classification':
            # 保存测试生成语句方便以后测试
            print_generated_dialogs(test_output, experiment, mode=config.data_mode,
                                    config=config)
        if config.eval_metrics is not None:
            log.info("评估模型！")
            test_results = get_eval_metrics(test_output, model, config, tokenizer)
            return test_results
        return None
    log.info(f"发现测试输出结果缓存，准备加载...: {test_output_path}")
    test_output = read_by(test_output_path + '/test_output.pt', data_name="测试输出")

    # 将所有输出列名标准化以使用统一的评价指标函数
    data_processor = get_data_processor(config, tokenizer)
    test_output = data_processor.map_column(test_output)
    if config.data_mode != 'classification':
        log.info("展示部分生成文本！")
        print_generated_dialogs(test_output, experiment, config=config)
    if config.eval_metrics is not None:
        log.info("评估模型！")
        test_results = get_eval_metrics(test_output, model, config, tokenizer)
        return test_results
    
    log.info(f"运行结果保存在：")
    log.info(f"{config.result_path}")
    
    ###############################################
    # 删除Redis的Gpu占用记录
    ###############################################
    if config.task_id:
        redis_client = RedisClient()
        redis_client.deregister_gpus(config)
    
    return None
