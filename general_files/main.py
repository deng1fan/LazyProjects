##########################################################################
#
#
#        ______                  __   ___  __
#        |  _  \                 \ \ / (_)/ _|
#        | | | |___ _ __   __ _   \ V / _| |_ __ _ _ __
#        | | | / _ \ '_ \ / _` |   \ / | |  _/ _` | '_ \
#        | |/ /  __/ | | | (_| |   | | | | || (_| | | | |
#        |___/ \___|_| |_|\__, |   \_/ |_|_| \__,_|_| |_|
#                          __/ |
#                         |___/
#
#
# Github: https://github.com/D-Yifan
# Zhi hu: https://www.zhihu.com/people/deng_yifan
#
##########################################################################

"""
Author: Deng Yifan 553192215@qq.com
Date: 2022-08-25 08:27:32
LastEditors: Deng Yifan 553192215@qq.com
LastEditTime: 2022-09-19 07:23:23
FilePath: /faith_dial/run.py
Description: 

Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved. 
"""
# -*- coding: utf-8 -*-
from omegaconf import DictConfig
import os
import yaml
from general_files.trainer.processor import get_trainer_processor
from general_files.utils.others.data_processor.processor import get_data_processor
from omegaconf import DictConfig
import importlib
from general_files.utils.common_util import (
    send_msg_to_DingTalk_and_wx,
    Result,
    get_logger,
    check_config,
    print_config,
    set_config_gpus,
    init_context,
    dingtalk_sender_and_wx,
    print_error_info,
    print_generated_dialogs,
    init_comet_experiment,
    seed_everything,
    RedisClient,
    print_start_image,
)
from general_files.utils.model_util import (
    get_eval_metrics,
    generate_sentences,
    predict_labels,
)
from general_files.utils.data_util import (
    concatenate_multi_datasets,
    print_sample_data,
    get_custom_test_output,
)
import sys
import torch
import time
from datasets import Dataset

log = get_logger(__name__)

with open("./configs/default_config.yaml", "r") as file:
    global_config = DictConfig(yaml.safe_load(file.read()))


def main(config: DictConfig) -> float:

    print_start_image()

    ###############################################
    # 设置随机种子
    ###############################################
    log.info(f"设置 seed 为:  {config.seed}")
    seed_everything(config.seed)

    ###############################################
    # 检查配置
    ###############################################
    config = check_config(config)

    ###############################################
    # 打印配置信息
    ###############################################
    if config.print_config:
        print_config(config, resolve=True)

    ###############################################
    # 登记进程信息
    ###############################################
    redis_client = RedisClient()
    task_id = redis_client.register_process(config)
    config.task_id = task_id

    if config.logger == "comet":
        test_results, config = train_or_test_with_DingTalk(
            config)
    else:
        test_results, config = train_or_test(config)

    redis_client.deregister_process(config)

    return 0


@dingtalk_sender_and_wx(
    webhook_url=global_config.dingding_web_hook,
    secret=global_config.dingding_secret,
)
def train_or_test_with_DingTalk(config):
    return train_or_test(config)


def train_or_test(config):

    test_output = None
    test_results = Result()
    if config.get("script_path"):
        ###############################################
        # 自动选择 GPU
        ###############################################
        config = set_config_gpus(config)

        ###############################################
        # 第三方模型评估
        ###############################################
        test_output = get_custom_test_output(config)
        log.info("评估模型！")

        if config.eval_metrics is not None:
            test_results = get_eval_metrics(test_output, config)
        return test_results, config

    ###############################################
    # 加载测试输出结果缓存
    ###############################################
    log.info("初始化训练、测试等所需环境")
    if config.stage == "test":
        # 在测试阶段，如果有之前的生成缓存，则直接读取
        test_output_path = config.ckpt_path
        if ".ckpt" in test_output_path:
            test_output_path = "/".join(test_output_path.split("/")[:-1])
        if os.path.exists(test_output_path + "/test_output.csv"):
            log.info(f"发现测试输出结果缓存，准备加载...: {test_output_path}")
            # test_output = read_by(
            #     test_output_path + "/test_output.csv", data_name="测试输出")
            test_output = Dataset.from_csv(test_output_path + "/test_output.csv")
        if config.ckpt_path:
            # 微调、测试的分词器加载
            tokenizer_module_path = "general_files.modules.tokenizer"
            tokenizer_module = importlib.import_module(tokenizer_module_path)
            tokenizer = getattr(tokenizer_module, "Tokenizer")
            tokenizer = tokenizer(config=config)



    if test_output is None:

        ###############################################
        # 加载数据集、模型、分词器
        ###############################################
        # 训练或微调，或者测试时没有缓存，需要重新加载数据
        (model,
         tokenizer,
         train_data_tokenized,
         valid_data_tokenized,
         test_data_tokenized,
         raw_data,
         ) = init_context(config)

    ###############################################
    # 自动选择 GPU
    ###############################################
    config = set_config_gpus(config)

    
    ###############################################
    # 初始化 Comet
    ###############################################
    experiment = init_comet_experiment(config)

    try:
        if test_output is None:
            print_sample_data(
                tokenizer,
                [train_data_tokenized, valid_data_tokenized, test_data_tokenized],
                ["Train data", "Valid data", "Test data"],
                config=config,
                experiment=experiment,
            )

        ###############################################
        # 模型训练
        ###############################################
        if config.stage in ["train", "finetune", "pretrain"]:
            # 非测试阶段
            log.info(f"初始化 Trainer...")
            trainer_processor = get_trainer_processor(config)
            trainer = trainer_processor(
                config=config,
                model=model,
                train_dataset=train_data_tokenized,
                eval_dataset=valid_data_tokenized,
                tokenizer=tokenizer,
                experiment=experiment,
            )

            log.info(f"训练开始！")
            
            # 发送钉钉通知
            try:
                send_msg_to_DingTalk_and_wx(f"{config.comet_name} 开始训练！🏃🏻🏃🏻🏃🏻", config)
            except Exception as e:
                print_error_info(e)
                log.info(f"发送钉钉通知失败: {e}")
                
            model = trainer.train()

        ###############################################
        # 生成测试输出结果缓存
        ###############################################
        if test_output is None:

            ###############################################
            # 模型测试
            ###############################################
            model.eval()
            model = model.to(config.default_device)

            log.info(f"使用最优模型进行预测/生成！")
            if config.data_mode == "classification":
                test_output = test_data_tokenized.map(
                    lambda batch: {
                        "generated": predict_labels(model, batch, tokenizer, config=config)
                    },
                    batched=True,
                    batch_size=config.test_batch_size,
                    desc="正在预测分类标签",
                )
            else:
                test_output = test_data_tokenized.map(
                    lambda batch: {
                        "generated": generate_sentences(
                            model, batch, tokenizer, config=config
                        )
                    },
                    batched=True,
                    batch_size=config.test_batch_size,
                    desc="正在生成",
                )

            if config.eval_bad_case_analysis:
                test_output = concatenate_multi_datasets(
                    test_output, raw_data[-2])
            else:
                test_output = concatenate_multi_datasets(
                    test_output, raw_data[-1])

            if config.data_mode != "classification":
                test_output = test_output.map(
                    lambda batch: {
                        "generated_seqs": batch["generated"]["seqs"],
                        "generated_seqs_with_special_tokens": batch["generated"][
                            "seqs_with_special_tokens"
                        ],
                    },
                    desc="生成语句字典展开映射",
                )
                test_output = test_output.remove_columns(["generated"])


        ###############################################
        # 清空 cuda 缓存
        ###############################################
        if config.stage in ["train", "finetune", "pretrain"]:
            print(torch.cuda.memory.memory_summary())
            log.info("清空 cuda 缓存")

            model = model.to("cpu")
            torch.cuda.empty_cache()
            time.sleep(5)
        
        print(torch.cuda.memory.memory_summary())

        ###############################################
        # 将所有输出列名标准化以使用统一的评价指标函数
        ###############################################
        data_processor = get_data_processor(config, tokenizer)
        test_output = data_processor.map_column(test_output)
        if config.data_mode != "classification":
            # 保存测试生成语句方便以后测试
            print_generated_dialogs(
                test_output, mode=config.data_mode, config=config, experiment=experiment)

        ###############################################
        # 模型评估
        ###############################################
        log.info("评估模型！")
        if config.eval_metrics is not None:
            test_results = get_eval_metrics(test_output, config)

        ###############################################
        # 打印 ckpt 存储信息
        ###############################################
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
            
            
        tmux_session = ""
        for arg in sys.argv:
            if "tmux_session" in arg:
                tmux_session = arg.replace("+tmux_session=", "")

        test_results.add(
            run_name=config.comet_name,
            comet_name=config.comet_name,
            memo=config.memo,
        )
        if tmux_session != "":
            test_results.add(tmux_session=tmux_session)

        ###############################################
        # 更新 Comet 信息
        ###############################################
        if experiment:
            if test_results is not None:
                for key, value in test_results.items():
                    if key in ["run_name", "comet_name", "memo", "tmux_session"]:
                        continue
                    experiment.log_metric(key, value)
                experiment.add_tag("Metric")
            if config.eval_bad_case_analysis:
                experiment.add_tag("Bad Case Analysis")
            experiment.add_tag("Finish")

    except KeyboardInterrupt as e:
        print("程序受到人为中断！")
        if config.get("logger") == "comet" and experiment:
            experiment.add_tag("KeyboardInterrupt")
            experiment.set_name(config.comet_name + "  Interrupt!")
            raise e
    except RuntimeError as e:
        print_error_info(e)
        if config.get("logger") == "comet" and experiment:
            experiment.add_tag("Crashed")
            experiment.set_name(config.comet_name + "  Error!")
            raise Exception(e)
    except Exception as e:
        print_error_info(e)
        if config.get("logger") == "comet" and experiment:
            experiment.add_tag("Crashed")
            experiment.set_name(config.comet_name + "  Error!")
            raise e
    finally:
        if config.get("logger") == "comet" and experiment:
            experiment.end()
    return test_results, config
