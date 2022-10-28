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
FilePath: /run.py
Description: 

Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved. 
"""
# -*- coding: utf-8 -*-
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from general_files.trainer.processor import get_trainer_processor
from general_files.utils.others.data_processor.processor import get_data_processor
from omegaconf import DictConfig
import setproctitle
import yaml
from general_files.utils.common_util import (
    Result,
    get_logger,
    check_config,
    print_config,
    set_config_gpus,
    init_context,
    dingtalk_sender_and_wx,
    print_start_image,
    print_end_image,
    print_error_info,
    print_generated_dialogs,
    init_comet_experiment,
    seed_everything,
    RedisClient,
)
from general_files.utils.model_util import (
    get_eval_metrics,
    generate_sentences,
    predict_labels,
)
from general_files.utils.data_util import (
    concatenate_multi_datasets,
    save_as,
    read_by,
    print_sample_data,
)

log = get_logger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

with open("./configs/default_config.yaml", "r") as file:
    global_config = DictConfig(yaml.safe_load(file.read()))

###############################################
# 设置随机种子
###############################################
log.info(f"设置 seed 为:  {global_config.seed}")
seed_everything(global_config.seed)


@hydra.main(version_base="1.2", config_path="configs/", config_name="default_config.yaml")
def main(config: DictConfig) -> float:

    ###############################################
    # 设置个性化进程名
    ###############################################
    setproctitle.setproctitle(str(os.getpid()) + "->" + config.proc_title)

    ###############################################
    # 检查配置
    ###############################################
    config = check_config(config)

    ###############################################
    # 打印配置信息
    ###############################################
    if config.print_config:
        print_config(config, resolve=True)


    if config.logger == "comet":
        test_results = train_or_test_with_DingTalk(
            config)
    else:
        test_results = train_or_test(config)


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

    ###############################################
    # 加载测试输出结果缓存
    ###############################################
    log.info("初始化训练、测试等所需环境")
    if config.stage == "test":
        # 在测试阶段，如果有之前的生成缓存，则直接读取
        test_output_path = config.ckpt_path
        if ".ckpt" in test_output_path:
            test_output_path = "/".join(test_output_path.split("/")[:-1])
        if os.path.exists(test_output_path + "/test_output.pt"):
            log.info(f"发现测试输出结果缓存，准备加载...: {test_output_path}")
            test_output = read_by(
                test_output_path + "/test_output.pt", data_name="测试输出")
        if config.ckpt_path and os.path.exists(config.ckpt_path + "/tokenizer.pt"):
            # 微调、测试的分词器加载
            tokenizer = read_by(config.ckpt_path +
                                "/tokenizer.pt", data_name="tokenizer")

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
            trainer.train()

            log.info(f"使用最优模型进行预测/生成！")
            model = trainer.model

        ###############################################
        # 生成测试输出结果缓存
        ###############################################
        if test_output is None:

            ###############################################
            # 模型测试
            ###############################################
            model.eval()
            model = model.to(config.default_device)

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
                test_output = concatenate_multi_datasets(test_output, raw_data[-2])
            else:
                test_output = concatenate_multi_datasets(test_output, raw_data[-1])

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

            if not config.fast_run:
                if config.ckpt_path:
                    test_output_path = config.ckpt_path
                else:
                    test_output_path = config.result_path
                log.info(f"保存测试输出结果...: {test_output_path}")
                save_as(test_output, test_output_path +
                        "/test_output", data_name="测试输出")

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
            test_results = get_eval_metrics(test_output, config, tokenizer)

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

        test_results.add(
            run_name=config.comet_name,
            run_notes=config.run_notes,
        )

        ###############################################
        # 更新 Comet 信息
        ###############################################
        if experiment:
            if test_results is not None:
                experiment.log_metrics(dict(test_results))
                experiment.add_tag("Metric")
            if config.eval_bad_case_analysis:
                experiment.add_tag("Bad Case Analysis")
            experiment.add_tag("Finish")
            experiment.set_name(config.comet_name + "  OK!")
    
    except KeyboardInterrupt as e:
        print("程序受到人为中断！")
        if config.get("logger") == "comet" and experiment:
            experiment.add_tag("KeyboardInterrupt")
            experiment.set_name(config.comet_name + "  Interrupt!")
            experiment.end()
            raise e
    except Exception as e:
        print_error_info(e)
        if config.get("logger") == "comet" and experiment:
            experiment.add_tag("Crashed")
            experiment.set_name(config.comet_name + "  Error!")
            experiment.end()
            raise e
        

    return test_results


if __name__ == "__main__":

    print_start_image()

    main()

    print_end_image()
