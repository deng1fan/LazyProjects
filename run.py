'''
Author: Deng Yifan 553192215@qq.com
Date: 2022-08-25 08:27:32
LastEditors: Deng Yifan 553192215@qq.com
LastEditTime: 2022-08-26 17:33:01
FilePath: /dg_templete/run.py
Description: 

Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved. 
'''
# -*- coding: utf-8 -*-

import os
import sys
import hydra
import comet_ml
from omegaconf import DictConfig, OmegaConf, ListConfig
from test import test
from train import train
from general_files.utils.common_util import (
    get_logger,
    check_config,
    print_config,
    extras,
    Result,
    get_parent_dir,
)
from general_files.utils.data_util import pp
from general_files.utils.common_util import (
    RedisClient,
    dingtalk_sender_and_wx,
    print_start_image,
    print_end_image,
    print_error_info,
)
import setproctitle
import yaml

log = get_logger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

with open("./configs/config.yaml", 'r') as file:
    global_config = DictConfig(yaml.safe_load(file.read()))


@hydra.main(version_base="1.2", config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> float:
    setproctitle.setproctitle(config.proc_title)

    config = check_config(config)

    # Pretty print config using Rich library
    if config.print_config:
        print_config(config, resolve=True)

    experiment = None
    if config.logger and config.logger == "comet":
        comet_ml.init(
            project_name=config.logger_project,
            experiment_key=config.experiment_key,
        )
        experiment = comet_ml.Experiment(
            log_git_patch=True,
            log_git_metadata=True,
            auto_histogram_tensorboard_logging=True,
            display_summary_level=0,
            log_code=True,
            auto_histogram_weight_logging=True,
        )
        experiment_config = sys.argv[-1].replace("+experiment=", "")
        experiment_hyper_args = ' '.join(sys.argv[1:])
        experiment.set_name(config.comet_name)
        experiment.add_tag(config.stage)
        experiment.log_other("备注", config.run_notes)
        experiment.log_other("实验标识", config.task_full_name)
        experiment.log_other("进程ID", str(os.getpid()))
        experiment.log_other("experiment", experiment_config)
        experiment.log_other("experiment_hyper_args", experiment_hyper_args)
        # 设置上传代码文件
        # 上传config
        experiment.log_asset(config.config_dir +
                             "/experiment/" + experiment_config + ".yaml")
        experiment.log_asset(config.config_dir + "/config.yaml")
        # 上传数据处理文件
        experiment.log_asset(config.work_dir + "/data_processor/" +
                             config.dataset_processor.replace('.', '/') + ".py")
        # 上传模型文件
        model_processor_name = config.model_processor
        if 'base:' in model_processor_name:
            module_path = 'general_files.models.' + \
                model_processor_name.replace("base:", "")
        else:
            module_path = config.logger_project + '.models.' + model_processor_name
        module_path = module_path.replace('.', '/')
        experiment.log_asset(config.root_dir + '/' + module_path + ".py")

        if ':' in config.pretrain_model:
            sub_model_processor_name = config.pretrain_model.split(':')[0]
            sub_module_path = config.logger_project + \
                '.modules.' + sub_model_processor_name
            sub_module_path = sub_module_path.replace('.', '/')
            experiment.log_asset(config.root_dir + '/' +
                                 sub_module_path + ".py")

    extras(config)

    # 模型的训练或测试
    if experiment is None:
        try:
            test_results = train_or_test(config, log, experiment)
        except KeyboardInterrupt as e:
            pp("程序受到人为中断！")
        except Exception as e:
            print_error_info(e)
            raise e
    else:
        try:
            test_results = train_or_test_with_DingTalk(config, log, experiment)

            if config.logger == "comet":
                if test_results is not None:
                    experiment.log_metrics(dict(test_results))
                    experiment.add_tag("Metric")
                if config.eval_bad_case_analysis:
                    experiment.add_tag("Bad Case Analysis")
                experiment.add_tag("Finish")
                experiment.set_name(config.comet_name + "  OK!")
        except KeyboardInterrupt as e:
            print("程序受到人为中断！")
            if config.logger == "comet":
                experiment.add_tag("KeyboardInterrupt")
                experiment.set_name(config.comet_name + "  Interrupt!")
        except Exception as e:
            print_error_info(e)
            if config.logger == "comet":
                experiment.add_tag("Crashed")
                experiment.set_name(config.comet_name + "  Error!")
        finally:
            if config.logger == "comet":
                experiment.end()

    return float(test_results['f1'])


@dingtalk_sender_and_wx(
    webhook_url=global_config.dingding_web_hook,
    secret=global_config.dingding_secret,
)
def train_or_test_with_DingTalk(config, log, experiment):
    test_results = Result()
    if (
        config.stage == "train"
        or config.stage == "pretrain"
        or config.stage == "finetune"
    ):
        test_results = train(config, experiment)
    elif config.stage == "test":
        test_results = test(config, experiment)
    else:
        test_results.add(msg="请设置正确的stage!")
        print("请设置正确的stage!")
    test_results.add(
        run_name=config.run_name,
        run_notes=config.run_notes,
    )
    return test_results


def train_or_test(config, log, experiment):
    test_results = Result()
    if (
        config.stage == "train"
        or config.stage == "pretrain"
        or config.stage == "finetune"
    ):
        test_results = train(config, experiment)
    elif config.stage == "test":
        test_results = test(config, experiment)
    else:
        test_results.add(msg="请设置正确的stage!")
        print("请设置正确的stage!")
    test_results.add(
        run_name=config.run_name,
        run_notes=config.run_notes,
    )
    return test_results


if __name__ == "__main__":
    print_start_image()

    main()

    print_end_image()
