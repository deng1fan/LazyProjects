# -*- coding: utf-8 -*-
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

from omegaconf import DictConfig, ListConfig
import os
import hydra
import setproctitle


@hydra.main(version_base="1.2", config_path="configs/", config_name="default_config.yaml")
def main(config: DictConfig) -> float:

    ###############################################
    # 📍📍📍 设置个性化进程名
    ###############################################
    setproctitle.setproctitle(str(os.getpid()) + ": " + config.proc_title)

    ###############################################
    # 📍📍📍 设置环境变量
    ###############################################
    os.environ['HOME'] = "/home/TableSense/large_disk/zsh"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    os.environ["COMET_API_KEY"] = config.comet_api_key
    os.environ["COMET_PROJECT_NAME"] = config.logger_project
    os.environ["COMET_AUTO_LOG_ENV_CPU"] = "False"
    os.environ["COMET_AUTO_LOG_ENV_GPU"] = "False"
    os.environ["COMET_AUTO_LOG_ENV_DETAILS"] = "True"
    os.environ["COMET_AUTO_LOG_CO2"] = "False"
    os.environ["COMET_AUTO_LOG_GIT_METADATA"] = "True"
    os.environ["COMET_AUTO_LOG_GIT_PATCH"] = "False"
    # os.environ['MPLCONFIGDIR'] = os.getcwd() + "cache_dir/matplotlib/"
    if isinstance(config.visible_cuda, ListConfig):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            [str(cuda) for cuda in config.visible_cuda])

    from general_files.main import main
    main(config)


if __name__ == "__main__":

    main()
