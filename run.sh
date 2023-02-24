
###
# @Author: Deng Yifan 553192215@qq.com
# @Date: 2022-08-19 14:09:34
# @LastEditors: appleloveme 553192215@qq.com
# @LastEditTime: 2022-12-06 22:43:43
# @FilePath: /codes_frame/run.sh
# @Description:
#
# Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved.
###
ADD_COLOR(){
    RED_COLOR='\E[1;31m'
    GREEN_COLOR='\E[1;32m'
    YELLOW_COLOR='\E[1;33m'
    BLUE_COLOR='\E[1;34m'
    PINK_COLOR='\E[1;35m'
    RES='\E[0m'
    
    #这里判断传入的参数是否不等于2个，如果不等于2个就提示并退出
    if [ $# -ne 2 ];then
        echo "Usage $0 content {red|yellow|blue|green|pink}"
        exit
    fi
    case "$2" in
        red|RED)
            echo -e "${RED_COLOR}$1"
        ;;
        yellow|YELLOW)
            echo -e "${YELLOW_COLOR}$1"
        ;;
        green|GREEN)
            echo -e "${GREEN_COLOR}$1"
        ;;
        blue|BLUE)
            echo -e "${BLUE_COLOR}$1"
        ;;
        pink|PINK)
            echo -e "${PINK_COLOR}$1"
        ;;
        *)
            echo -e "请输入指定的颜色代码：{red|yellow|blue|green|pink}"
    esac
}

SEND_KEYS(){
    tmux send-keys -t "$2"  "$1"
}

ADD_COLOR "\n###########################################################################\n\n" blue
ADD_COLOR "        ______                  __   ___  __" blue
ADD_COLOR "        |  _  \x5C                 \x5C \x5C / (_)/ _|" blue
ADD_COLOR "        | | | |___ _ __   __ _   \x5C V / _| |_ __ _ _ __" blue
ADD_COLOR "        | | | / _ \x5C '_ \x5C / _\x60 |   \x5C / | |  _/ _\x60 | '_ \x5C" blue
ADD_COLOR "        | |/ /  __/ | | | (_| |   | | | | || (_| | | | |" blue
ADD_COLOR "        |___/ \x5C___|_| |_|\x5C__, |   \x5C_/ |_|_| \x5C__,_|_| |_|" blue
ADD_COLOR "                          __/ |" blue
ADD_COLOR "                         |___/" blue
ADD_COLOR "\n\n" blue
ADD_COLOR "        Github: https://github.com/D-Yifan" blue
ADD_COLOR "        Zhi hu: https://www.zhihu.com/people/deng_yifan" blue
ADD_COLOR "\n###########################################################################\n" blue


# 项目变化时自动变换路径
base_path=$(pwd)
export PYTHONPATH=$base_path
read -p "请输入要运行的实验项目名称：" project_name
#read -p "使用的后台运行命令：1、Tmux；2、Nohup（输入序号，为空则默认为 Tmux）" background_task
#if [[ ${background_task =~ ""  ]]; then
#  background_task=1
exps=(`cat configs/experiments/$project_name/experimental_plan.yaml | shyaml keys experiments`)

exps_num=${#exps[@]}  # 实验数量

run_num=0  # 已经运行的实验数量

ADD_COLOR "\n本次计划实验共有：$exps_num 个\n" yellow

day_time=$(date "+%Y-%m-%d")
hour_time=$(date "+%H-%M-%S")

experiment_plan_id=$day_time-$hour_time

for exp in ${exps[@]};
do
    run_num=$((run_num+1))
    config_name=`cat configs/experiments/$project_name/experimental_plan.yaml | shyaml get-value 'experiments.'$exp'.config_name'`
    memo=`cat configs/experiments/$project_name/experimental_plan.yaml | shyaml get-value 'experiments.'$exp'.hyper_params.memo'`
    ADD_COLOR "🎬🎬🎬   $run_num、 启动 $exp 实验！  🎬🎬🎬" green
    ADD_COLOR "实验备注：$memo" pink
    ADD_COLOR "实验配置文件：$config_name" pink

    # 获取配置文件中的参数
    exp_keys=`cat configs/experiments/$project_name/experimental_plan.yaml | shyaml keys 'experiments.'$exp`
    config_keys=`cat configs/experiments/$config_name.yaml | shyaml keys`
    default_config_keys=`cat configs/default_config.yaml | shyaml keys`
    sweep_args=""
    if [[ ${exp_keys[@]}  =~ "hyper_params"  ]]; then
        hyper_params=`cat configs/experiments/$project_name/experimental_plan.yaml | shyaml keys 'experiments.'$exp'.hyper_params'`
        for hyper_param_key in ${hyper_params[@]};
        do
            hyper_param_value=`cat configs/experiments/$project_name/experimental_plan.yaml | shyaml get-value 'experiments.'$exp'.hyper_params.'$hyper_param_key`
            SUB="'"
            if [[ $hyper_param_value =~ ^(\.[0-9]+|[0-9]+(\.[0-9]*)?)$ ]]; then
                sub_sweep_args=$hyper_param_key"="$hyper_param_value
            else
                sub_sweep_args=$hyper_param_key"=\\\""$hyper_param_value"\\\""
            fi
            if [[ $config_keys =~ $hyper_param_key ]]; then
                sweep_args=$sweep_args${sub_sweep_args//" "/""}" "
            else
                if [[ $default_config_keys =~ $hyper_param_key ]]; then
                    sweep_args=$sweep_args${sub_sweep_args//" "/""}" "
                else
                    sweep_args=$sweep_args"+"${sub_sweep_args//" "/""}" "
                fi
            fi
        done
    fi

############################################################################################################
#    使用 tmux 启动实验
#    tmux_session=${exp//" "/"--"}-$day_time-$hour_time
#    tmux new-session -d -s $tmux_session
#    ADD_COLOR "tmux session name: "${exp//" "/"--"}-$day_time-$hour_time pink
#    SEND_KEYS "cd "$base_path $tmux_session
#    SEND_KEYS C-m $tmux_session
#    SEND_KEYS "conda activate lightning" $tmux_session
#    SEND_KEYS C-m $tmux_session
#    SEND_KEYS C-m $tmux_session
#    run_command="python run.py $sweep_args +tmux_session=$tmux_session +experiment_plan_id=$experiment_plan_id comet_name=$exp fast_run=False use_gpu=True wait_gpus=True force_reload_data=True logger=comet "$experiment
#
#    SEND_KEYS "$run_command" $tmux_session
#    ADD_COLOR "启动命令:" pink
#    ADD_COLOR "python run.py $sweep_args +tmux_session=$tmux_session +experiment_plan_id=$experiment_plan_id comet_name=$exp fast_run=False use_gpu=True wait_gpus=True force_reload_data=True logger=comet "$experiment pink
#    ADD_COLOR "使用 tmux a -t $tmux_session 命令查看实验进度"
#    ADD_COLOR "使用 tmux kill-session -t $tmux_session 命令结束实验" pink
#    SEND_KEYS C-m $tmux_session
############################################################################################################

############################################################################################################
#   使用 nohup 启动
    cd $base_path
    tmux_session="None"
    run_command="python run.py $sweep_args +tmux_session=$tmux_session +experiment_plan_id=$experiment_plan_id $experiment comet_name=$exp fast_run=False use_gpu=True wait_gpus=True force_reload_data=True logger=comet "
    run_command=${run_command//"\\"/""}
    nohup $run_command > nohups/${exp//" "/"--"}-$day_time-$hour_time.out 2>&1 &
    ADD_COLOR "启动命令:" pink
    ADD_COLOR "nohup $run_command > nohups/${exp//" "/"--"}-$day_time-$hour_time.out 2>&1 &" pink
    ADD_COLOR "可使用下面命令打开 nohup 日志：" pink
    ADD_COLOR "tail -f $base_path/nohups/${exp//" "/"--"}-$day_time-$hour_time.out" pink
    ADD_COLOR "使用 Ctrl + C 命令退出日志（不会终止实验）" pink
############################################################################################################

    # 防止误占 GPU
    if [ $run_num -lt $exps_num ]; then
        ADD_COLOR "等待 15 秒钟，防止误占 GPU ......" red
        sleep 15
    fi
    ADD_COLOR "" red
done



ADD_COLOR "\n🎉🎉🎉    实验计划已经全部启动！后续请关注 Comet.ml 和 钉钉 获取最新动态！       🎉🎉🎉\n" yellow
ADD_COLOR "" blue
