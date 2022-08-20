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

# 项目变化时自动变换路径
base_path=$(dirname $(dirname $(pwd)))
export PYTHONPATH=$base_path

cd $base_path
work_path=$base_path/nohup

# 参数解释
# >覆盖，
# >>追加
# 2>&1 表示不仅命令行正常的输出保存到log中，产生错误信息的输出也保存到log文件中
day_time=$(date "+%Y-%m-%d")
hour_time=$(date "+%H-%M-%S")
read -p "如需启用experiment，请输入名称(eg:'e1,e2')，否则按回车: " experiment


mkdir -p $work_path/$day_time

ADD_COLOR "此次运行对应的日志保存在: " bleu
ADD_COLOR $work_path/$day_time/$hour_time green

if [ -z ${experiment} ]; then
		experiment=""
elif [[ $experiment == *","* ]]; then
		experiment="--multirun +experiment="$experiment
		ADD_COLOR "Using experiment: "$experiment pink
else
		experiment="+experiment="$experiment
		ADD_COLOR "Using experiment: "$experiment pink
fi

nohup python run.py fast_run=False use_gpu=True wait_gpus=True force_reload_data=True logger=comet > $work_path/$day_time/$hour_time.txt 2>&1 $experiment