# 项目变化时自动变换路径
# shellcheck disable=SC2046

base_path=$(pwd)
export PYTHONPATH=$base_path

cd $base_path/general_files/utils/others/redis_client || exit
work_path=$base_path/general_files/utils/others/redis_client

nohup python maintain_redis_data.py > $work_path/redis_nohup.txt 2>&1
