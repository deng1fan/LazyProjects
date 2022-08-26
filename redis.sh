# 项目变化时自动变换路径
# shellcheck disable=SC2046

base_path=$(pwd)
export PYTHONPATH=$base_path

nohup python $base_path/general_files/utils/others/redis_client/maintain_redis_data.py > $base_path/redis_nohup.txt 2>&1
