# 项目变化时自动变换路径
###
 # @Author: appleloveme 553192215@qq.com
 # @Date: 2022-08-19 14:09:34
 # @LastEditors: appleloveme 553192215@qq.com
 # @LastEditTime: 2022-10-16 22:44:11
 # @FilePath: /faith_dial/redis.sh
 # @Description: 
 # 
 # Copyright (c) 2022 by appleloveme 553192215@qq.com, All Rights Reserved. 
### 
# shellcheck disable=SC2046
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

base_path=$(pwd)
export PYTHONPATH=$base_path

nohup python $base_path/general_files/utils/others/redis_client/maintain_redis_data.py > $base_path/nohups/redis_nohup.txt 2>&1 &
ADD_COLOR "redis checker 服务已启动!" green
