###
# @Author: Deng Yifan 553192215@qq.com
# @Date: 2022-09-26 22:12:54
 # @LastEditors: appleloveme 52270975+appleloveme@users.noreply.github.com
 # @LastEditTime: 2022-10-28 21:23:53
 # @FilePath: /dg_templete/allennlp.sh
# @Description:
#
# Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved.
###

# 下载地址：http://parl.ai/downloads/personachat/personachat.tgz
# 参考自： https://github.com/facebookresearch/ParlAI/blob/mturk_archive/parlai/tasks/personachat/build.py
wget http://parl.ai/downloads/personachat/personachat.tgz
tar -zxvf personachat.tgz

# 下载地址：http://parl.ai/downloads/convai2/convai2_fix_723.tgz
# 参考自： https://github.com/facebookresearch/ParlAI/blob/mturk_archive/parlai/tasks/convai2/build.py
wget http://parl.ai/downloads/convai2/convai2_fix_723.tgz
tar -zxvf convai2_fix_723.tgz

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

ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Replace conda channels...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" red
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

echo ''
echo ''
ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Replace pip channels...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " red
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo ''
echo ''
ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Create conda env...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " yellow
# Configure conda env
env_name=allennlp
conda create -n $env_name python=3.8
ADD_COLOR "activate allennlp environment" blue
source activate
conda activate $env_name
echo ''
echo ''
ADD_COLOR "Current environment: "$env_name blue

ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Install allennlp...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " yellow
pip install allennlp
pip install allennlp-models
conda install -c conda-forge spacy-model-en_core_web_sm


