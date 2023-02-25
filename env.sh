#!/bin/bash
###
 # @Author: Deng Yifan 553192215@qq.com
 # @Date: 2022-08-19 16:22:15
 # @LastEditors: Deng Yifan 553192215@qq.com
 # @LastEditTime: 2022-10-05 16:08:33
 # @FilePath: /faith_dial/env.sh
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

project_path=$(pwd)
export PYTHONPATH=project_path:$PYTHONPATH

#ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Replace conda channels...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" red
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --set show_channel_urls yes

#echo ''
#echo ''
#ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Replace pip channels...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " red
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo ''
echo ''
ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Create conda env...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " yellow
# Configure conda env
read -rp "Enter environment name: " env_name
read -rp "Enter python version (recommended 3.8) : " python_version
ADD_COLOR "Init conda environment!!!" blue
# conda create -y -n lightning pandas pip python=3.8
conda create -y -n "$env_name" pandas pip python="$python_version"
ADD_COLOR "activate this environment" blue

# 如果出现conda activate无法使用的情况，可以试试下面的命令
# source activate
# conda deactivate
source activate
conda activate $env_name

echo ''
echo ''
ADD_COLOR "Current environment: "$env_name blue

echo ''
echo ''
ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Installing Basic package...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" pink
python -m pip install ranger-fm  # 文件管理器
# 根据自己的环境选择合适的版本
# https://pytorch.org/get-started/locally/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

echo ''
echo ''
ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Installing requirements...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " green
python -m pip install git+https://gitclone.com/github.com/Maluuba/nlg-eval.git@master
python -m pip install hydra-core --upgrade
python -m pip install -r requirements.txt

# 去官网下载安装包，直接用代码下载速度较慢
# https://spacy.io/models/en/#en_core_web_sm
# 安装下载好的en_core_web_sm-3.4.0.tar.gz，路径替换为自己的路径
#python -m pip install en_core_web_sm-3.4.0.tar.gz
#python -m spacy download en_core_web_sm

# 如果出现CUDA无法调用的问题，比如：
# RuntimeError: CUDA error: no kernel image is available for execution on the device
# 很有可能是torch的问题，在下面链接安装对应的版本
# https://download.pytorch.org/whl/torch_stable.html
# 比如：在CUDA11.6上，搜索"cu116"然后找与自己python版本相对应的链接，例如"cp39"
# pip install https://download.pytorch.org/whl/cu116/torch-1.12.1%2Bcu116-cp39-cp39-linux_x86_64.whl
# 或者：在CUDA11.3上，搜索"cu113"然后找与自己python版本相对应的链接，例如"cp38"
# python -m pip install https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp38-cp38-linux_x86_64.whl

echo ''
echo ''
ADD_COLOR "Current environment:  "$env_name" is ready to use! " blue
ADD_COLOR "Use --> conda activate "$env_name" <-- to activate this environment" blue
echo ''
echo ''

# 下面的命令需要确认相关文件夹是否有权限写入
# mv .vscode/.vscode_server ~/.vscode_server
# 缺少 nltk_data，从这里下载： https://github.com/D-Yifan/nltk_data
# git clone https://github.com/D-Yifan/nltk_data【提示的搜索路径】

# 安装 stanford-corenlp-4.5.1stanford-corenlp-4.5.1
# cd $project_path/general_files/utils/others/stanford_nlp
# wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.1.zip
# unzip stanford-corenlp-4.5.1.zip

# 安装 java
# sudo apt-get install openjdk-8-jdk


ADD_COLOR "安装 Redis" blue
cd $project_path
base_path=$(pwd)
export PYTHONPATH=$base_path:$PYTHONPATH
wget https://download.redis.io/redis-stable.tar.gz
tar -xzvf redis-stable.tar.gz
cd redis-stable
make distclean && make
cd src
make install PREFIX=$base_path/redis-stable/redis
cd ..
cd redis/bin
# 参考: https://zhuanlan.zhihu.com/p/552627015
# 设置 redis.conf
git clone https://github.com/D-Yifan/redis.conf.git
cd ..
./bin/redis-server ./bin/redis.conf

ADD_COLOR ""
ADD_COLOR ""
ADD_COLOR "环境准备完毕！" blue


#wget https://code-common-resources.oss-cn-beijing.aliyuncs.com/nltk_data.zip
#unzip nltk_data.zip
