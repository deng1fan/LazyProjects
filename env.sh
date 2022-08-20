#!/bin/bash

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
read -rp "Enter environment name: " env_name
read -rp "Enter python version (recommended 3.9) : " python_version
ADD_COLOR "Init conda environment!!!" blue
conda create -y -n "$env_name" matplotlib pandas pip python="$python_version"
ADD_COLOR "activate this environment" blue
source activate
conda activate $env_name
echo ''
echo ''
ADD_COLOR "Current environment: "$env_name blue

echo ''
echo ''
ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Installing Basic package...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" pink
# ！如果出现CUDA无法调用的问题，比如：
# RuntimeError: CUDA error: no kernel image is available for execution on the device
# 很有可能是torch的问题，在下面链接安装对应的版本
# https://download.pytorch.org/whl/torch_stable.html
# 比如：在CUDA11.6上，搜索"cu116"然后找与自己python版本相对应的链接，例如"cp39"
pip install https://download.pytorch.org/whl/cu116/torch-1.12.0%2Bcu116-cp39-cp39-linux_x86_64.whl
# ！根据自己的环境选择合适的版本
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio zip cudatoolkit

# 安装tokenizers依赖rust，所以需要先安装rust
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 去官网下载安装包，直接用代码下载速度较慢
# https://spacy.io/models/en/#en_core_web_sm
# 安装下载好的en_core_web_sm-3.4.0.tar.gz，路径替换为自己的路径
pip install /home/dengyf/en_core_web_sm-3.4.0.tar.gz

echo ''
echo ''
ADD_COLOR ">>>>>>>>>>>>>>>>>>>>>>>Installing requirements...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " green
pip install git+https://gitclone.com/github.com/Maluuba/nlg-eval.git@master
pip install -r requirements.txt

echo ''
echo ''
ADD_COLOR "Current environment:  "$env_name" is ready to use! " blue
echo ''
echo ''