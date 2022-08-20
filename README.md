# dg_templete
![](https://img.shields.io/badge/License-GNU%20General%20Public%20License%20v3.0-green)
![](https://img.shields.io/badge/Python-3.8-blue)
![](https://img.shields.io/badge/知乎-一个邓-orange)

[博客](https://zhuanlan.zhihu.com/p/552293287)

深度学习项目框架

使用前需要先根据env.sh中的注释修改相应的选项

如果有微信、钉钉通知的需要，需要申请对应的Webhook和token，然后在代码中查找todo填进去

# 依赖包
    bash env.sh

# 数据集

    # 数据集下载
    wget http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
    # 解压缩 
    tar -zxvf wizard_of_wikipedia.tgz

# 运行
    bash run.sh
    
# 启动redis维护
    bash redis.sh
