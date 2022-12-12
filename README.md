<!--
 * @Author: Deng Yifan 553192215@qq.com
 * @Date: 2022-08-26 14:02:16
 * @LastEditors: Deng Yifan 553192215@qq.com
 * @LastEditTime: 2022-08-26 16:54:56
 * @FilePath: /dg_templete/README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved. 
-->
# dg_templete

![](https://img.shields.io/badge/License-GNU%20General%20Public%20License%20v3.0-green)

![](https://img.shields.io/badge/Python-3.8-blue)

![](https://img.shields.io/badge/知乎-一个邓-orange)

[博客](https://zhuanlan.zhihu.com/p/552293287)

深度学习项目框架

使用前需要先根据env.sh中的注释修改相应的选项

安装 redis 实现GPU排队，安装 redis 可参考[非Root用户在Linux安装Redis，并允许远程连接此数据库](https://zhuanlan.zhihu.com/p/552627015)

如果有微信、钉钉通知的需要，按照configs/config.yaml中的说明申请对应的Webhook和token

如果要使用Comet.ml实验管理平台，需要申请[API key](https://www.comet.com)，填入到.comet.config中，然后将此文件移到用户根目录下

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
