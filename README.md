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
# AgileLightning

![](https://img.shields.io/badge/License-GNU%20General%20Public%20License%20v3.0-green)
![](https://img.shields.io/badge/Python-3.8-blue)
![](https://img.shields.io/badge/知乎-一个邓-orange)

_文档不断完善中，欢迎大家提出宝贵意见_

_The document is under continuous improvement, welcome your valuable comments_

本框架借鉴网站开发中前后端分离的思想，对现有的框架进行接口定义，旨在更加快捷地进行深度学习实验，尤其是基于 Huggingface 的模型。通过对 PytorchLightning 进行进一步封装，在保证原有框架的灵活性的前提下，加入了更多新的功能：**基于 Redis 的GPU实验排队**、**统一参数接口**、**实验结果自动上传 Comet**、**快速 Debug** 以及**批量运行**等多种特性。仅仅需要新增三个文件就可以实现一个新的实验的开发并享有框架的全部功能，同时由于对模型、数据、训练和测试等流程的解耦，可以轻松移植同样使用本框架的相同项目。

_This framework draws on the idea of separation of front-end and backend in website development, and defines interfaces to existing frameworks, aiming to perform deep learning experiments more quickly, especially for Huggingface-based models. By further encapsulating PytorchLightning, it adds more new features while maintaining the flexibility of the original framework: **Redis-based GPU experiment queuing**, **unified parameter interface**, **automatic uploading of experiment results to Comet**, **fast debug**, **batch runs**, and many other features. Only three new files are needed to develop a new experiment and enjoy the full functionality of the framework, and because of the decoupling of the model, data, training and testing processes, you can easily migration the same project using the same framework._

更多框架细节请关注该博客 👉 [一种优雅却不失灵活的深度学习框架理念](https://zhuanlan.zhihu.com/p/552293287)

_More details on the use of the framework can be found in the blog 👉 [一种优雅却不失灵活的深度学习框架理念](https://zhuanlan.zhihu.com/p/552293287)_


# 依赖包 _Requirements_
根据 env.sh 中的提示修改适合自己环境的配置，然后启动改脚本进行环境配置

_Follow the prompts in env.sh to modify the configuration to suit your environment, and then start the change script to configure the environment_

    bash env.sh

# 特性支持 _Features_
* 安装 redis 实现GPU排队，安装 redis 可参考[非Root用户在Linux安装Redis，并允许远程连接此数据库](https://zhuanlan.zhihu.com/p/552627015)
* 如果有微信、钉钉通知的需要，按照configs/config.yaml中的说明申请对应的Webhook和token
* 如果要使用Comet.ml实验管理平台，需要申请[API key](https://www.comet.com)，填入到.comet.config中，然后将此文件移到用户根目录下

# 数据集

    # 数据集下载
    wget http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
    # 解压缩 
    tar -zxvf wizard_of_wikipedia.tgz

# 运行

    bash run.sh
    
# 启动redis维护

    bash redis.sh
