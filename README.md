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
![](https://img.shields.io/badge/知乎-邓什么邓-orange)


# 本仓库已不再维护，相关功能会逐渐被拆解，融入 Lazydl 项目中

本框架借鉴网站开发中[前后端分离](https://zhuanlan.zhihu.com/p/66711706)的思想，对现有的框架进行接口定义，旨在更加快捷地进行深度学习实验，尤其是基于 [Huggingface](https://huggingface.co/models) 的模型。通过对 [PytorchLightning](https://pytorch-lightning.readthedocs.io/en/latest/) 进行进一步封装，在保证原有框架的灵活性的前提下，加入了更多新的功能：**基于 [Redis](https://redis.io) 的GPU实验排队**、**统一参数接口**、**实验结果自动上传 [Comet.ml](https://www.comet.com)**、**快速 Debug** 以及**批量运行**等多种特性。仅仅需要新增三个文件就可以实现一个新的实验的开发并享有框架的全部功能，同时由于对模型、数据、训练和测试等流程的解耦，可以轻松移植同样使用本框架的相同项目。

_This framework draws on the idea of separation of front-end and backend in website development, and defines interfaces to existing frameworks, aiming to perform deep learning experiments more quickly, especially for Huggingface-based models. By further encapsulating PytorchLightning, it adds more new features while maintaining the flexibility of the original framework: **Redis-based GPU experiment queuing**, **unified parameter interface**, **automatic uploading of experiment results to Comet**, **fast debug**, **batch runs**, and many other features. Only three new files are needed to develop a new experiment and enjoy the full functionality of the framework, and because of the decoupling of the model, data, training and testing processes, you can easily migration the same project using the same framework._

更多框架细节请关注该博客 👉 [一种优雅却不失灵活的深度学习框架理念](https://zhuanlan.zhihu.com/p/552293287)

_More details on the use of the framework can be found in the blog 👉 [一种优雅却不失灵活的深度学习框架理念](https://zhuanlan.zhihu.com/p/552293287)_


## 依赖包 _Requirements_
根据 [env.sh](https://github.com/D-Yifan/AgileLightning/blob/master/env.sh) 中的提示修改适合自己环境的配置，然后启动改脚本进行环境配置

_Follow the prompts in env.sh to modify the configuration to suit your environment, and then start the change script to configure the environment_

    bash env.sh

## 特性支持 _Features_
* 安装 redis 实现GPU排队，安装 redis 可参考[非Root用户在Linux安装Redis，并允许远程连接此数据库](https://zhuanlan.zhihu.com/p/552627015)
* 如果有微信、钉钉通知的需要，按照 [configs/default_config.yaml](https://github.com/D-Yifan/AgileLightning/blob/master/configs/default_config.yaml) 中的说明申请对应的Webhook和token
* 如果要使用Comet.ml实验管理平台，需要申请[API key](https://www.comet.com)，填入到 [.comet.config](https://github.com/D-Yifan/AgileLightning/blob/master/.comet.config) 中，然后将此文件移到用户根目录下

## 数据集
exp_demo 中使用的示例数据集来自 [WizardOfWikipedia](https://parl.ai/projects/wizard_of_wikipedia/)

_The example dataset used in exp_demo is from WizardOfWikipedia_

    # 数据集下载  _Download_
    wget http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
    # 解压缩   _Decompress_
    tar -zxvf wizard_of_wikipedia.tgz

## 运行
环境准备好之后，可以使用 [run.sh](https://github.com/D-Yifan/AgileLightning/blob/master/run.sh) 脚本启动实验

_Once the environment is ready, you can start the experiment using the run.sh script_

    bash run.sh
    
启动好后，你可以看到下面的界面：

_After starting up, you can see the following screen_

![](https://github.com/D-Yifan/AgileLightning/blob/master/figures/start.jpg)

同时你的钉钉还会收到以下通知：

_You will also receive the following notifications on your DingDing_

![](https://github.com/D-Yifan/AgileLightning/blob/master/figures/dingding_noti.jpg)

这说明此时代码程序已经开始准备数据和模型了，在准备完毕后，会根据需要进行 GPU 排队或直接使用 CPU，如果成功占用计算资源，这时钉钉会收到如下通知：

_This means that the code program has started to prepare the data and model at this point, and when it is ready, it will either queue the GPU or use the CPU directly as needed, and if it successfully occupies computing resources, the nail will then receive a notification as follows_

![](https://github.com/D-Yifan/AgileLightning/blob/master/figures/dingding_start_noti.png)

这时，我们的模型就已经在训练了，如果我们想要查看训练过程，可以使用代码启动界面中的命令查看或者登录你的 Comet 查看实验的各种信息：

_At this point, our model is already in training, and if we want to view the training process, we can use the command in the code launch screen to view it or log into your Comet to view various information about the experiment:_

![](https://github.com/D-Yifan/AgileLightning/blob/master/figures/comet.png)

实时的指标曲线：

_Real-time indicator curves._

![](https://github.com/D-Yifan/AgileLightning/blob/master/figures/comet_panal.png)

我们还可以在 Redis 管理界面中（我使用的是[AnotherRedisDesktopManager](https://github.com/qishibo/AnotherRedisDesktopManager)）查看实验的 GPU 占用信息、进程信息、以及排队信息等等

_We can also view the GPU usage information, process information, and queuing information for the experiment in the Redis management interface (I'm using [AnotherRedisDesktopManager](https://github.com/qishibo/AnotherRedisDesktopManager)) etc_

![](https://github.com/D-Yifan/AgileLightning/blob/master/figures/redis_window.png)



    
# 启动redis维护

    bash redis.sh
