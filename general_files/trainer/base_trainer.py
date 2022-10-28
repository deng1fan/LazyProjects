'''
Author: D-Yifan 553192215@qq.com
Date: 2022-08-17 13:04:55
LastEditors: D-Yifan https://github.com/D-Yifan
LastEditTime: 2022-10-18 13:08:06
FilePath: /code_frame/general_files/trainer/base_trainer.py
Description: 

Copyright (c) 2022 by D-Yifan 553192215@qq.com, All Rights Reserved. 
'''
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from general_files.utils.common_util import (
    CustomCometLoggerForPL,
    get_logger,
    LiteProgressBar,
)
from general_files.utils.data_util import dict_list_to_tensor, DataModule

log = get_logger(__name__)


class ModelTrainer:
    def __init__(
        self,
        config,
        model,
        train_dataset,
        eval_dataset,
        tokenizer,
        experiment,
    ):
        if config.stage != "finetune":
            # 因为框架的特殊性，如果不设置这里模型会从ckpt的断点继续训练
            # 如果想只加载模型权重，需要下面的设置
            config.ckpt_path = None
        self.config = config
        self.model = model
        self.config.dataset_size = len(train_dataset)
        self.model.train_dataset = train_dataset
        self.model.val_dataset = eval_dataset

        self.data_module = DataModule(
            train_dataset, eval_dataset, self.collate_fn, config
        )

        self.tokenizer = tokenizer
        if experiment:
            logger = CustomCometLoggerForPL()
            logger._experiment = experiment
        else:
            logger = None

        checkpoint_callback = ModelCheckpoint(
            monitor=config.checkpoint_monitor,
            mode=config.checkpoint_monitr_mode,
            save_top_k=config.save_total_limit,
            save_last=True,
            verbose=True,
            dirpath=config.result_path,
            filename="best_model",
            auto_insert_metric_name=False,
        )

        early_stop_callback = EarlyStopping(
            monitor=config.checkpoint_monitor,
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode=config.checkpoint_monitr_mode,
        )

        # Explicitly specify the process group backend if you choose to
        callbacks = [
            early_stop_callback,
            LiteProgressBar(),
        ]
        if config.stage != "fast_run":
            callbacks.append(checkpoint_callback)
        
        if config.get("use_swa"):
            callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))

        if config.get("use_swa"):
            callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))

        self.trainer = pl.Trainer(
            logger=logger, callbacks=callbacks, **config.pl_train_args
        )

    def collate_fn(self, batch):
        return dict_list_to_tensor(batch)

    def train(self):
        if self.config.pl_train_args.auto_lr_find:
            lr_finder = self.trainer.tuner.lr_find(
                self.model, datamodule=self.data_module
            )
            # 展示loss和学习率的曲线
            fig = lr_finder.plot(suggest=True)
            fig.show()
            # 设置为推荐的学习率
            self.model.config.lr = lr_finder.suggestion()
        self.trainer.fit(model=self.model, datamodule=self.data_module)
