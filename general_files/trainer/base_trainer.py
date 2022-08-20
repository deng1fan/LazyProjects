from typing import Dict, Union
import pytorch_lightning as pl
from colorama import Fore
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from torch.utils.data import DataLoader
from general_files.utils.common_util import CustomCometLoggerForPL, get_logger
from general_files.utils.data_util import dict_list_to_tensor

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
        if config.stage == 'finetune':
            # 因为框架的特殊性，如果不设置这里模型会从ckpt的断点继续训练
            # 如果想只加载模型权重，需要下面的设置
            config.ckpt_path = None
        self.config = config
        self.model = model
        self.config.dataset_size = len(train_dataset)
        self.model.train_dataset = train_dataset
        self.model.val_dataset = eval_dataset

        class DataModule(pl.LightningDataModule):
            def __init__(self, train_dataset, eval_dataset, collate_fn):
                super().__init__()
                self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
                self.collate_fn = collate_fn

            def train_dataloader(self):
                return DataLoader(self.train_dataset,
                                  batch_size=config.train_batch_size,
                                  shuffle=True,
                                  pin_memory=config.dataloader_pin_memory,
                                  num_workers=config.dataloader_num_workers,
                                  collate_fn=self.collate_fn
                                  )

            def val_dataloader(self):
                return DataLoader(self.eval_dataset,
                                  batch_size=config.train_batch_size,
                                  shuffle=True,
                                  pin_memory=config.dataloader_pin_memory,
                                  num_workers=config.dataloader_num_workers,
                                  collate_fn=self.collate_fn)
        self.data_module = DataModule(train_dataset, eval_dataset, self.collate_fn)

        self.tokenizer = tokenizer
        if experiment:
            logger = CustomCometLoggerForPL()
            logger._experiment = experiment
        else:
            logger = None

        class LiteProgressBar(pl.callbacks.progress.TQDMProgressBar):
            def __init__(self, refresh_rate: int = 1, process_position: int = 0):
                super().__init__(refresh_rate, process_position)

            def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
                items = super().get_metrics(trainer, pl_module)
                # items['ppl'] = round(items['ppl'], 1) if 'ppl' in items else None
                items['lr'] = round(items['lr'], 7) if 'lr' in items else None
                items.pop("v_num", None)
                return items

            def init_train_tqdm(self):
                bar = super().init_train_tqdm()
                bar.bar_format = '%s{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.GREEN, Fore.GREEN)
                return bar

            def init_validation_tqdm(self):
                bar = super().init_validation_tqdm()
                bar.set_description('Validating')
                bar.bar_format = '%s{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.GREEN, Fore.GREEN)
                bar.leave = False
                return bar

        checkpoint_callback = ModelCheckpoint(monitor=config.checkpoint_monitor,
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
        callbacks = [checkpoint_callback,
                     early_stop_callback,
                     LiteProgressBar(),
                     ]

        self.trainer = pl.Trainer(logger=logger,
                                  callbacks=callbacks,
                                  **config.pl_train_args)

    def collate_fn(self, batch):
        return dict_list_to_tensor(batch)

    def train(self):
        if self.config.pl_train_args.auto_lr_find:
            lr_finder = self.trainer.tuner.lr_find(self.model, datamodule=self.data_module)
            # 展示loss和学习率的曲线
            fig = lr_finder.plot(suggest=True)
            fig.show()
            # 设置为推荐的学习率
            self.model.config.lr = lr_finder.suggestion()
        self.trainer.fit(model=self.model,
                         datamodule=self.data_module
                         )
