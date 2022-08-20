import logging
import random
import re
import warnings
from typing import Sequence
import numpy as np
import pandas as pd
import rich.syntax
import rich.tree
import torch
import transformers
from omegaconf import DictConfig, OmegaConf, ListConfig
from pytorch_lightning.utilities import rank_zero_only
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.table import Column
from rich.table import Table
from general_files.utils.data_util import save_as, pp
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, List
from pytorch_lightning.loggers import CometLogger
import sys
from nvitop import Device, GpuProcess, NA, colored
from redis import Redis
from typing import List
import os
import datetime
import traceback
import functools
import json
import socket
import time
import hmac
import hashlib
import base64
import urllib
import requests
from nvitop import select_devices

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        # log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # disable adding new keys to config
    # OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "root_dir",
                "work_dir",
                "data_path",
                "pl_train_args",
                "seed",
                "fast_run",
                "pretrain_model",
                "use_gpu",
                "visible_cuda",
                "default_device",
                "task_full_name",
                "model_processor",
                "dataset_processor",
                "trainer_processor",
                "stage",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "cyan"
    tree = rich.tree.Tree("CONFIG", style=style, highlight=True, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)

    # with open(f"config_tree.txt", "w") as fp:
    #     rich.print(tree, file=fp)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def check_config(config):
    config.cache_dir = config.cache_dir + config.pretrain_model.split(':')[-1]
    config.run_name = f'<{config.dataset}><{config.pretrain_model}>'
    config.cache_dataset_path = config.cache_dataset_path + f"<{config.dataset}><{config.pretrain_model}><{config.run_notes}>"
    if config.get('fast_run') and config.get('stage') != 'test':
        # 快速运行整个训练和测试流程，便于查找bug
        config.run_name = config.run_name + f'__fast_run__'
        config.pl_train_args.auto_lr_find = False
    if config.stage == 'test' and config.eval_bad_case_analysis:
        config.dataset_part = ['valid', 'test']
    if config.stage in ['train', 'pretrain']:
        config.ckpt_path = None
    config.comet_name = f"<{config.stage}><{config.run_notes}>{config.run_name}"
    config.task_full_name = f"<{config.stage}><{config.run_notes}><{config.generate_method}>{config.run_name}{config.base_identifier_str}"
    # 设置cuda
    if not config.use_gpu:
        # 不使用gpu
        config.default_device = 'cpu'
        config.want_gpu_num = 0
        config.pl_train_args.gpus = 0
        config.visible_cuda = None
    else:
        # 使用gpu
        if config.wait_gpus:
            config.want_gpu_num = int(config.visible_cuda.split('auto_select_')[-1]) if 'auto_select_' in str(config.visible_cuda) else len(config.visible_cuda)
            config.default_device = f'cpu'
        else:
            gpus = config.visible_cuda.split(',') if ',' in str(config.visible_cuda) else config.visible_cuda
            if isinstance(gpus, int):
                config.want_gpu_num = 1
                config.default_device = f'cuda:{gpus}'
            else:
                config.want_gpu_num = len(gpus)
                config.default_device = f'cuda:{gpus[0]}'
    return config


def get_parent_dir(path=None, offset=-1):
    result = path if path else __file__
    for i in range(abs(offset)):
        result = os.path.dirname(result)
    return result

class MyProgressCallback(transformers.TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_bar = None
        self.prediction_bar = None
        self.progress = None
        self.train_loss = 0
        self.valid_loss = 0
        self.training_bar_state = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.progress, self.training_bar = get_progress_bar('Train', total_step=state.max_steps)
            self.progress.start()  ## 开启
        self.current_step = 0

    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and not control.should_evaluate:
            if self.prediction_bar != None:
                self.progress.remove_task(self.prediction_bar)
                self.prediction_bar = None
            if self.training_bar is None:
                self.training_bar = self.progress.add_task(f"[green]Train",
                                                           total=self.training_bar_state.total,
                                                           completed=self.training_bar_state.completed,
                                                           **self.training_bar_state.fields,
                                                           )
            self.progress.update(self.training_bar, advance=state.global_step - self.current_step, visible=True,
                                 refresh=True,
                                 loss=self.train_loss)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero and transformers.trainer_utils.has_length(eval_dataloader.dataset):
            if self.training_bar is not None:
                tasks = self.progress.tasks
                for task in tasks:
                    if task.id == self.training_bar:
                        self.training_bar_state = task
                        break
                self.progress.remove_task(self.training_bar)
                self.training_bar = None
            if self.prediction_bar is None:
                if self.progress is not None:
                    self.prediction_bar = self.progress.add_task(f"[red]Predict", total=len(eval_dataloader),
                                                                 loss='???')
                else:
                    self.progress, self.prediction_bar = get_progress_bar('Predict', total_step=state.max_steps)
            self.progress.update(self.prediction_bar, advance=1, refresh=True, loss=self.valid_loss)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar != None:
                # print()
                self.progress.remove_task(self.prediction_bar)
                self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            _ = logs.pop("total_flos", None)
            if logs is not None and 'loss' in logs:
                if control.should_evaluate:
                    self.valid_loss = logs['loss']
                else:
                    self.train_loss = logs['loss']

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.training_bar is not None:
                self.progress.remove_task(self.training_bar)
            if self.prediction_bar is not None:
                self.progress.remove_task(self.prediction_bar)
            self.progress.stop()
            self.training_bar = None
            self.prediction_bar = None


def get_progress_bar(task_name, total_step):
    job_progress = Progress(
        TextColumn("[bold bright_green]{task.description}"),
        SpinnerColumn(),
        BarColumn(bar_width=None,
                  table_column=Column(ratio=1),
                  style='red',
                  complete_style='green',
                  finished_style='green_yellow',
                  pulse_style='green_yellow'),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        "·",
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        TextColumn("· [bright_yellow]{task.completed}[bright_black]/[turquoise2]{task.total}"),
        TextColumn("· [bold bright_red]loss:{task.fields[loss]}"),
        expand=True,
        transient=True,
        refresh_per_second=1
    )
    progress_bar = job_progress.add_task(f"[green]{task_name}", total=total_step, loss='???')
    return job_progress, progress_bar

@rank_zero_only
def print_parameters(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _dict = {}
    for _, param in enumerate(model.named_parameters()):
        total_params = param[1].numel()
        k = param[0].split('.')[0]
        if k in _dict.keys():
            _dict[k] += total_params
        else:
            _dict[k] = 0
            _dict[k] += total_params
    # 打印可训练参数大小
    _dict['total_parameters'] = total_num
    _dict['trainable_parameters'] = trainable_num
    console = Console(color_system='256', style="cyan")
    table = Table(style="cyan",
                  show_footer=False,
                  title="[bold green]Model Parameters", )
    table.add_column('Layers :dizzy:', justify="right", style="magenta")
    table.add_column('Parameters(MB)', justify="left", style="magenta")
    for k, v in _dict.items():
        table.add_row(k, str(round(v / (1024 * 1024), 4)))
    console.print(table, justify='center')

@rank_zero_only
def print_dict_to_table(input_dict, column1_name, column2_name, title, config):
    console = Console(color_system='256', style="cyan")
    table = Table(style="cyan",
                  show_footer=False,
                  title=title)
    table.add_column(column1_name, justify="right", style="magenta")
    table.add_column(column2_name, justify="left", style="magenta")
    for k, v in input_dict.items():
        table.add_row(k, str(v))
    console.print(table)
    # 去除rich的格式修饰符
    save_title_name = re.sub(r'\[.*\]', '', title)
    with open(f"{config.result_path}/{save_title_name}.txt", "w") as fp:
        rich.print(table, file=fp)

@rank_zero_only
def print_generated_dialogs(test_output, experiment, show_num=5, mode='dial', config=None):
    save_path = config.ckpt_path if config.ckpt_path is not None else config.result_path
    console = Console(color_system='256', style="cyan")
    columns = []
    save_columns = []
    ignore_columns = ['input_ids', 'labels', '__index_level_0__']
    features = dict()
    for column in test_output.column_names:
        if column not in ignore_columns and 'decoder_' not in column and '_id' not in column:
            save_columns.append(column)
            columns.append(column)
            features[column] = test_output[column]
    for i in range(min(show_num, len(features['source']))):
        console.print("[bold]···········································································",
                      justify='center')
        console.print(f"[bold green]Generated Example {i}", justify='center')
        for k in features:
            console.print(f"[bold red]>>>> [bold orange1]{k} [bold red]<<<<", justify='left')
            console.print("[bold cyan]" + str(features[k][i]), justify='left')
        console.print("[bold]···········································································",
                      justify='center')

    if save_path:
        test_output_df = pd.DataFrame(test_output)
        test_output_df = test_output_df.loc[:, save_columns]
        if '.ckpt' in save_path:
            save_path = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        test_output_df.to_excel(save_path + '/test_output.xlsx')
        generated = [str(s) + '\n' for s in test_output['generated_seqs']]
        generated_with_special_tokens = [str(s) + '\n' for s in test_output['generated_seqs_with_special_tokens']]
        save_as(generated, save_path + '/generated_' + mode, data_name='generated_' + mode, file_format='txt')
        save_as(generated_with_special_tokens, save_path + '/generated_with_special_tokens_' + mode, data_name='generated_with_special_tokens_' + mode, file_format='txt')
        if experiment and config.logger == 'comet':
            features_df = pd.DataFrame(features)
            experiment.log_table(tabular_data=features_df, filename='generated_' + mode + '.csv')
            experiment.log_asset(save_path + '/generated_' + mode + '.txt', file_name='generated_' + mode)
            experiment.log_asset(save_path + '/generated_with_special_tokens_' + mode + '.txt', file_name='generated_with_special_tokens_' + mode)
            log.info(f'已将生成结果:generated_{mode}、generated_with_special_tokens_{mode}保存到comet!')
            ###############################################
            # 推送到钉钉
            ###############################################
            run_name = config.task_full_name.replace('/', '--')
            send_msg_to_DingTalk_and_wx("正在上传生成结果！！！🎉🎉🎉", config)
            send_file_to_DingTalk(save_path + '/test_output.xlsx', f"生成结果__{run_name}.xlsx")
            send_file_to_DingTalk(save_path + '/generated_' + mode + '.txt', f"生成句子__{run_name}.txt")
            send_file_to_DingTalk(save_path + '/generated_with_special_tokens_' + mode + '.txt', f"带特殊符的生成句子__{run_name}.txt")


def switch_color(color=None):
    if color is None:
        return '[green]'
    if color == '[green]':
        return '[red]'
    if color == '[red]':
        return '[green]'


class Result(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __init__(self, **kwargs):
        super(Result, self).__init__()
        # self['msg'] = msg
        # self['status'] = True
        self.add(**kwargs)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def delete(self, keys):
        for k in keys:
            self.pop(k)

    def merge(self, merge_dict):
        if not isinstance(merge_dict, Result):
            raise TypeError('不支持的合并类型')
        for k, v in merge_dict.items():
            if k in ['msg', 'status'] or k in self:
                continue
            self[k] = v

    def merge_or_update(self, merge_dict):
        if not isinstance(merge_dict, Result) and not isinstance(merge_dict, dict):
            raise TypeError('不支持的合并类型')
        for k, v in merge_dict.items():
            if k in ['msg', 'status']:
                continue
            self[k] = v

    @staticmethod
    def create_error_msg_result(msg='Error Result', **kwargs):
        result = Result()
        result['msg'] = msg
        result['status'] = False
        result.add(**kwargs)
        return result

    def get(self, name):
        if name is None:
            return list(self.values())
        elif isinstance(name, str):
            return self[name]
        elif isinstance(name, list):
            values = [self[n] for n in name]
            return values
        else:
            return self.create_error_msg_result(msg=f'Key值类型{type(name)}不支持')

    def print(self, name=None):
        pp('  =====' + self['msg'] + '=====')
        values = self.get(name)
        if name is None:
            name = list(self.keys())
        for i, k in enumerate(name):
            v = values[i]
            pp(f'  {k}:    {v}')
        pp('  =====' + self['msg'] + '=====')

    def flatten_to_print(self):
        value_str = ""
        keys = self.keys()
        for i, k in enumerate(keys):
            v = self[k]
            value_str = value_str + k + " : " + str(v) + "\n\n"
        return value_str

    def append_values(self, next_dict):
        if not isinstance(next_dict, Result) and not isinstance(next_dict, dict):
            raise TypeError('不支持的合并类型')
        for key in next_dict.keys():
            if key not in self.keys():
                self[key] = []

            self[key].append(next_dict[key]) if isinstance(self[key], list) else [self[key]].append(next_dict[key])


class CustomCometLoggerForPL(CometLogger):
    def __init__(self):
        super(CustomCometLoggerForPL, self).__init__()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        r"""
        When calling ``self.experiment.end()``, that experiment won't log any more data to Comet.
        That's why, if you need to log any more data, you need to create an ExistingCometExperiment.
        For example, to log data when testing your model after training, because when training is
        finalized :meth:`CometLogger.finalize` is called.

        This happens automatically in the :meth:`~CometLogger.experiment` property, when
        ``self._experiment`` is set to ``None``, i.e. ``self.reset_experiment()``.
        """
        # self.experiment.end()
        # self.reset_experiment()

def dingtalk_sender_and_wx(webhook_url: str,
                           user_mentions: List[str] = [],
                           secret: str = '',
                           keywords: List[str] = []):
    """
    DingTalk sender wrapper: execute func, send a DingTalk notification with the end status
    (sucessfully finished or crashed) at the end. Also send a DingTalk notification before
    executing func.

    `webhook_url`: str
        The webhook URL to access your DingTalk chatroom.
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for more details.
    `user_mentions`: List[str] (default=[])
        Optional users phone number to notify.
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for more details.
    `secret`: str (default='')
        DingTalk chatroom robot are set with at least one of those three security methods
        (ip / keyword / secret), the chatroom will only accect messages that:
            are from authorized ips set by user (ip),
            contain any keyword set by user (keyword),
            are posted through a encrypting way (secret).
        Vist https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq from more details.
    `keywords`: List[str] (default=[])
        see `secret`

    """
    msg_template = {
        "msgtype": "text",
        "text": {
            "content": ""
        },
        "at": {
            "atMobiles": user_mentions,
            "isAtAll": False
        }
    }

    def _construct_encrypted_url():
        '''
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for details
        '''
        timestamp = round(datetime.datetime.now().timestamp() * 1000)
        secret_enc = secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        encrypted_url = webhook_url + '&timestamp={}'.format(timestamp) \
                        + '&sign={}'.format(sign)
        return encrypted_url

    def decorator_sender(func):
        @functools.wraps(func)
        def wrapper_sender(*args, **kwargs):

            start_time = datetime.datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__

            # Handling distributed training edge case.
            # In PyTorch, the launch of `torch.distributed.launch` sets up a RANK environment variable for each process.
            # This can be used to detect the master process.
            # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py#L211
            # Except for errors, only the master process will send notifications.
            if 'RANK' in os.environ:
                master_process = (int(os.environ['RANK']) == 0)
                host_name += ' - RANK: %s' % os.environ['RANK']
            else:
                master_process = True
            visible_cuda = str(args[0]['visible_cuda'])
            config = args[0]
            run_name = args[0]['run_name']
            run_notes = args[0]['run_notes']
            if master_process:
                contents = ['训练准备就绪，即将开始 🎬\n',
                            '机器名: %s\n' % host_name,
                            '使用显卡序号: %s\n' % visible_cuda,
                            '进程ID: %s\n' % str(os.getpid()),
                            '开始时间: %s\n' % start_time.strftime(DATE_FORMAT),
                            'run_name: %s\n' % run_name,
                            'run_notes: %s\n' % run_notes,
                            ]

                wx_contents = contents
                contents.extend(['@{}'.format(i) for i in user_mentions])
                contents.extend(keywords)

                msg_template['text']['content'] = '\n'.join(contents)
                if secret:
                    postto = _construct_encrypted_url()
                    requests.post(postto, json=msg_template)
                else:
                    requests.post(webhook_url, json=msg_template)
                # send_wechat(config.run_notes, '\n'.join(wx_contents))

            try:
                value = func(*args, **kwargs)

                if master_process:
                    end_time = datetime.datetime.now()
                    elapsed_time = end_time - start_time
                    contents = ["训练已经完成！！！ 🎉\n",
                                '机器名: %s\n' % host_name,
                                '使用显卡序号: %s\n' % visible_cuda,
                                '进程ID: %s\n' % str(os.getpid()),
                                '开始时间: %s\n' % start_time.strftime(DATE_FORMAT),
                                '结束时间: %s\n' % end_time.strftime(DATE_FORMAT),
                                '训练时长: %s\n' % str(elapsed_time)]

                    try:
                        str_value = "\n\n" + value.flatten_to_print()
                        contents.append('=====运行信息===== %s' % str_value)
                    except:
                        contents.append('=====运行信息=====\n %s' % "ERROR - Couldn't str the returned value.")

                    wx_contents = contents

                    contents.extend(['@{}'.format(i) for i in user_mentions])
                    contents.extend(keywords)

                    msg_template['text']['content'] = '\n'.join(contents)
                    if secret:
                        postto = _construct_encrypted_url()
                        requests.post(postto, json=msg_template)
                    else:
                        requests.post(webhook_url, json=msg_template)
                        pp(msg_template)
                send_wechat(config.run_notes, '\n'.join(wx_contents))
                return value

            except Exception as ex:
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                contents = ["啊哦！训练遇到了一点问题 ☠️",
                            '机器名: %s' % host_name,
                            '使用显卡序号: %s' % visible_cuda,
                            '进程ID: %s\n' % str(os.getpid()),
                            '开始时间: %s' % start_time.strftime(DATE_FORMAT),
                            '崩溃时间: %s' % end_time.strftime(DATE_FORMAT),
                            '用时: %s\n\n' % str(elapsed_time),
                            "错误信息:",
                            '%s\n\n' % ex,
                            "错误回溯:",
                            '%s\n\n' % traceback.format_exc(),
                            'run_name: %s\n' % run_name,
                            'run_notes: %s\n' % run_notes,
                            ]
                wx_contents = contents

                contents.extend(['@{}'.format(i) for i in user_mentions])
                contents.extend(keywords)
                ###############################################
                # 修改comet状态
                ###############################################
                experiment = args[-1]
                if config.logger == 'comet':
                    experiment.add_tag('Crashed')
                    experiment.set_name(config.comet_name + '  Error!')
                    experiment.end()

                msg_template['text']['content'] = '\n'.join(contents)
                if secret:
                    postto = _construct_encrypted_url()
                    requests.post(postto, json=msg_template)
                else:
                    requests.post(webhook_url, json=msg_template)
                    pp(msg_template)
                send_wechat(config.run_notes, '\n'.join(wx_contents))
                raise ex

        return wrapper_sender

    return decorator_sender


def send_msg_to_DingTalk_and_wx(msg, config):
    """
    DingTalk sender wrapper: execute func, send a DingTalk notification with the end status
    (sucessfully finished or crashed) at the end. Also send a DingTalk notification before
    executing func.

    `webhook_url`: str
        The webhook URL to access your DingTalk chatroom.
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for more details.
    `user_mentions`: List[str] (default=[])
        Optional users phone number to notify.
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for more details.
    `secret`: str (default='')
        DingTalk chatroom robot are set with at least one of those three security methods
        (ip / keyword / secret), the chatroom will only accect messages that:
            are from authorized ips set by user (ip),
            contain any keyword set by user (keyword),
            are posted through a encrypting way (secret).
        Vist https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq from more details.
    `keywords`: List[str] (default=[])
        see `secret`

    """
    # todo 替换成自己的钉钉webhook
    webhook_url = "xxx"
    secret = "xxxx"
    user_mentions = []
    msg_template = {
        "msgtype": "text",
        "text": {
            "content": ""
        },
        "at": {
            "atMobiles": user_mentions,
            "isAtAll": False
        }
    }

    def _construct_encrypted_url():
        '''
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for details
        '''
        timestamp = round(datetime.datetime.now().timestamp() * 1000)
        secret_enc = secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        encrypted_url = webhook_url + '&timestamp={}'.format(timestamp) \
                        + '&sign={}'.format(sign)
        return encrypted_url


    start_time = datetime.datetime.now()
    host_name = socket.gethostname()

    # Handling distributed training edge case.
    # In PyTorch, the launch of `torch.distributed.launch` sets up a RANK environment variable for each process.
    # This can be used to detect the master process.
    # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py#L211
    # Except for errors, only the master process will send notifications.
    if 'RANK' in os.environ:
        master_process = (int(os.environ['RANK']) == 0)
        host_name += ' - RANK: %s' % os.environ['RANK']
    else:
        master_process = True
    visible_cuda = str(config['visible_cuda'])

    try:
        if master_process:
            contents = [str(msg) + '\n',
                        '机器名: %s\n' % host_name,
                        '使用显卡序号: %s\n' % visible_cuda,
                        '进程ID: %s\n' % str(os.getpid()),
                        ]

            try:
                config_info = Result(
                    run_name=config.run_name,
                    run_notes=config.run_notes,)
                str_value = "\n\n" + config_info.flatten_to_print()
                contents.append('=====运行信息===== %s' % str_value)
            except:
                contents.append('=====运行信息=====\n %s' % "ERROR - Couldn't str the returned value.")

            wx_contents = contents

            contents.extend(['@{}'.format(i) for i in user_mentions])

            msg_template['text']['content'] = '\n'.join(contents)
            if secret:
                postto = _construct_encrypted_url()
                requests.post(postto, json=msg_template)
            else:
                requests.post(webhook_url, json=msg_template)
                pp(msg_template)
        # send_wechat(config.run_notes, '\n'.join(wx_contents))
        return msg

    except Exception as ex:
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        contents = ["啊哦！推送遇到了一点问题 ☠️",
                    '机器名: %s' % host_name,
                    '使用显卡序号: %s' % visible_cuda,
                    '进程ID: %s\n' % str(os.getpid()),
                    '开始时间: %s' % start_time.strftime(DATE_FORMAT),
                    '崩溃时间: %s' % end_time.strftime(DATE_FORMAT),
                    '用时: %s\n\n' % str(elapsed_time),
                    "错误信息:",
                    '%s\n\n' % ex,
                    "错误回溯:",
                    '%s\n\n' % traceback.format_exc()]

        wx_contents = contents

        contents.extend(['@{}'.format(i) for i in user_mentions])

        try:
            config_info = Result(
                run_name=config.run_name,
                run_notes=config.run_notes, )
            str_value = "\n\n" + config_info.flatten_to_print()
            contents.append('=====运行信息===== %s' % str_value)
        except:
            contents.append('=====运行信息=====\n %s' % "ERROR - Couldn't str the returned value.")

        msg_template['text']['content'] = '\n'.join(contents)
        if secret:
            postto = _construct_encrypted_url()
            requests.post(postto, json=msg_template)
        else:
            requests.post(webhook_url, json=msg_template)
            pp(msg_template)
        send_wechat(config.run_notes, '\n'.join(wx_contents))
        raise ex


def send_file_to_DingTalk(file_path, file_name):
    def getAccess_token():
        appkey = 'xxxxx'
        # todo 替换自己的key
        appsecret = 'xxxxx'  # 替换自己的secret
        url = 'https://oapi.dingtalk.com/gettoken?appkey=%s&appsecret=%s' % (appkey, appsecret)
        headers = {'Content-Type': "application/x-www-form-urlencoded"}
        data = {'appkey': appkey, 'appsecret': appsecret}
        r = requests.request('GET', url, data=data, headers=headers)
        access_token = r.json()["access_token"]
        return access_token

    def getMedia_id(file_path, file_name):
        access_token = getAccess_token()  # 拿到接口凭证 #获取要推送文件的路径 path os.getcwd()
        file = os.path.join(file_path)  # path='./helloworld.txt'#文件地址
        url = 'https://oapi.dingtalk.com/media/upload?access_token=%s&type=file' % access_token
        files = {'media': (file_name, open(file, 'rb'))}
        data = {'access_token': access_token,
         'type': 'file'}
        response = requests.post(url, files=files, data=data)
        json = response.json()
        return json["media_id"]


    access_token = getAccess_token()
    media_id = getMedia_id(file_path, file_name)
    url = 'https://oapi.dingtalk.com/chat/send?access_token=' + access_token
    header = {
        'Content-Type': 'application/json'
    }
    data = {'access_token': access_token,
            # todo 替换自己的群id
            'chatid': 'xxxx',
            'msg': {
                'msgtype': 'file',
                'file': {'media_id': media_id}
            }}
    response = requests.request('POST', url, data=json.dumps(data), headers=header)
    if response.ok:
        log.info(f'已成功推送文件-->{file_name} 到钉钉！')
    else:
        log.info(f'推送文件-->{file_name}到钉钉失败：{response.text}！')

def send_wechat(title, msg):
    # todo 替换自己的token 申请地址：https://www.pushplus.plus
    token = 'xxxx'
    title = title
    content = msg
    template = 'txt'
    url = f"https://www.pushplus.plus/send?token={token}&title={title}&content={content}&template={template}"
    requests.get(url=url)

def print_gpu_info(gpus):
    devices = Device.cuda.from_cuda_indices(gpus)  # or `Device.all()` to use NVML ordinal instead
    separator = False
    for device in devices:
        processes = device.processes()  # type: Dict[int, GpuProcess]
        print(colored(str(device), color='green', attrs=('bold',)))
        print(colored('  - GPU physical index: ', color='blue', attrs=('bold',)) + f'{device.physical_index}')
        print(colored('  - GPU utilization: ', color='blue', attrs=('bold',)) + f'{device.gpu_utilization()}%')
        print(colored('  - Total memory:    ', color='blue', attrs=('bold',)) + f'{device.memory_total_human()}')
        print(colored('  - Used memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_used_human()}')
        print(colored('  - Free memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_free_human()}')

        if len(processes) > 0:
            processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
            processes.sort(key=lambda process: (process.username, process.pid))

            print(colored(f'  - Processes ({len(processes)}):', color='blue', attrs=('bold',)))
            fmt = '    {pid:<5}  {username:<8} {cpu:>5}  {host_memory:>8} {time:>8}  {gpu_memory:>8}  {sm:>3}  {command:<}'.format
            print(colored(fmt(pid='PID', username='USERNAME',
                              cpu='CPU%', host_memory='HOST-MEM', time='TIME',
                              gpu_memory='GPU-MEM', sm='SM%',
                              command='COMMAND'),
                          attrs=('bold',)))
            for snapshot in processes:
                print(fmt(pid=snapshot.pid,
                          username=snapshot.username[:7] + (
                              '+' if len(snapshot.username) > 8 else snapshot.username[7:8]),
                          cpu=snapshot.cpu_percent, host_memory=snapshot.host_memory_human,
                          time=snapshot.running_time_human,
                          gpu_memory=(snapshot.gpu_memory_human if snapshot.gpu_memory_human is not NA else 'WDDM:N/A'),
                          sm=snapshot.gpu_sm_utilization,
                          command=snapshot.command))
        else:
            print(colored('  - No Running Processes', attrs=('bold',)))
        if separator:
            print('-' * 120)
        separator = True

def set_config_gpus(config):
    redis_client = RedisClient()
    if config.use_gpu and isinstance(config.visible_cuda, str) and 'auto_select_' in config.visible_cuda:
        # 如果是自动选择GPU
        min_count = int(config.visible_cuda.split('auto_select_')[-1])
        gpus = select_devices(format='index', min_count=min_count,
                              min_free_memory=config.cuda_min_free_memory,
                              max_memory_utilization=config.cuda_max_memory_utilization)
        self_occupied_gpus = redis_client.get_self_occupied_gpus()
        available_gpus = list(set(gpus) - self_occupied_gpus)
        if len(available_gpus) > 0 and len(available_gpus) >= min_count:
            # 有足够可用GPU
            config.wait_gpus = False
            config.visible_cuda = available_gpus[:min_count]
            config.want_gpu_num = len(available_gpus)
            if isinstance(available_gpus, int):
                config.want_gpu_num = 1
                config.default_device = f'cuda:{available_gpus}'
            else:
                config.want_gpu_num = len(available_gpus)
                config.default_device = f'cuda:{available_gpus[0]}'
            redis_client.register_gpus(config)
            log.info(f"自动选择GPU：{str(config.visible_cuda)}")
        else:
            # 可用GPU不足
            if config.wait_gpus:
                # 排队
                config.task_id = redis_client.join_wait_queue(config)
            else:
                # 不排队
                raise Exception("可用GPU数量不足，建议使用排队功能！")
    elif config.use_gpu:
        # 如果指定了GPU
        reserve_gpus = config.visible_cuda
        min_count = len(reserve_gpus)
        self_occupied_gpus = redis_client.get_self_occupied_gpus()
        gpu_all_free = True
        for gpu in reserve_gpus:
            if gpu in self_occupied_gpus:
                gpu_all_free = False
        if not config.wait_gpus and not gpu_all_free:
            raise Exception("指定GPU并未全部空闲，建议使用排队功能！")
        elif gpu_all_free:
            available_gpus = reserve_gpus
            config.wait_gpus = False
            config.visible_cuda = available_gpus[:min_count]
            config.want_gpu_num = len(available_gpus)
            if isinstance(available_gpus, int):
                config.want_gpu_num = 1
                config.default_device = f'cuda:{available_gpus}'
            else:
                config.want_gpu_num = len(available_gpus)
                config.default_device = f'cuda:{available_gpus[0]}'
            redis_client.register_gpus(config)
        else:
            # 排队
            config.task_id = redis_client.join_wait_queue(config)
    else:
        # 使用CPU
        pass

    ###############################################
    # 检查是否需要等待Gpu
    ###############################################
    while config.use_gpu and config.wait_gpus:
        # 判断当前是否轮到自己
        if redis_client.is_my_turn(config):
            # 循环获取当前可用Gpu
            try:
                min_count = config.want_gpu_num
                gpus = select_devices(format='index', min_count=min_count,
                                      min_free_memory=config.cuda_min_free_memory,max_memory_utilization=config.cuda_max_memory_utilization)
                self_occupied_gpus = redis_client.get_self_occupied_gpus()
                if not isinstance(config.visible_cuda, str):
                    # 如果指定了GPU
                    reserve_gpus = config.visible_cuda
                    gpu_all_free = True
                    for gpu in reserve_gpus:
                        if gpu in self_occupied_gpus:
                            gpu_all_free = False
                    if gpu_all_free:
                        available_gpus = reserve_gpus
                    else:
                        available_gpus = []
                    min_count = len(reserve_gpus)
                else:
                    # 自动选择
                    available_gpus = list(set(gpus) - self_occupied_gpus)

                if len(available_gpus) > 0 and len(available_gpus) >= min_count:
                    # 自动选择，确认等待
                    if config.confirm_gpu_free and config.last_confirm_gpus == available_gpus[:min_count]:
                        # 如果满足条件退出循环
                        log.info("发现足够可用GPU并二次确认成功！")
                        config.wait_gpus = False
                        config.visible_cuda = available_gpus[:min_count]
                        config.want_gpu_num = len(available_gpus)
                        if isinstance(available_gpus, int):
                            config.want_gpu_num = 1
                            config.default_device = f'cuda:{available_gpus}'
                        else:
                            config.want_gpu_num = len(available_gpus)
                            config.default_device = f'cuda:{available_gpus[0]}'
                        redis_client.pop_wait_queue(config)
                        redis_client.register_gpus(config)
                        break
                    else:
                        # 设置单次确认空闲
                        log.info("发现足够可用GPU！即将进行二次确认！")
                        config.confirm_gpu_free = True
                        config.last_confirm_gpus = available_gpus[:min_count]
                        redis_client.update_queue(config)
                        time.sleep(30)
                        continue
                # 重置确认信息
                log.info("当前无足够可用GPU，继续等待......")
                if config.confirm_gpu_free:
                    log.info("二次确认失败，继续等待......")
                config.confirm_gpu_free = False
                config.last_confirm_gpus = []
                redis_client.update_queue(config)
                time.sleep(30)
            except Exception as e:
                print_error_info(e)
                raise e
        else:
            # 排队ing......
            wait_num = len(redis_client.client.lrange('wait_queue', 0, -1)) - 1
            log.info(f"正在排队中！ 前方还有 {wait_num} 个训练任务！")
            time.sleep(60)

    return config


class RedisClient:
    def __init__(self):
        self.client = Redis(host='127.0.0.1',
                              port=6379,
                              decode_responses=True,
                              charset='UTF-8',
                              encoding='UTF-8')

    def get_self_occupied_gpus(self, only_gpus=True):
        """
        获取自己已经占用的Gpu序号
        """
        self_occupied_gpus = self.client.hgetall('self_occupied_gpus')
        if only_gpus:
            all_gpus = []
            for task in self_occupied_gpus.values():
                gpus = [int(device) for device in json.loads(task)["use_gpus"].split(",")]
                all_gpus.extend(gpus)
            return set(all_gpus)
        return [json.loads(g) for g in self_occupied_gpus.values()]

    def join_wait_queue(self, config):
        """
        加入等待队列
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        task_id = str(os.getpid()) + '*' + str(int(time.mktime(time.strptime(creat_time, "%Y-%m-%d %H:%M:%S"))))
        content = {
            "want_gpus": config.want_gpu_num,
            "create_time": creat_time,
            "update_time": creat_time,
            "system_pid": os.getpid(),
            "task_id": task_id,
            "run_notes": config.run_notes,
            "run_name": config.run_name,
            "comet_name": config.comet_name,
            "logger_project": config.logger_project,
        }
        wait_num = len(self.client.lrange('wait_queue', 0, -1))
        self.client.rpush("wait_queue", json.dumps(content))
        if wait_num == 0:
            log.info(f"正在排队中！ 目前排第一位哦！")
        else:
            log.info(f"正在排队中！ 前方还有 {wait_num} 个训练任务！")
        log.info(f"tips: 如果想要对任务进行调整可以移步Redis客户端进行数据修改，只建议进行修改 want_gpus 参数以及删除训练任务操作，其他操作可能会影响Redis读取的稳定性")
        return task_id

    def is_my_turn(self, config):
        """
        排队这么长时间，是否轮到我了？
        """
        curr_task = json.loads(self.client.lrange('wait_queue', 0, -1)[0])
        return curr_task['task_id'] == config.task_id

    def update_queue(self, config):
        """
        更新等待队列
        """
        task = json.loads(self.client.lrange('wait_queue', 0, -1)[0])
        if task['task_id'] != config.task_id:
            # 登记异常信息
            log.info("当前训练任务并不排在队列第一位，请检查Redis数据正确性！")
        curr_time = datetime.datetime.now()
        update_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        task['update_time'] = update_time
        self.client.lset("wait_queue", 0, json.dumps(task))
        log.info("更新训练任务时间戳成功！")

    def pop_wait_queue(self, config):
        """
        弹出当前排位第一的训练任务
        """
        task = json.loads(self.client.lrange('wait_queue', 0, -1)[0])
        if task['task_id'] != config.task_id:
            # 登记异常信息
            log.info("当前训练任务并不排在队列第一位，请检查Redis数据正确性！")
        next_task = self.client.lpop("wait_queue")
        return next_task

    def register_gpus(self, config):
        """
        将当前训练任务登记到GPU占用信息中
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        if not config.task_id:
            task_id = str(os.getpid()) + '*' + str(int(time.mktime(time.strptime(creat_time, "%Y-%m-%d %H:%M:%S"))))
        else:
            task_id = config.task_id
        content = {
            "use_gpus": ','.join([str(gpu) for gpu in list(config.visible_cuda)]),
            "register_time": datetime.datetime.strftime(curr_time,
                                                        '%Y-%m-%d %H:%M:%S'),
            "system_pid": os.getpid(),
            "task_id": task_id,
            "run_notes": config.run_notes,
            "run_name": config.run_name,
            "comet_name": config.comet_name,
            "logger_project": config.logger_project,
        }
        self.client.hset("self_occupied_gpus", task_id, json.dumps(content))
        log.info("成功登记Gpu使用信息到Redis服务器！")

    def deregister_gpus(self, config):
        """
        删除当前训练任务的占用信息
        """
        task = self.client.hget("self_occupied_gpus", config.task_id)
        if task:
            self.client.hdel("self_occupied_gpus", config.task_id)
            log.info("成功删除Redis服务器上的Gpu使用信息！")
        else:
            log.info("无法找到当前训练任务在Redis服务器上的Gpu使用信息！或许可以考虑检查一下Redis的数据 🤔")


@rank_zero_only
def print_start_image():
    console = Console(color_system='256', style="cyan")
    console.print(
        "[bold cyan]\u256D\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256E")
    console.print(
        "[bold cyan]\u2502 \u25CE \u25CB \u25CB \u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591  Start  \u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2502")
    console.print(
        "[bold cyan]\u251C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502                _     _ _______ __   __        /              \u2502")
    console.print("[bold cyan]\u2502                |_____| |______   \\_/         /               \u2502")
    console.print("[bold cyan]\u2502                |     | |______    |         .                \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502                     _____ _______ . _______                  \u2502")
    console.print("[bold cyan]\u2502                       |      |    ' |______                  \u2502")
    console.print("[bold cyan]\u2502                     __|__    |      ______|                  \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502                 ______  _______ __   _  ______               \u2502")
    console.print("[bold cyan]\u2502                 |     \\ |______ | \\  | |  ____               \u2502")
    console.print("[bold cyan]\u2502                 |_____/ |______ |  \\_| |_____|               \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print(
        "[bold cyan]\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")

@rank_zero_only
def print_end_image():
    console = Console(color_system='256', style="cyan")
    console.print()
    console.print("[bold cyan]\u256D\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256E")
    console.print()
    console.print("[bold cyan]\u2502 \u25CE \u25CB \u25CB \u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591 Ending  \u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2502")
    console.print()
    console.print("[bold cyan]\u251C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502                  ______  _____   _____  ______               \u2502")
    console.print("[bold cyan]\u2502                 |  ____ |     | |     | |     \\              \u2502")
    console.print("[bold cyan]\u2502                 |_____| |_____| |_____| |_____/              \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502      ______ _______ _______ _     _        _______        /  \u2502")
    console.print("[bold cyan]\u2502     |_____/ |______ |______ |     | |         |          /   \u2502")
    console.print("[bold cyan]\u2502     |    \\_ |______ ______| |_____| |_____    |         .    \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502              ______  __   __ _______        /   /            \u2502")
    console.print("[bold cyan]\u2502              |_____]   \\_/   |______       /   /             \u2502")
    console.print("[bold cyan]\u2502              |_____]    |    |______      .   .              \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print("[bold cyan]\u2502                                                              \u2502")
    console.print(
        "[bold cyan]\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")

def print_error_info(e):
    print('str(Exception):\t', str(Exception))
    print('str(e):\t\t', str(e))
    print('repr(e):\t', repr(e))
    # Get information about the exception that is currently being handled
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print('e.message:\t', exc_value)
    print("Note, object e and exc of Class %s is %s the same." %
          (type(exc_value), ('not', '')[exc_value is e]))
    print('traceback.print_exc(): ', traceback.print_exc())
    print('traceback.format_exc():\n%s' % traceback.format_exc())

if __name__ == "__main__":
    r = Result()
    pp()
