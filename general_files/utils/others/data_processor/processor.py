import importlib


def get_data_processor(config, tokenizer=None, model=None, only_test=False):
    module_path = config.logger_project + '.data_processor.' + config.dataset_processor
    processor_name = 'Processor'
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as r:
        raise ValueError(f"Please add a processor for this : {config.dataset_processor}")
    except Exception as r:
        raise Exception('未知错误: %s' % r)
    processor_class = getattr(module, processor_name)
    processor = processor_class(config, tokenizer, model, only_test)  # 实例化对象
    return processor
