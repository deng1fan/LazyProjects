import importlib

def get_trainer_processor(config):
    trainer_processor = config.trainer_processor
    module_path = 'general_files.trainer.' + trainer_processor
    processor_name = 'ModelTrainer'
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as r:
        raise ValueError(f"Please add a trainer processor named like this: {trainer_processor}.")
    except Exception as r:
        raise Exception('未知错误 %s' % r)
    processor_class = getattr(module, processor_name)
    return processor_class

