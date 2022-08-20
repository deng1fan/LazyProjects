import importlib
from general_files.utils.common_util import get_logger

log = get_logger(__name__)


def get_model_processor(config):
    model_processor_name = config.model_processor
    if 'base:' in model_processor_name:
        module_path = 'general_files.models.' + model_processor_name.replace("base:", "")
    else:
        module_path = config.logger_project + '.models.' + model_processor_name
    processor_name = 'ModelNet'
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as r:
        raise ValueError(f"Please add a processor for this model: {model_processor_name}\n"
                         f"Error module path：{module_path}")
    processor_class = getattr(module, processor_name)

    # 初始化分词器
    tokenizer_module_path = 'general_files.modules.tokenizer'
    tokenizer_module = importlib.import_module(tokenizer_module_path)
    tokenizer = getattr(tokenizer_module, 'Tokenizer')
    tokenizer = tokenizer(config=config)
    tokenizer.add_special_tokens(list(config.additional_special_tokens))
    config.vocab_size = len(tokenizer)
    # 初始化模型
    # 使用pytorch lightning框架
    config.setdefault('vocab_size', tokenizer.vocab_size)
    model = processor_class(config, tokenizer)  # 实例化对象
    model.print_model_introduction()
    if config.stage in ['test', 'finetune']:
        # pytorch lightning框架在测试和微调时加载模型权重
        log.info(f"加载来自 {config.ckpt_path} 的权重！")
        if '.ckpt' in config.get('ckpt_path'):
            model = model.load_from_checkpoint(config.get('ckpt_path'), config=config, tokenizer=tokenizer, strict=False)
        else:
            model = model.load_from_checkpoint(config.get('ckpt_path') + '/best_model.ckpt', config=config, tokenizer=tokenizer, strict=False)
    return model, tokenizer

