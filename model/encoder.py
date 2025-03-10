from peft import LoraConfig, get_peft_model, BOFTConfig, LoKrConfig, LoHaConfig
from transformers import AutoModel


PEFT_DICT = {
    'loha': lambda modules: LoHaConfig(target_modules=modules, r=4, module_dropout=0.1),
    'lora': lambda modules: LoraConfig(target_modules=modules, r=4, bias='none', lora_dropout=0.1),
    'lokr': lambda modules: LoKrConfig(target_modules=modules, r=4, module_dropout=0.1),
}

def get_lora_model(base_model_name, peft_name):
    
    base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
    if peft_name == 'none':
        for param in base_model.parameters():
            param.requires_grad = False
        return base_model
    if 'sam' in base_model_name:
        base_model = base_model.vision_encoder
    target_modules = ["qkv", "proj"]
    if 'dino' in base_model_name:
        target_modules = ["query", "key", "value", "dense"]
    config = PEFT_DICT[peft_name](target_modules)
    return get_peft_model(base_model, config)