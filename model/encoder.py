from peft import LoraConfig, get_peft_model
from transformers import AutoModel


def get_lora_model(base_model_name):
    
    base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
    if 'sam' in base_model_name:
        base_model = base_model.vision_encoder
    lora_config = LoraConfig(
        r=2, lora_alpha=16, lora_dropout=0.3, 
        target_modules=["qkv", "proj"], bias="none"
    )
    return get_peft_model(base_model, lora_config)