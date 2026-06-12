import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from bitnet_chat_inference import convert_model_to_bitnet, BitLinear

MODEL_ID = "facebook/opt-350m"
WEIGHTS_PATH = "bitnet_model_ternary.pt"

config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_config(config)
model = convert_model_to_bitnet(model, group_size=128)

loaded_dict = torch.load(WEIGHTS_PATH, map_location="cpu")

print("Checking loaded dict keys against model...")
model_keys = set()
for name, module in model.named_modules():
    if isinstance(module, BitLinear):
        model_keys.add(f"{name}.weight_ternary_packed")
        model_keys.add(f"{name}.scale_factors")
        if module.bias is not None:
            model_keys.add(f"{name}.bias")
    else:
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            model_keys.add(full_name)

loaded_keys = set(loaded_dict.keys())

missing_in_loaded = model_keys - loaded_keys
extra_in_loaded = loaded_keys - model_keys

print(f"Total keys expected by model logic: {len(model_keys)}")
print(f"Total keys in loaded_dict: {len(loaded_keys)}")
print(f"Keys missing in loaded_dict (present in model structure but not in checkpoint): {sorted(list(missing_in_loaded))}")
print(f"Extra keys in loaded_dict (present in checkpoint but not matched by model logic): {len(extra_in_loaded)}")

# Check if actually any weights were copied
copied_count = 0
not_copied_keys = []
for name, module in model.named_modules():
    if isinstance(module, BitLinear):
        packed_key = f"{name}.weight_ternary_packed"
        if packed_key in loaded_dict:
            copied_count += 1
        else:
            not_copied_keys.append(packed_key)
    else:
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            if full_name in loaded_dict:
                copied_count += 1
            else:
                not_copied_keys.append(full_name)

print(f"Successfully copied parameters: {copied_count}")
print(f"Not copied parameters (init random): {sorted(not_copied_keys)}")
