# ==============================================================================
# BITNET B1.58 PURE TERNARY INFERENCE CHAT CONSOLE
# Run this script with: python bitnet_chat_inference.py
# ==============================================================================
import os
import sys
import time
import gc
import ctypes
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Set execution device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Execution Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

def clean_system_ram():
    """Aggressively releases unused RAM from both GPU cache and system heap back to OS."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

def quantize_activation(x):
    """Quantize activations to 8-bit per token."""
    scale = 127.0 / (torch.max(torch.abs(x), dim=-1, keepdim=True).values + 1e-5)
    x_quant = torch.clamp(torch.round(x * scale), -128, 127)
    x_dequant = x_quant / scale
    return x + (x_dequant - x).detach()

def unpack_ternary_weights(packed, original_shape):
    """Unpack uint8 byte array back into ternary {-1,0,1} tensor."""
    val0 = packed & 3
    val1 = (packed >> 2) & 3
    val2 = (packed >> 4) & 3
    val3 = (packed >> 6) & 3
    flat = torch.stack([val0, val1, val2, val3], dim=1).view(-1)
    flat_signed = torch.where(flat == 2, torch.tensor(-1, dtype=torch.int8, device=packed.device), flat.to(torch.int8))
    num_elements = original_shape[0] * original_shape[1]
    return flat_signed[:num_elements].view(original_shape)

def pack_ternary_weights(w_quant):
    """Pack ternary weights {-1,0,1} into 2-bit, 4 per uint8 byte."""
    w_unsigned = torch.where(w_quant == -1, torch.tensor(2, dtype=torch.int8, device=w_quant.device), w_quant.to(torch.int8))
    flat = w_unsigned.view(-1)
    pad_len = (4 - (len(flat) % 4)) % 4
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len), value=0)
    flat_reshaped = flat.view(-1, 4)
    packed = (flat_reshaped[:, 0] | (flat_reshaped[:, 1] << 2) | (flat_reshaped[:, 2] << 4) | (flat_reshaped[:, 3] << 6))
    return packed.to(torch.uint8)

def quantize_weight_grouped(w, group_size=128):
    """Quantize weight matrix to ternary values {-1, 0, 1} in groups of group_size."""
    out_features, in_features = w.shape
    num_groups = (in_features + group_size - 1) // group_size
    padded_in_features = num_groups * group_size
    if padded_in_features > in_features:
        w_pad = F.pad(w, (0, padded_in_features - in_features), value=0.0)
    else:
        w_pad = w
    w_reshaped = w_pad.view(out_features, num_groups, group_size)
    beta = torch.mean(torch.abs(w_reshaped), dim=-1, keepdim=True)
    w_scaled = w_reshaped / (beta + 1e-5)
    w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
    return w_quant.view(out_features, -1)[:, :in_features], beta.squeeze(-1)


class BitLinear(nn.Module):
    """
    BitLinear Layer optimized for RAM: keeps weights in int8 and scales in float16.
    Dequantizes on-the-fly during forward pass to save 80% RAM.
    """
    def __init__(self, in_features, out_features, bias=True, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Guardamos en int8 y float16 nativo para no gastar 2GB de RAM
        self.register_buffer("w_quant", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.zeros((out_features, in_features // group_size), dtype=torch.float16))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.ternary_mode = True
            
    def forward(self, x):
        # Desempaquetado dinámico directo en VRAM/RAM (Ahorra muchísima memoria)
        out_features, in_features = self.w_quant.shape
        num_groups = self.scales.shape[1]
        
        scales_expanded = self.scales.unsqueeze(-1).expand(out_features, num_groups, self.group_size)
        scales_expanded = scales_expanded.reshape(out_features, -1)[:, :in_features]
        
        w = self.w_quant.to(x.dtype) * scales_expanded.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        
        x_quant = quantize_activation(x)
        return F.linear(x_quant, w, bias)
        
    def load_ternary_weights(self, w_quant, scale_factors):
        """Store int8 weights and scales directly."""
        self.w_quant.copy_(w_quant.to(torch.int8))
        self.scales.copy_(scale_factors.to(torch.float16))
        self.ternary_mode = True

def convert_model_to_bitnet(model, group_size=128):
    """Replaces nn.Linear layers with BitLinear framework layers."""
    def replace_linears(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                if name == "lm_head":
                    continue
                bit_linear = BitLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    group_size=group_size
                )
                bit_linear.to(child.weight.dtype)
                setattr(module, name, bit_linear)
            else:
                replace_linears(child)

    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        replace_linears(model.model.decoder.layers)
    else:
        replace_linears(model)
    return model


def main():
    # Completely silence Hugging Face Hub logging and warnings to avoid any network-related messages
    import logging
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()

    MODEL_ID = "facebook/opt-350m"
    WEIGHTS_PATH = "bitnet_model_ternary.pt"

    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Error: '{WEIGHTS_PATH}' not found in current directory!")
        print("Please verify the file exists and run the script in its directory.")
        sys.exit(1)

    print("📥 Cargando tokenizador original de facebook/opt-350m usando caché (ignora el warning de symlinks en Windows, es inofensivo)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)

    print("🏗️ Creating model structure from config (strictly local skeleton, no weights downloaded)...")
    # Usa torch.bfloat16 para inicialización si es posible para ahorrar RAM inicial
    model = AutoModelForCausalLM.from_config(config)
    model = convert_model_to_bitnet(model, group_size=128)
    
    # Limpiamos agresivamente la RAM de los esqueletos FP32 originales de nn.Linear
    clean_system_ram()
    model = model.to(device)

    print("🔓 Loading and unpacking local ternary weights directly into model...")
    loaded_dict = torch.load(WEIGHTS_PATH, map_location=device)

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            packed_key = f"{name}.weight_ternary_packed"
            if packed_key in loaded_dict:
                packed_w = loaded_dict[packed_key].to(device)
                shape = tuple(loaded_dict[f"{name}.original_shape"].tolist())
                w_quant = unpack_ternary_weights(packed_w, shape).to(device)
                scale_factors = loaded_dict[f"{name}.scale_factors"].to(device)
                
                module.load_ternary_weights(w_quant, scale_factors)
                
                if f"{name}.bias" in loaded_dict:
                    module.bias.data.copy_(loaded_dict[f"{name}.bias"])
        else:
            # Load standard weights for normal Layers (lm_head, embeddings)
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}" if name else param_name
                if full_name in loaded_dict:
                    param.data.copy_(loaded_dict[full_name])
                else:
                    print(f"⚠️ PELIGRO - CAPA FALTANTE EN EL .PT: {full_name} (Esto causa basura en la salida)")

    model.eval()
    print("💻 Pasando el modelo a FP32 para evitar el desbordamiento infinito del LayerNorm en la CPU...")
    model.float()
    
    # Release memory
    del loaded_dict
    clean_system_ram()
    print(f"\n📊 --- DATOS DEL MODELO CARGADO ---")
    print(f"   • Arquitectura: {config.model_type.upper()}")
    print(f"   • Capas Ocultas (Layers): {getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 'Desconocido'))}")
    print(f"   • Dimensión (Hidden Size): {getattr(config, 'hidden_size', getattr(config, 'n_embd', 'Desconocido'))}")
    print(f"   • Cabezales de Atención (Heads): {getattr(config, 'num_attention_heads', getattr(config, 'n_head', 'Desconocido'))}")
    print(f"   • Contexto Máximo (Cache/Seq Len): {getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 'Desconocido'))} tokens")
    print(f"   • Tamaño del Vocabulario: {config.vocab_size} tokens")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   • Parámetros Totales: {total_params:,}")
    print(f"-----------------------------------\n")

    print("\n✅ Pure Ternary BitNet model loaded successfully!\n")

    # Generation Parameters
    max_new_tokens = 60
    temperature = 0.7
    top_k = 50
    top_p = 0.95

    print("💬 BitNet Chat Console Started! Type 'exit' to quit.")
    print("-" * 60)
    print("💡 Commands:")
    print("  /config max=100 temp=0.5 topk=30 topp=0.9   <- Adjust parameters on the fly")
    print("-" * 60)
    print(f"⚙️ Active Config: max={max_new_tokens} tokens | temp={temperature} | top_k={top_k} | top_p={top_p}\n")

    while True:
        try:
            user_input = input("👤 User: ")
            if user_input.strip().lower() in ['exit', 'quit', 'q']:
                print("👋 Session closed. Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            if user_input.strip().startswith("/config"):
                parts = user_input.split()
                for part in parts[1:]:
                    if part.startswith("max="):
                        max_new_tokens = int(part.split("=")[1])
                    elif part.startswith("temp="):
                        temperature = float(part.split("=")[1])
                    elif part.startswith("topk="):
                        top_k = int(part.split("=")[1])
                    elif part.startswith("topp="):
                        top_p = float(part.split("=")[1])
                print(f"⚙️ Param updated: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}\n")
                continue
                
            inputs = tokenizer(user_input, return_tensors="pt").to(device)
            
            # Measure generation speed
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature
                )
            end_time = time.time()
            
            prompt_len = inputs.input_ids.shape[1]
            response_tokens = outputs[0][prompt_len:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            tokens_gen = len(response_tokens)
            duration = end_time - start_time
            tokens_per_sec = tokens_gen / duration if duration > 0 else 0.0
            
            print(f"🤖 BitNet: {response.strip()}")
            print(f"   ⏱️ [Speed: {tokens_gen} tokens generated in {duration:.3f}s ({tokens_per_sec:.2f} tokens/sec)]\n")
            
        except KeyboardInterrupt:
            print("\n👋 Chat closed by user.")
            break
        except Exception as e:
            print(f"❌ Error in generation: {str(e)}\n")

if __name__ == "__main__":
    main()
