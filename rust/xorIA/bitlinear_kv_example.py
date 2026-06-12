import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

class STEActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        q_b = 127.0
        # gamma = max(|x|) per-tensor
        gamma = torch.max(torch.abs(x)).clamp(min=1e-8)
        
        x_scaled = x * (q_b / gamma)
        x_rounded = torch.round(x_scaled)
        x_clamped = torch.clamp(x_rounded, -q_b, q_b)
        
        # Dequantize
        rescale = gamma / q_b
        return x_clamped * rescale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class STEWeightTernary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        # AbsMean scale factor
        scale = torch.mean(torch.abs(w)).clamp(min=1e-8)
        
        w_scaled = w / scale
        w_rounded = torch.round(w_scaled)
        w_ternary = torch.clamp(w_rounded, -1.0, 1.0)
        
        # Dequantize
        return w_ternary * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.rms_norm = RMSNorm(in_features)

    def forward(self, x):
        # 1. Sub-LN
        x_norm = self.rms_norm(x)
        
        # 2. Quantize activations
        x_q = STEActivation.apply(x_norm)
        
        # 3. Quantize weights
        w_q = STEWeightTernary.apply(self.weight)
        
        # 4. MatMul
        out = F.linear(x_q, w_q, self.bias)
        return out

class BitAttentionWithKVCache(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = BitLinear(embed_dim, embed_dim)
        self.k_proj = BitLinear(embed_dim, embed_dim)
        self.v_proj = BitLinear(embed_dim, embed_dim)
        self.o_proj = BitLinear(embed_dim, embed_dim)

    def forward(self, x, kv_cache=None):
        B, S, D = x.shape
        
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # KV Cache usage
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            kv_cache = (k, v) # Update cache
            
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.o_proj(context), kv_cache

if __name__ == "__main__":
    B, S, D = 1, 5, 64
    heads = 4
    
    layer = BitAttentionWithKVCache(embed_dim=D, num_heads=heads)
    
    # Simulating token-by-token generation with KV Cache
    print("Iniciando generación con KV Cache...")
    
    # Prompt initial (3 tokens)
    prompt = torch.randn(B, 3, D)
    out, kv_cache = layer(prompt, kv_cache=None)
    print(f"Paso 1 (Prompt): input {prompt.shape}, cache sizes: K={kv_cache[0].shape}, V={kv_cache[1].shape}")
    
    # Generate token 1
    new_token = torch.randn(B, 1, D)
    out, kv_cache = layer(new_token, kv_cache=kv_cache)
    print(f"Paso 2 (Gen 1): input {new_token.shape}, cache sizes: K={kv_cache[0].shape}, V={kv_cache[1].shape}")
    
    # Generate token 2
    new_token = torch.randn(B, 1, D)
    out, kv_cache = layer(new_token, kv_cache=kv_cache)
    print(f"Paso 3 (Gen 2): input {new_token.shape}, cache sizes: K={kv_cache[0].shape}, V={kv_cache[1].shape}")
