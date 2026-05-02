// ─── Transformer Module ─────────────────────────────────────────────────────
//
// Fully configurable Transformer with:
//   - Grouped Query Attention (GQA): configurable num_heads & num_kv_groups
//   - Rotary Position Embeddings (RoPE) with NTK-aware scaling
//   - SwiGLU or standard GELU feed-forward networks
//   - Pre-Norm (RMSNorm) residual connections
//   - Causal masking for autoregressive generation
//   - Attention logit soft-capping
//
// Example configurations:
//
//   // LLaMA-style: 8 heads, 2 KV groups (GQA)
//   TransformerConfig {
//       num_layers: 6,
//       layer: TransformerLayerConfig {
//           d_model: 512,
//           num_heads: 8,
//           num_kv_groups: 2,  // Each KV group shared by 4 Q heads
//           use_swiglu: true,
//           ..Default::default()
//       },
//   }
//
//   // Standard MHA (GPT-style)
//   TransformerConfig {
//       num_layers: 6,
//       layer: TransformerLayerConfig {
//           d_model: 512,
//           num_heads: 8,
//           num_kv_groups: 0,  // 0 = same as num_heads (MHA)
//           use_swiglu: false,
//           ..Default::default()
//       },
//   }

pub mod rope;
pub mod heads;
pub mod attention;
pub mod feedforward;
pub mod layer;

// Re-export main types for convenient access
pub use rope::{RoPE, RoPEConfig};
pub use heads::{HeadConfig, QKVProjection, OutputProjection, repeat_kv};
pub use attention::{Attention, AttentionConfig, KVCache};
pub use feedforward::{FeedForwardBlock, FeedForwardConfig, FeedForward, SwiGLUFeedForward};
pub use layer::{
    TransformerLayer, TransformerLayerConfig,
    Transformer, TransformerConfig,
    RMSNorm,
};
