pub const KUANTGRAD_ADAMW_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void kuantgrad_adamw(
    float* params,
    float* m_state,
    float* v_state,
    const unsigned char* compressed_grads,
    int N,
    float lr, float beta1, float beta2, float eps, float wd,
    float inv_beta1_t, float inv_beta2_t
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int n_groups = (N + 7) / 8;
    if (g >= n_groups) return;

    int idx_start = g * 8;
    int n_in_group = (g == n_groups - 1) ? (N - idx_start) : 8;
    
    const unsigned char* ptr = compressed_grads + g * 7;
    
    // Read scale (f16)
    unsigned short scale_bits = ptr[0] | (ptr[1] << 8);
    __half scale_half = *reinterpret_cast<__half*>(&scale_bits);
    float scale = __half2float(scale_half);

    unsigned long long bits = 0;
    if (scale != 0.0f) {
        bits = (unsigned long long)ptr[2] |
               ((unsigned long long)ptr[3] << 8) |
               ((unsigned long long)ptr[4] << 16) |
               ((unsigned long long)ptr[5] << 24) |
               ((unsigned long long)ptr[6] << 32);
    }

    for (int i = 0; i < n_in_group; i++) {
        float g_val = 0.0f;
        if (scale != 0.0f) {
            int bit_pos = i * 5;
            unsigned int q = (bits >> bit_pos) & 0x1F;
            float norm = (q / 15.5f) - 1.0f;
            g_val = norm * scale;
        }

        int idx = idx_start + i;
        float param = params[idx];
        float grad = g_val + wd * param;
        
        float m_new = beta1 * m_state[idx] + (1.0f - beta1) * grad;
        float v_new = beta2 * v_state[idx] + (1.0f - beta2) * grad * grad;
        
        m_state[idx] = m_new;
        v_state[idx] = v_new;
        
        float m_hat = m_new / inv_beta1_t;
        float v_hat = v_new / inv_beta2_t;
        
        params[idx] = param - lr * (m_hat / (sqrtf(v_hat) + eps));
    }
}
"#;
