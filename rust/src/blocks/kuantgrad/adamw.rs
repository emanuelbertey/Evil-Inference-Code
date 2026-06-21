// ─── AdamW optimizador manual (sin Burn) para Vec<f32> ─────────────
//
// State por parámetro: m (momento), v (RMS), t (contador).
// step() toma params, grads y los hiperparámetros.

#[derive(Clone, Debug)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8, wd: 0.0 }
    }
}

#[derive(Clone, Debug)]
pub struct AdamWState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: i32,
}

impl AdamWState {
    pub fn new(n: usize) -> Self {
        Self { m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32], cfg: &AdamWConfig) {
        self.t += 1;
        let t = self.t as f64;
        let inv_beta1_t = 1.0 - (cfg.beta1 as f64).powi(t as i32);
        let inv_beta2_t = 1.0 - (cfg.beta2 as f64).powi(t as i32);
        for i in 0..params.len() {
            let g = grads[i] + cfg.wd * params[i];
            self.m[i] = cfg.beta1 * self.m[i] + (1.0 - cfg.beta1) * g;
            self.v[i] = cfg.beta2 * self.v[i] + (1.0 - cfg.beta2) * g * g;
            let m_hat = self.m[i] as f64 / inv_beta1_t;
            let v_hat = self.v[i] as f64 / inv_beta2_t;
            params[i] -= cfg.lr * (m_hat / (v_hat.sqrt() + cfg.eps as f64)) as f32;
        }
    }
}

/// Grupo de parámetros: cada uno con su slice de params, gradientes y estado AdamW.
pub struct ParamGroup<'a> {
    pub params: &'a mut [f32],
    pub state: &'a mut AdamWState,
}

/// Aplica un paso AdamW a un lote de grupos de parámetros.
pub fn step_group(groups: &mut [ParamGroup], cfg: &AdamWConfig, grads_list: &[&[f32]]) {
    for (group, grads) in groups.iter_mut().zip(grads_list) {
        group.state.step(group.params, grads, cfg);
    }
}
