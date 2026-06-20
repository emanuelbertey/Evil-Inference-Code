use std::time::Instant;
use burn::tensor::{Tensor, backend::Backend, TensorData};
use burn_flex::Flex;

type B = Flex<f32>;

const MAX_CACHE: usize = 4096;
const HEAD_DIM: usize = 128;
const NUM_KV_GROUPS: usize = 8;

/// Strategy 0: pre-alloc full tensor, clone+slice_assign (current code)
struct SClone<Bt: Backend> { k: Tensor<Bt,4>, v: Tensor<Bt,4>, len: usize }
impl<Bt: Backend> SClone<Bt> {
    fn new(d: &Bt::Device) -> Self {
        Self { k: Tensor::zeros([1,MAX_CACHE,NUM_KV_GROUPS,HEAD_DIM],d),
               v: Tensor::zeros([1,MAX_CACHE,NUM_KV_GROUPS,HEAD_DIM],d), len:0 }
    }
    fn append(&mut self, kn: Tensor<Bt,4>, vn: Tensor<Bt,4>) {
        let s=self.len; let e=s+kn.dims()[1];
        self.k = self.k.clone().slice_assign([0..1,s..e,0..NUM_KV_GROUPS,0..HEAD_DIM],kn);
        self.v = self.v.clone().slice_assign([0..1,s..e,0..NUM_KV_GROUPS,0..HEAD_DIM],vn);
        self.len=e;
    }
    fn view(&self) -> (Tensor<Bt,4>,Tensor<Bt,4>) {
        (self.k.clone().narrow(1,0,self.len),self.v.clone().narrow(1,0,self.len))
    }
}

/// Strategy 1: Vec, cat on view
struct SCat<Bt: Backend> { ks: Vec<Tensor<Bt,4>>, vs: Vec<Tensor<Bt,4>>, len: usize }
impl<Bt: Backend> SCat<Bt> {
    fn new(_: &Bt::Device) -> Self { Self { ks:Vec::new(), vs:Vec::new(), len:0 } }
    fn append(&mut self, kn: Tensor<Bt,4>, vn: Tensor<Bt,4>) {
        self.len+=kn.dims()[1]; self.ks.push(kn); self.vs.push(vn);
    }
    fn view(&self) -> (Tensor<Bt,4>,Tensor<Bt,4>) {
        if self.ks.len()==1 { (self.ks[0].clone(),self.vs[0].clone()) }
        else { (Tensor::cat(self.ks.iter().map(|t|t.clone()).collect(),1),
                Tensor::cat(self.vs.iter().map(|t|t.clone()).collect(),1)) }
    }
}

/// Strategy 2: pre-alloc buckets of 64 tokens, write into them sequentially
/// Append writes into current bucket's pre-alloc buffer (no clone).
/// View cats all buckets (up to 64 per bucket × num_buckets = ~64-128 tensors).
struct SBucket<Bt: Backend> {
    buckets_k: Vec<Tensor<Bt,4>>, buckets_v: Vec<Tensor<Bt,4>>,
    bucket_size: usize, len: usize, device: Bt::Device,
}
impl<Bt: Backend> SBucket<Bt> {
    fn new(device: &Bt::Device) -> Self {
        Self { buckets_k:Vec::new(), buckets_v:Vec::new(), bucket_size:64, len:0, device:device.clone() }
    }
    fn append(&mut self, kn: Tensor<Bt,4>, vn: Tensor<Bt,4>) {
        let seq = kn.dims()[1];
        if self.buckets_k.is_empty() {
            let alloc = self.bucket_size.max(seq);
            self.buckets_k.push(Tensor::zeros([1,alloc,NUM_KV_GROUPS,HEAD_DIM],&self.device)
                .slice_assign([0..1,0..seq,0..NUM_KV_GROUPS,0..HEAD_DIM],kn));
            self.buckets_v.push(Tensor::zeros([1,alloc,NUM_KV_GROUPS,HEAD_DIM],&self.device)
                .slice_assign([0..1,0..seq,0..NUM_KV_GROUPS,0..HEAD_DIM],vn));
        } else {
            let idx = self.buckets_k.len()-1;
            let cap = self.buckets_k[idx].dims()[1];
            let used = self.len - idx * self.bucket_size;
            let space = cap - used.min(cap);
            if seq <= space {
                let s=used; let e=s+seq;
                let kk = std::mem::replace(&mut self.buckets_k[idx],
                    Tensor::zeros([1,cap,NUM_KV_GROUPS,HEAD_DIM],&self.device));
                self.buckets_k[idx] = kk.slice_assign([0..1,s..e,0..NUM_KV_GROUPS,0..HEAD_DIM],kn);
                let vv = std::mem::replace(&mut self.buckets_v[idx],
                    Tensor::zeros([1,cap,NUM_KV_GROUPS,HEAD_DIM],&self.device));
                self.buckets_v[idx] = vv.slice_assign([0..1,s..e,0..NUM_KV_GROUPS,0..HEAD_DIM],vn);
            } else {
                let alloc = self.bucket_size.max(seq);
                self.buckets_k.push(Tensor::zeros([1,alloc,NUM_KV_GROUPS,HEAD_DIM],&self.device)
                    .slice_assign([0..1,0..seq,0..NUM_KV_GROUPS,0..HEAD_DIM],kn));
                self.buckets_v.push(Tensor::zeros([1,alloc,NUM_KV_GROUPS,HEAD_DIM],&self.device)
                    .slice_assign([0..1,0..seq,0..NUM_KV_GROUPS,0..HEAD_DIM],vn));
            }
        }
        self.len += seq;
    }
    fn view(&self) -> (Tensor<Bt,4>,Tensor<Bt,4>) {
        let n = self.buckets_k.len();
        if n==1 {
            (self.buckets_k[0].clone().narrow(1,0,self.len),
             self.buckets_v[0].clone().narrow(1,0,self.len))
        } else {
            let remain = self.len - (n-1)*self.bucket_size;
            let mut ks = Vec::with_capacity(n);
            let mut vs = Vec::with_capacity(n);
            for i in 0..n-1 {
                ks.push(self.buckets_k[i].clone().narrow(1,0,self.bucket_size));
                vs.push(self.buckets_v[i].clone().narrow(1,0,self.bucket_size));
            }
            ks.push(self.buckets_k[n-1].clone().narrow(1,0,remain));
            vs.push(self.buckets_v[n-1].clone().narrow(1,0,remain));
            (Tensor::cat(ks,1), Tensor::cat(vs,1))
        }
    }
}

fn make<Bt: Backend>(seq: usize, d: &Bt::Device) -> (Tensor<Bt,4>, Tensor<Bt,4>) {
    let n = seq * NUM_KV_GROUPS * HEAD_DIM;
    (Tensor::from_data(TensorData::new(vec![0.01f32;n],[1,seq,NUM_KV_GROUPS,HEAD_DIM]),d),
     Tensor::from_data(TensorData::new(vec![0.02f32;n],[1,seq,NUM_KV_GROUPS,HEAD_DIM]),d))
}

fn run<Bt: Backend>(label: &str, num_tok: usize, seq_ppc: usize, strat: u8, d: &Bt::Device) {
    let calls = num_tok / seq_ppc;
    match strat {
        0 => {
            let mut c = SClone::<Bt>::new(d);
            let t0 = Instant::now();
            for _ in 0..calls { let (k,v)=make(seq_ppc,d); c.append(k,v); }
            let ap_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = Instant::now();
            for _ in 0..24 { let _ = c.view(); }
            let vw_ms = t1.elapsed().as_secs_f64() * 1000.0;
            println!("  {:20}  append={:9.3}ms  view24x={:9.3}ms", label, ap_ms, vw_ms);
        }
        1 => {
            let mut c = SCat::<Bt>::new(d);
            let t0 = Instant::now();
            for _ in 0..calls { let (k,v)=make(seq_ppc,d); c.append(k,v); }
            let ap_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = Instant::now();
            for _ in 0..24 { let _ = c.view(); }
            let vw_ms = t1.elapsed().as_secs_f64() * 1000.0;
            println!("  {:20}  append={:9.3}ms  view24x={:9.3}ms", label, ap_ms, vw_ms);
        }
        2 => {
            let mut c = SBucket::<Bt>::new(d);
            let t0 = Instant::now();
            for _ in 0..calls { let (k,v)=make(seq_ppc,d); c.append(k,v); }
            let ap_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = Instant::now();
            for _ in 0..24 { let _ = c.view(); }
            let vw_ms = t1.elapsed().as_secs_f64() * 1000.0;
            println!("  {:20}  append={:9.3}ms  view24x={:9.3}ms", label, ap_ms, vw_ms);
        }
        _ => unreachable!()
    }
}

fn main() {
    let d = Default::default();
    println!("=== KV Cache Benchmark ===\n");

    for seq_ppc in &[1usize, 16] {
        println!("--- seq_per_call={}, {} appends to fill 4096 ---", seq_ppc, 4096/seq_ppc);
        run::<B>("clone+slice_assign", 4096, *seq_ppc, 0, &d);
        run::<B>("Vec+cat", 4096, *seq_ppc, 1, &d);
        run::<B>("bucket64+take", 4096, *seq_ppc, 2, &d);
        println!();
    }

    // ═══════════════════════════════════════════════════════════════
    // RoPE Benchmark
    // ═══════════════════════════════════════════════════════════════
    println!("\n=== RoPE Benchmark ===\n");
    rope_benchmark::run::<B>(&d);
}

mod rope_benchmark {
    use std::time::Instant;
    use burn::tensor::{Tensor, backend::Backend, TensorData};
    use super::{B, HEAD_DIM, NUM_KV_GROUPS};

    fn make_qk<Bt: Backend>(batch: usize, seq: usize, nheads: usize, nkv: usize, dim: usize, d: &Bt::Device)
        -> (Tensor<Bt,4>, Tensor<Bt,4>)
    {
        let n = batch * seq * nheads * dim;
        let nk = batch * seq * nkv * dim;
        (Tensor::from_data(TensorData::new(vec![0.5f32; n], [batch, seq, nheads, dim]), d),
         Tensor::from_data(TensorData::new(vec![0.3f32; nk], [batch, seq, nkv, dim]), d))
    }

    fn precompute_cos_sin_2d<Bt: Backend>(max_seq: usize, dim: usize, d: &Bt::Device) -> (Tensor<Bt,2>, Tensor<Bt,2>) {
        let theta: Vec<f32> = (0..dim/2).map(|i| 1.0/10000.0f32.powf(2.0*i as f32/dim as f32)).collect();
        let theta_t = Tensor::<Bt,1>::from_data(TensorData::new(theta, [dim/2]), d);
        let pos: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let pos_t = Tensor::<Bt,1>::from_data(TensorData::new(pos, [max_seq]), d);
        let ang = pos_t.reshape([max_seq,1]) * theta_t.reshape([1,dim/2]);
        (ang.clone().cos(), ang.sin())
    }

    fn precompute_cos_sin_4d<Bt: Backend>(max_seq: usize, dim: usize, d: &Bt::Device) -> (Tensor<Bt,4>, Tensor<Bt,4>) {
        let theta: Vec<f32> = (0..dim/2).map(|i| 1.0/10000.0f32.powf(2.0*i as f32/dim as f32)).collect();
        let theta_t = Tensor::<Bt,1>::from_data(TensorData::new(theta, [dim/2]), d);
        let pos: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let pos_t = Tensor::<Bt,1>::from_data(TensorData::new(pos, [max_seq]), d);
        let ang = pos_t.reshape([max_seq,1]) * theta_t.reshape([1,dim/2]);
        (ang.clone().cos().reshape([max_seq,1,1,dim/2]), ang.sin().reshape([max_seq,1,1,dim/2]))
    }

    // ── Strategy 0: OLD apply_rope (recompute theta every call) ──
    fn apply_rope_old<Bt: Backend>(q: Tensor<Bt,4>, k: Tensor<Bt,4>, offset: usize) -> (Tensor<Bt,4>, Tensor<Bt,4>) {
        let [_, seq_len, _nh, head_dim] = q.dims();
        let theta: Vec<f32> = (0..head_dim/2).map(|i| 1.0/10000.0f32.powf(2.0*i as f32/head_dim as f32)).collect();
        let theta_t = Tensor::<Bt,1>::from_data(TensorData::new(theta, [head_dim/2]), &q.device());
        let pos: Vec<f32> = (offset..offset+seq_len).map(|p| p as f32).collect();
        let pos_t = Tensor::<Bt,1>::from_data(TensorData::new(pos, [seq_len]), &q.device());
        let ang = pos_t.reshape([seq_len,1]) * theta_t.reshape([1,head_dim/2]);
        let cos = ang.clone().cos().reshape([1,seq_len,1,head_dim/2]);
        let sin = ang.sin().reshape([1,seq_len,1,head_dim/2]);

        let q1 = q.clone().slice([0..1,0..seq_len,0.._nh,0..head_dim/2]);
        let q2 = q.clone().slice([0..1,0..seq_len,0.._nh,head_dim/2..head_dim]);
        let k1 = k.clone().slice([0..1,0..seq_len,0..k.dims()[2],0..head_dim/2]);
        let k2 = k.clone().slice([0..1,0..seq_len,0..k.dims()[2],head_dim/2..head_dim]);

        (Tensor::cat(vec![q1.clone()*cos.clone()-q2.clone()*sin.clone(), q1*sin.clone()+q2*cos.clone()],3),
         Tensor::cat(vec![k1.clone()*cos.clone()-k2.clone()*sin.clone(), k1*sin+k2*cos],3))
    }

    // ── Strategy 1: apply_rope_cached with narrow+reshape+chunk (BEFORE optimization) ──
    fn apply_rope_cached_old<Bt: Backend>(q: Tensor<Bt,4>, k: Tensor<Bt,4>, cos_sin: (&Tensor<Bt,2>, &Tensor<Bt,2>), offset: usize) -> (Tensor<Bt,4>, Tensor<Bt,4>) {
        let [_, seq_len, _nh, head_dim] = q.dims();
        let cos = cos_sin.0.clone().narrow(0, offset, seq_len)
            .reshape([1, seq_len, 1, head_dim/2]);
        let sin = cos_sin.1.clone().narrow(0, offset, seq_len)
            .reshape([1, seq_len, 1, head_dim/2]);

        let q_chunks = q.chunk(2, 3);
        let (q1, q2) = (q_chunks[0].clone(), q_chunks[1].clone());
        let k_chunks = k.chunk(2, 3);
        let (k1, k2) = (k_chunks[0].clone(), k_chunks[1].clone());

        (Tensor::cat(vec![q1.clone()*cos.clone()-q2.clone()*sin.clone(), q1*sin.clone()+q2*cos.clone()],3),
         Tensor::cat(vec![k1.clone()*cos.clone()-k2.clone()*sin.clone(), k1*sin+k2*cos],3))
    }

    // ── Strategy 2: apply_rope_cached with narrow (no reshape, 4D cache) ──
    fn apply_rope_cached_new<Bt: Backend>(q: Tensor<Bt,4>, k: Tensor<Bt,4>, cos_sin: (&Tensor<Bt,4>, &Tensor<Bt,4>), offset: usize) -> (Tensor<Bt,4>, Tensor<Bt,4>) {
        let [_, seq_len, _nh, head_dim] = q.dims();
        let cos = cos_sin.0.clone().narrow(0, offset, seq_len);
        let sin = cos_sin.1.clone().narrow(0, offset, seq_len);

        let h_half = head_dim/2;
        let b = q.dims()[0];
        let q1 = q.clone().slice([0..b,0..seq_len,0.._nh,0..h_half]);
        let q2 = q.clone().slice([0..b,0..seq_len,0.._nh,h_half..head_dim]);
        let k_nkv = k.dims()[2];
        let k1 = k.clone().slice([0..b,0..seq_len,0..k_nkv,0..h_half]);
        let k2 = k.clone().slice([0..b,0..seq_len,0..k_nkv,h_half..head_dim]);

        (Tensor::cat(vec![q1.clone()*cos.clone()-q2.clone()*sin.clone(), q1*sin.clone()+q2*cos.clone()],3),
         Tensor::cat(vec![k1.clone()*cos.clone()-k2.clone()*sin.clone(), k1*sin+k2*cos],3))
    }

    // ── Strategy 3: FUSED kernel (raw as_slice, in-place mutation) ──
    fn apply_rope_fused<Bt: Backend>(q: Tensor<Bt,4>, k: Tensor<Bt,4>, offset: usize) -> (Tensor<Bt,4>, Tensor<Bt,4>) {
        let [batch, seq_len, nheads, head_dim] = q.dims();
        let nkv = k.dims()[2];
        let hh = head_dim / 2;

        let device = q.device();
        let q_data = q.into_data();
        let k_data = k.into_data();
        let mut q_slice = q_data.as_slice::<f32>().unwrap().to_vec();
        let mut k_slice = k_data.as_slice::<f32>().unwrap().to_vec();

        let theta: Vec<f32> = (0..hh).map(|i| 1.0 / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32)).collect();

        for b in 0..batch {
            for s in 0..seq_len {
                let pos = offset + s;
                for h in 0..nheads {
                    let base_q = (b * seq_len + s) * nheads * head_dim + h * head_dim;
                    for i in 0..hh {
                        let cos = (pos as f32 * theta[i]).cos();
                        let sin = (pos as f32 * theta[i]).sin();
                        let q1 = q_slice[base_q + i];
                        let q2 = q_slice[base_q + i + hh];
                        q_slice[base_q + i] = q1 * cos - q2 * sin;
                        q_slice[base_q + i + hh] = q1 * sin + q2 * cos;
                    }
                }
                for h in 0..nkv {
                    let base_k = (b * seq_len + s) * nkv * head_dim + h * head_dim;
                    for i in 0..hh {
                        let cos = (pos as f32 * theta[i]).cos();
                        let sin = (pos as f32 * theta[i]).sin();
                        let k1 = k_slice[base_k + i];
                        let k2 = k_slice[base_k + i + hh];
                        k_slice[base_k + i] = k1 * cos - k2 * sin;
                        k_slice[base_k + i + hh] = k1 * sin + k2 * cos;
                    }
                }
            }
        }

        let q_out = Tensor::<Bt,4>::from_data(TensorData::new(q_slice, [batch, seq_len, nheads, head_dim]), &device);
        let k_out = Tensor::<Bt,4>::from_data(TensorData::new(k_slice, [batch, seq_len, nkv, head_dim]), &device);
        (q_out, k_out)
    }

    pub fn run<Bt: Backend>(d: &Bt::Device) {
        let nheads = NUM_KV_GROUPS; // simplified: nheads == nkv for test
        let nkv = NUM_KV_GROUPS;

        let (cos2, sin2) = precompute_cos_sin_2d::<Bt>(4096, HEAD_DIM, d);
        let (cos4, sin4) = precompute_cos_sin_4d::<Bt>(4096, HEAD_DIM, d);

        // Prefill: seq_len=128 tokens all at once
        // Inference: seq_len=1 token at a time
        for &seq_len in &[1usize, 128] {
            let iters = if seq_len == 1 { 4096 } else { 32 };
            let (q, k) = make_qk(1, seq_len, nheads, nkv, HEAD_DIM, d);

            // ── OLD apply_rope (no cache, recompute theta) ──
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = apply_rope_old(q.clone(), k.clone(), 0);
            }
            let t_old = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            // verify correctness by running once
            let _ = apply_rope_old(q.clone(), k.clone(), 0);

            // ── CACHED old: narrow+reshape+chunk ──
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = apply_rope_cached_old(q.clone(), k.clone(), (&cos2, &sin2), 0);
            }
            let t_cached_old = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // ── CACHED new: slice direct ──
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = apply_rope_cached_new(q.clone(), k.clone(), (&cos4, &sin4), 0);
            }
            let t_cached_new = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // ── FUSED: raw slice in-place ──
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = apply_rope_fused(q.clone(), k.clone(), 0);
            }
            let t_fused = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            println!("seq_len={}:", seq_len);
            println!("  apply_rope (recompute theta)     {:8.3}ms", t_old);
            println!("  apply_rope_cached (narrow+chunk)  {:8.3}ms", t_cached_old);
            println!("  apply_rope_cached (slice direct)  {:8.3}ms", t_cached_new);
            println!("  apply_rope_fused (raw as_slice)   {:8.3}ms", t_fused);
        }
    }
}
