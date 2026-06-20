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
}
