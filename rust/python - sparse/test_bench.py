import torch, time, gc, sys
sys.path.insert(0, r"C:\Users\Emabe\Documents\GitHub\xlstm\rust\python - sparse")
from sparse import SparseAttentionMio3
from attention import Attention

BSZ = 128
S, K = 1024, 2
torch.manual_seed(0)
x = torch.randn(1, S, 768)

NB = max(1, (S + BSZ - 1) // BSZ)
print(f"S={S}  BSZ={BSZ}  NB={NB}  topK={K}  ratio={K*100//NB}%")

gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
t0 = time.perf_counter()
with torch.no_grad(): a = Attention(768, 12, 4, 64).eval()(x)
mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
print(f"GQA:   {1000*(time.perf_counter()-t0):.0f}ms  {mem:.0f}MB  sum={a.sum():.2f}")

gc.collect(); torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
t0 = time.perf_counter()
with torch.no_grad(): b = SparseAttentionMio3(768, 12, 4, 64, num_selected_blocks=K).eval()(x)
mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
print(f"Mio3K={K}: {1000*(time.perf_counter()-t0):.0f}ms  {mem:.0f}MB  sum={b.sum():.2f}")
print(f"diff_max={((a-b).abs().max().item()):.4f}")
