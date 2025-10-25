############################################################
# GEMM BENCHMARK APP  (Phase A + simple Streamlit UI)
# ----------------------------------------------------------
# - Compare CPU truth, torch.mm (CPU/GPU), torch.bmm (batched)
# - Compute GFLOP/s, relative error, timing
# - Display results interactively in Streamlit
############################################################

import time
import torch
import streamlit as st
import pandas as pd
import platform

# ----------------------- DEVICE SETUP ---------------------
st.title("🔬 GEMM Benchmark Dashboard (Phase A)")
st.write("Compare CPU vs GPU performance & precision on  **B200 GPU**")

torch.set_float32_matmul_precision("high")
dtype = torch.float32
cpu = torch.device("cpu")
cuda_ok = torch.cuda.is_available()
gpu = torch.device("cuda") if cuda_ok else None
sync = (lambda: torch.cuda.synchronize()) if cuda_ok else (lambda: None)

# Show environment info
st.sidebar.header("📟 System Info")
st.sidebar.write("Python:", platform.python_version())
st.sidebar.write("PyTorch:", torch.__version__)
if cuda_ok:
    props = torch.cuda.get_device_properties(0)
    st.sidebar.write("GPU:", props.name)
    st.sidebar.write(f"SMs: {props.multi_processor_count},  HBM: {props.total_memory/1e9:.1f} GB")
    st.sidebar.write(f"Compute capability: {props.major}.{props.minor}")
else:
    st.sidebar.write("⚠️ No CUDA GPU detected")

# ----------------------- UTILITIES ------------------------
def cpu_truth_gemm(A, B):
    """Naive triple-loop GEMM — correctness oracle."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += float(A[i, k]) * float(B[k, j])
            C[i, j] = s
    return C

def gflops(M, N, K, sec):
    """Compute GFLOP/s."""
    return (2.0 * M * N * K) / (sec * 1e9)

def time_op(fn, warmup=3, iters=10, sync=None):
    """Average runtime of a callable (s)."""
    for _ in range(warmup):
        fn();  sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn();  sync()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def rel_fro_err(C, Cref):
    """Relative Frobenius norm error."""
    num = torch.linalg.norm((C - Cref).float())
    den = torch.linalg.norm(Cref.float())
    return (num / (den + 1e-12)).item()

# ----------------------- TEST CONFIG ----------------------
shapes_square = [(1024,1024,1024),(2048,2048,2048),(4096,4096,4096)]
shapes_skinny_wide = [(4096,128,4096),(128,4096,4096)]
big_shapes = [(8192,8192,8192)]
batch_cfgs = [(32,256,256,256),(64,512,512,512)]

# ----------------------- BENCH FUNCTION -------------------
def run_one(M,N,K):
    A = torch.randn(M,K,device=cpu,dtype=dtype)
    B = torch.randn(K,N,device=cpu,dtype=dtype)
    C_ref = cpu_truth_gemm(A,B)
    # CPU matmul
    t_cpu = time_op(lambda: A@B,5,5)
    err_cpu = rel_fro_err(A@B,C_ref)
    rows=[["torch.mm CPU","float32",M,N,K,gflops(M,N,K,t_cpu),t_cpu,err_cpu]]
    if cuda_ok:
        A_g,B_g=A.to(gpu),B.to(gpu)
        for allow_tf32 in (False,True):
            torch.backends.cuda.matmul.allow_tf32=allow_tf32
            label="FP32 strict" if not allow_tf32 else "TF32"
            sync(); t_gpu=time_op(lambda: A_g@B_g,5,30,sync)
            err=rel_fro_err((A_g@B_g).cpu(),C_ref)
            rows.append([f"torch.mm GPU ({label})","float32",M,N,K,gflops(M,N,K,t_gpu),t_gpu,err])
        # FP16/BF16
        for cast,name in ((torch.float16,"FP16"),(torch.bfloat16,"BF16")):
            A16,B16=A_g.to(cast),B_g.to(cast)
            sync(); t16=time_op(lambda: A16@B16,5,30,sync)
            err16=rel_fro_err((A16@B16).float().cpu(),C_ref)
            rows.append([f"torch.mm GPU","{name}",M,N,K,gflops(M,N,K,t16),t16,err16])
    return rows

def run_batched(Bsz,M,K,N):
    A=torch.randn(Bsz,M,K,device=cpu,dtype=dtype)
    B=torch.randn(Bsz,K,N,device=cpu,dtype=dtype)
    # slow ref
    C_ref=torch.stack([cpu_truth_gemm(A[b],B[b]) for b in range(Bsz)])
    t_cpu=time_op(lambda: torch.bmm(A,B),3,5)
    err_cpu=rel_fro_err(torch.bmm(A,B),C_ref)
    rows=[["torch.bmm CPU","float32",M,N,K,gflops(Bsz*M,N,K,t_cpu),t_cpu,err_cpu]]
    if cuda_ok:
        A_g,B_g=A.to(gpu),B.to(gpu)
        sync(); t_gpu=time_op(lambda: torch.bmm(A_g,B_g),5,30,sync)
        err_gpu=rel_fro_err(torch.bmm(A_g,B_g).cpu(),C_ref)
        rows.append(["torch.bmm GPU","float32",M,N,K,gflops(Bsz*M,N,K,t_gpu),t_gpu,err_gpu])
    return rows

# ----------------------- MAIN RUN -------------------------
all_rows=[]
st.header("▶️ Running GEMM Benchmarks...")
for M,N,K in shapes_square+shapes_skinny_wide+big_shapes:
    all_rows+=run_one(M,N,K)
for (Bsz,M,K,N) in batch_cfgs:
    all_rows+=run_batched(Bsz,M,K,N)

df=pd.DataFrame(all_rows,columns=["Op","Precision","M","N","K","GFLOP/s","Time (s)","Rel Error"])
st.dataframe(df.style.background_gradient(subset=["GFLOP/s"], cmap="Blues"))

# Simple summary chart
st.subheader("📈 Performance Chart (GFLOP/s)")
st.bar_chart(df[["Op","GFLOP/s"]].set_index("Op"))

# Save CSV
csv=df.to_csv(index=False)
st.download_button("💾 Download results as CSV",csv,"gemm_results.csv","text/csv")

st.success("Benchmark complete!")
