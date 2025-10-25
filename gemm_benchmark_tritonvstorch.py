# the_app.py
import time
import platform
import streamlit as st
import pandas as pd
import torch

# ---------- Triton ----------
import triton
import triton.language as tl

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="Triton GEMM — Tutorial kernel + variants", layout="wide")
st.title("🚀 Triton GEMM — Tutorial kernel + variants")
st.caption(
    "Baseline kernel from the official Triton tutorial, with optional: Tensor Cores, "
    "grouped reuse mapping, and fused epilogue (bias + activation)."
)

# --------- ENV CHECKS ----------
if not torch.cuda.is_available():
    st.error("No CUDA GPU detected. This app requires an NVIDIA GPU with CUDA.")
    st.stop()

st.sidebar.header("System")
st.sidebar.write("Python:", platform.python_version())
st.sidebar.write("PyTorch:", torch.__version__)
st.sidebar.write("GPU:", torch.cuda.get_device_name(0))

# --------- HELPERS ----------
def gflops(M, N, K, sec, batches=1):
    return (2.0 * batches * M * N * K) / (sec * 1e9)

def total_flops(M, N, K, batches=1):
    return 2 * batches * M * N * K

def time_op(fn, warmup=5, iters=30, sync=torch.cuda.synchronize):
    for _ in range(warmup):
        fn(); sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(); sync()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def apply_activation(x, kind: str):
    if kind == "none":
        return x
    if kind == "relu":
        return torch.nn.functional.relu(x)
    if kind == "gelu":
        return torch.nn.functional.gelu(x)
    if kind == "silu":
        return torch.nn.functional.silu(x)
    raise ValueError(kind)

# =========================================================
#  BASELINE KERNEL (from Triton tutorial)
#  https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
# =========================================================
@triton.jit
def matmul_kernel_tutorial(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k0 + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k0 + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_gemm_tutorial(A, B, block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=2):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_kernel_tutorial[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages
    )
    return C

# =========================================================
#  VARIANT 1: Grouped mapping (L2 reuse)
# =========================================================
@triton.jit
def matmul_kernel_tutorial_grouped(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # 1D program id, grouped across M for better cache reuse
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_id = pid // grid_n
    first_m = (group_id * GROUP_M) % grid_m
    group_size = tl.minimum(grid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size)
    pid_n = (pid % grid_n)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k0 + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k0 + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_gemm_tutorial_grouped(A, B, block_m=128, block_n=128, block_k=32, group_m=8, num_warps=4, num_stages=2):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    matmul_kernel_tutorial_grouped[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps, num_stages=num_stages
    )
    return C

# =========================================================
#  VARIANT 2: Fused epilogue (bias + activation) — GELU uses erf
# =========================================================
@triton.jit
def matmul_kernel_tutorial_fused(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACT_CODE: tl.constexpr,  # 0=none,1=relu,2=gelu,3=silu
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k0 + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k0 + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Bias + activation epilogue (vector bias over N)
    bias_vec = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)[None, :]
    acc = acc + bias_vec

    if ACT_CODE == 1:      # ReLU
        acc = tl.maximum(acc, 0)
    elif ACT_CODE == 2:    # GELU (erf-based)
        inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
        acc = 0.5 * acc * (1 + tl.math.erf(acc * inv_sqrt2))
    elif ACT_CODE == 3:    # SiLU
        acc = acc / (1 + tl.exp(-acc))
    # else: no activation

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_gemm_tutorial_fused(
    A, B, bias,
    block_m=128, block_n=128, block_k=32,
    num_warps=4, num_stages=2, act="gelu"
):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert bias is not None and bias.is_cuda and bias.dtype == torch.float32 and bias.numel() == N
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    act_map = {"none": 0, "relu": 1, "gelu": 2, "silu": 3}
    act_code = act_map[act]

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_kernel_tutorial_fused[grid](
        A, B, bias, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        ACT_CODE=act_code,
        num_warps=num_warps, num_stages=max(2, num_stages)
    )
    return C

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Config")
dtype_name = st.sidebar.selectbox("Precision (inputs)", ["float16", "bfloat16", "float32"], index=0)
dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
dtype = dtype_map[dtype_name]

M = st.sidebar.number_input("M", min_value=1, value=1024, step=64)
K = st.sidebar.number_input("K", min_value=1, value=1024, step=64)
N = st.sidebar.number_input("N", min_value=1, value=1024, step=64)

st.sidebar.markdown("**Tiling (advanced)**")
bm = st.sidebar.selectbox("BLOCK_M", [64, 128, 256], index=1)
bn = st.sidebar.selectbox("BLOCK_N", [64, 128, 256], index=1)
bk_base = st.sidebar.selectbox("BLOCK_K", [16, 32, 64, 128], index=2)
warps = st.sidebar.selectbox("num_warps", [2, 4, 8], index=2)
stages = st.sidebar.selectbox("num_stages", [1, 2, 3, 4], index=2)

st.sidebar.markdown("**Reuse mapping**")
use_grouped = st.sidebar.checkbox("Enable grouped mapping (GROUP_M)", value=True)
group_m = st.sidebar.selectbox("GROUP_M (if grouped)", [1, 4, 8, 16], index=2, disabled=not use_grouped)

st.sidebar.markdown("**Fused epilogue**")
use_fused = st.sidebar.checkbox("Fuse bias + activation", value=True)
activation = st.sidebar.selectbox("Activation", ["gelu", "relu", "silu", "none"], index=0, disabled=not use_fused)
compare_all_acts = st.sidebar.checkbox("Also compare all activations", value=False, disabled=not use_fused)

# Guardrail: Tensor Cores hints
if dtype_name in ("float16", "bfloat16"):
    bad = []
    if bm % 16: bad.append("BLOCK_M")
    if bn % 16: bad.append("BLOCK_N")
    if bk_base % 16: bad.append("BLOCK_K")
    if bad:
        st.warning("For Tensor Cores, set " + ", ".join(bad) + " to multiples of 16.")

run_btn = st.sidebar.button("Run", type="primary")

# ---------------- MAIN ACTION ----------------
if run_btn:
    device = "cuda"
    torch.manual_seed(0)

    # Inputs
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=torch.float32) if use_fused else None

    # --- torch baselines ---
    # 1) plain GEMM
    t_torch = time_op(lambda: A @ B)
    C_ref = (A @ B).float()

    # 2) fused baseline: GEMM + bias + activation (in PyTorch)
    def torch_fused():
        Y = (A @ B).float()
        if bias is not None:
            Y = Y + bias
        return apply_activation(Y, activation if use_fused else "none")
    t_torch_fused = time_op(torch_fused)
    C_ref_fused = torch_fused()

    rows = []
    def add_row(name, t, C, ref, tag, act="none"):
        diff = (C - ref)
        rel_err = (diff.norm() / (ref.norm() + 1e-12)).item()
        rows.append({
            "Impl": name,
            "Tag": tag,
            "Activation": act,
            "M": M, "N": N, "K": K,
            "Precision": str(dtype).split(".")[-1],
            "Time (s)": t,
            "GFLOP/s": gflops(M, N, K, t),
            "Rel Error vs ref": rel_err,
            "Total FLOPs": total_flops(M, N, K)
        })

    # ---- Triton tutorial baseline ----
    t_tri_base = time_op(lambda: triton_gemm_tutorial(A, B, bm, bn, bk_base, warps, stages))
    C_tri_base = triton_gemm_tutorial(A, B, bm, bn, bk_base, warps, stages)
    add_row("Triton tutorial (baseline)", t_tri_base, C_tri_base, C_ref, "baseline")

    # ---- Tensor Core–friendly tiles (for fp16/bf16) ----
    # Uses the same tutorial kernel, but with 16-multiple tiles.
    bk_tc = max(16, (bk_base // 16) * 16)
    bm_tc = bm - (bm % 16)
    bn_tc = bn - (bn % 16)
    if bm_tc == 0: bm_tc = bm
    if bn_tc == 0: bn_tc = bn
    t_tri_tc = time_op(lambda: triton_gemm_tutorial(A, B, bm_tc, bn_tc, bk_tc, warps, max(2, stages)))
    C_tri_tc = triton_gemm_tutorial(A, B, bm_tc, bn_tc, bk_tc, warps, max(2, stages))
    add_row("Triton tutorial (TC-friendly tiles)", t_tri_tc, C_tri_tc, C_ref, "A:TC")

    # ---- Grouped mapping (reuse) variant ----
    if use_grouped:
        t_tri_grouped = time_op(lambda: triton_gemm_tutorial_grouped(
            A, B, bm_tc, bn_tc, bk_tc, group_m, warps, max(2, stages)
        ))
        C_tri_grouped = triton_gemm_tutorial_grouped(
            A, B, bm_tc, bn_tc, bk_tc, group_m, warps, max(2, stages)
        )
        add_row("Triton tutorial (TC + grouped reuse)", t_tri_grouped, C_tri_grouped, C_ref, "C:reuse")
    else:
        t_tri_grouped, C_tri_grouped = None, None

    # ---- Fused epilogue (bias + activation) ----
    if use_fused:
        t_tri_fused = time_op(lambda: triton_gemm_tutorial_fused(
            A, B, bias, bm_tc, bn_tc, bk_tc, warps, max(2, stages), activation
        ))
        C_tri_fused = triton_gemm_tutorial_fused(
            A, B, bias, bm_tc, bn_tc, bk_tc, warps, max(2, stages), activation
        )
        add_row(f"Triton tutorial (TC + fused {activation})", t_tri_fused, C_tri_fused, C_ref_fused, "B:fused", activation)

        # Optional: compare all activations
        if compare_all_acts:
            for act_kind in ["relu", "gelu", "silu", "none"]:
                t_tmp = time_op(lambda: triton_gemm_tutorial_fused(
                    A, B, bias, bm_tc, bn_tc, bk_tc, warps, max(2, stages), act_kind
                ))
                C_tmp = triton_gemm_tutorial_fused(
                    A, B, bias, bm_tc, bn_tc, bk_tc, warps, max(2, stages), act_kind
                )
                ref_tmp = apply_activation((A @ B).float() + bias, act_kind)
                add_row(f"Triton tutorial (TC + fused {act_kind})", t_tmp, C_tmp, ref_tmp, "B:fused", act_kind)

    # ---- Torch rows (for reference) ----
    rows.insert(0, {
        "Impl": "torch.mm (baseline)",
        "Tag": "torch",
        "Activation": "none",
        "M": M, "N": N, "K": K,
        "Precision": str(dtype).split(".")[-1],
        "Time (s)": t_torch,
        "GFLOP/s": gflops(M, N, K, t_torch),
        "Rel Error vs ref": 0.0,
        "Total FLOPs": total_flops(M, N, K),
    })
    rows.append({
        "Impl": f"torch + bias + {activation if use_fused else 'none'}",
        "Tag": "torch_fused",
        "Activation": activation if use_fused else "none",
        "M": M, "N": N, "K": K,
        "Precision": str(dtype).split(".")[-1],
        "Time (s)": t_torch_fused,
        "GFLOP/s": gflops(M, N, K, t_torch_fused),
        "Rel Error vs ref": 0.0,
        "Total FLOPs": total_flops(M, N, K),
    })

    df = pd.DataFrame(rows)
    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    # Quick deltas to show improvement
    def speedup(t_ref, t_new):
        return t_ref / t_new if (t_new is not None and t_new > 0) else float("inf")

    # speedups
    s_tc = speedup(t_torch, t_tri_tc)
    s_reuse = speedup(t_torch, t_tri_grouped if t_tri_grouped is not None else float("nan"))
    # find fused impl time matching the chosen activation
    fused_impl_times = [r["Time (s)"] for r in rows if r["Tag"] == "B:fused" and "Triton" in r["Impl"] and r["Activation"] == (activation if use_fused else "none")]
    s_fused = speedup(t_torch_fused, fused_impl_times[0]) if fused_impl_times else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("Speedup (TC-tiles vs torch.mm)", f"{s_tc:.2f}×")
    c2.metric("Speedup (TC+grouped vs torch.mm)", f"{s_reuse:.2f}×" if use_grouped else "—")
    c3.metric(f"Speedup (TC+fused {activation} vs torch+epilogue)", f"{s_fused:.2f}×" if use_fused else "—")

    st.metric("Rel error (TC-tiles vs torch.mm)", f"{((C_tri_tc - C_ref).norm() / (C_ref.norm()+1e-12)).item():.3e}")
    if use_fused and fused_impl_times:
        rel_fused = ((C_tri_fused - C_ref_fused).norm() / (C_ref_fused.norm()+1e-12)).item()
        st.metric("Rel error (TC+fused vs torch+epilogue)", f"{rel_fused:.3e}")

    # -------- Charts --------
    import altair as alt
    st.subheader("Performance (GFLOP/s) — by Implementation")
    chart_main = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Impl:N", title="Implementation", sort="-y"),
            y=alt.Y("GFLOP/s:Q", title="Throughput (GFLOP/s)"),
            color="Tag:N",
            tooltip=["Impl","Tag","Activation","M","N","K","Precision","GFLOP/s","Time (s)","Rel Error vs ref"]
        )
        .properties(width=280, height=340)
    )
    st.altair_chart(chart_main, use_container_width=True)

    df_acts = df[df["Tag"] == "B:fused"]
    if len(df_acts) > 0:
        st.subheader("Fused Epilogue — Activation Breakdown (GFLOP/s)")
        chart_act = (
            alt.Chart(df_acts)
            .mark_bar()
            .encode(
                x=alt.X("Activation:N", title="Activation"),
                y=alt.Y("GFLOP/s:Q", title="Throughput (GFLOP/s)"),
                color="Activation:N",
                column=alt.Column("Impl:N", title="Implementation"),
                tooltip=["Impl","Activation","GFLOP/s","Time (s)"]
            )
            .properties(height=320)
        )
        st.altair_chart(chart_act, use_container_width=True)

else:
    st.info(
        "Choose dtype, sizes, and options in the sidebar, then click **Run**.\n\n"
        "Notes:\n"
        "- **Baseline** is the Triton tutorial GEMM kernel (tile loop with `tl.dot`).\n"
        "- **Tensor Cores** kick in when inputs are fp16/bf16 **and** tiles are multiples of 16.\n"
        "- **Grouped mapping** improves L2 locality (mapping multiple M-tiles per group).\n"
        "- **Fused epilogue** performs bias + activation inside the kernel."
    )
