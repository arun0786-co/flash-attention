#pragma once
// Standard attention, the way a framework's plain path does it: the scores S = Q K^T go to
// memory as an N x N matrix per head, a softmax kernel turns them into P, and O = P V. The
// backward pass keeps P and builds dP and dS, N x N again. The matrix multiplies are cuBLAS
// (fp16 inputs, fp32 accumulation); S and dP are kept in fp32, P and dS in fp16.
//
// cuBLAS is column-major and these matrices are row-major. A row-major r x c matrix is a
// column-major c x r one, so each product below is written for the transposes.

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "flash.cuh"

namespace standard {

using flash::D;
using flash::Shape;

#define CB(call)                                                                          \
    do {                                                                                  \
        cublasStatus_t st_ = (call);                                                      \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                               \
            std::fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #call,     \
                         static_cast<int>(st_));                                          \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

// row-major C (N x N, fp32) = alpha * X Y^T, X and Y N x 64: the scores, or dP = dO V^T
inline void abt(cublasHandle_t h, Shape s, float alpha, const half* X, const half* Y, float* C) {
    const float beta = 0.0f;
    const long long nd = static_cast<long long>(s.N) * D, nn = static_cast<long long>(s.N) * s.N;
    CB(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.N, s.N, D, &alpha, Y, CUDA_R_16F, D, nd, X,
                                  CUDA_R_16F, D, nd, &beta, C, CUDA_R_32F, s.N, nn, s.heads, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
}

// row-major C (N x 64, fp16) = alpha * P Y, P N x N fp16: the output O = P V, or dQ = dS K
inline void pv(cublasHandle_t h, Shape s, float alpha, const half* P, const half* Y, half* C) {
    const float beta = 0.0f;
    const long long nd = static_cast<long long>(s.N) * D, nn = static_cast<long long>(s.N) * s.N;
    CB(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, s.N, s.N, &alpha, Y, CUDA_R_16F, D, nd, P,
                                  CUDA_R_16F, s.N, nn, &beta, C, CUDA_R_16F, D, nd, s.heads, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
}

// row-major C (N x 64, fp16) = alpha * P^T Y: dV = P^T dO, or dK = dS^T Q
inline void ptv(cublasHandle_t h, Shape s, float alpha, const half* P, const half* Y, half* C) {
    const float beta = 0.0f;
    const long long nd = static_cast<long long>(s.N) * D, nn = static_cast<long long>(s.N) * s.N;
    CB(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_T, D, s.N, s.N, &alpha, Y, CUDA_R_16F, D, nd, P,
                                  CUDA_R_16F, s.N, nn, &beta, C, CUDA_R_16F, D, nd, s.heads, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
}

// one block of 256 threads per row: P = softmax(S) with the causal mask
template <bool Causal>
__global__ void softmax(const float* __restrict__ S, half* __restrict__ P, int N) {
    const size_t row = blockIdx.x;
    const int q = static_cast<int>(row % N);
    const float* s = S + row * N;
    half* p = P + row * N;
    __shared__ float red[32];
    auto block_reduce = [&](float v, bool is_max) {
        for (int o = 16; o > 0; o /= 2) {
            float w = __shfl_xor_sync(0xffffffffu, v, o);
            v = is_max ? fmaxf(v, w) : v + w;
        }
        if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
        __syncthreads();
        v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : (is_max ? -INFINITY : 0.0f);
        if (threadIdx.x < 32)
            for (int o = 16; o > 0; o /= 2) {
                float w = __shfl_xor_sync(0xffffffffu, v, o);
                v = is_max ? fmaxf(v, w) : v + w;
            }
        if (threadIdx.x == 0) red[0] = v;
        __syncthreads();
        v = red[0];
        __syncthreads();
        return v;
    };
    float mx = -INFINITY;
    for (int k = threadIdx.x; k < N; k += blockDim.x)
        if (!Causal || k <= q) mx = fmaxf(mx, s[k]);
    mx = block_reduce(mx, true);
    float sum = 0.0f;
    for (int k = threadIdx.x; k < N; k += blockDim.x)
        if (!Causal || k <= q) sum += __expf(s[k] - mx);
    sum = block_reduce(sum, false);
    const float inv = 1.0f / sum;
    for (int k = threadIdx.x; k < N; k += blockDim.x)
        p[k] = __float2half(!Causal || k <= q ? __expf(s[k] - mx) * inv : 0.0f);
}

// dS = P (dP - D_row), written over P
__global__ void softmax_grad(half* __restrict__ PdS, const float* __restrict__ dP, const float* __restrict__ Dv,
                             int N, size_t total) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x)
        PdS[i] = __float2half(__half2float(PdS[i]) * (dP[i] - Dv[i / N]));
}

// fp32 S (or dP) and fp16 P (or dS), each N x N per head
inline size_t workspace(Shape s) { return size_t(s.heads) * s.N * s.N * (sizeof(float) + sizeof(half)); }

struct Buffers {
    float* S;  // scores, then dP
    half* P;   // probabilities, then dS
    explicit Buffers(void* ws, Shape s)
        : S(static_cast<float*>(ws)), P(reinterpret_cast<half*>(static_cast<float*>(ws) + size_t(s.heads) * s.N * s.N)) {}
};

inline void forward(cublasHandle_t h, Shape s, bool causal, const half* Q, const half* K, const half* V, half* O,
                    void* ws) {
    Buffers b(ws, s);
    abt(h, s, flash::softmax_scale(), Q, K, b.S);
    if (causal)
        softmax<true><<<size_t(s.heads) * s.N, 256>>>(b.S, b.P, s.N);
    else
        softmax<false><<<size_t(s.heads) * s.N, 256>>>(b.S, b.P, s.N);
    pv(h, s, 1.0f, b.P, V, O);
}

// needs the workspace as forward() left it (P), plus heads * N floats for D
inline void backward(cublasHandle_t h, Shape s, const half* Q, const half* K, const half* V, const half* O,
                     const half* dO, half* dQ, half* dK, half* dV, void* ws, float* Dv) {
    Buffers b(ws, s);
    const size_t rows = size_t(s.heads) * s.N;
    flash::row_dot<<<(rows + 7) / 8, dim3(32, 8)>>>(dO, O, Dv, rows);
    ptv(h, s, 1.0f, b.P, dO, dV);       // dV = P^T dO
    abt(h, s, 1.0f, dO, V, b.S);        // dP = dO V^T
    softmax_grad<<<1024, 256>>>(b.P, b.S, Dv, s.N, rows * s.N);
    pv(h, s, flash::softmax_scale(), b.P, K, dQ);   // dQ = scale dS K
    ptv(h, s, flash::softmax_scale(), b.P, Q, dK);  // dK = scale dS^T Q
}

}  // namespace standard
