#pragma once
// FlashAttention-2 forward and backward for head dimension 64, fp16 in and out, fp32
// accumulation, on the tensor cores through WMMA (compute capability 7.0 and up).
//
// Layout: Q, K, V, O and their gradients are [batch * heads][N][64], row-major, contiguous.
// Any N works: keys past N are masked and query rows past N are not written.
//
// The forward pass never builds the N x N score matrix. A block takes 64 queries and walks
// over the keys 64 at a time, keeping a running row maximum m and sum l, and rescales its
// output accumulator whenever the maximum grows (the online softmax). It stores O and the
// log-sum-exp L per query, which the backward pass uses to rebuild the probabilities.
//
// The backward pass gives each block 64 keys. It walks over the query tiles, rebuilds
// P = exp(S - L) for the tile, and accumulates dK and dV for its keys in registers. dQ
// gets contributions from every key block, so those are added into an fp32 buffer with
// atomics and converted to fp16 at the end.
//
// WMMA does not say which accumulator element sits in which thread, so the row-wise
// softmax goes through shared memory: scores are stored there, each pair of lanes owns
// one row, and the probabilities are written back as fp16 for the next multiply.

#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>

namespace flash {

constexpr int D = 64;          // head dimension
constexpr int TILE = 64;       // queries per forward block, keys per backward block, and step size
constexpr int WARPS = 4;       // each warp owns 16 rows of a 64-row tile
constexpr int THREADS = 32 * WARPS;
constexpr int LDH = D + 8;     // fp16 row stride in shared memory (16-byte aligned, fewer conflicts)
constexpr int LDF = TILE + 4;  // fp32 row stride
constexpr float LOG2E = 1.4426950408889634f;

constexpr size_t kTileBytes = TILE * LDH * sizeof(half);
constexpr size_t kWarpScratch = 16 * LDF * sizeof(float);
constexpr size_t kFwdSmem = 3 * kTileBytes + 2 * WARPS * kWarpScratch;                  // Q, K, V, S/P, O
constexpr size_t kBwdSmem = 5 * kTileBytes + WARPS * kWarpScratch + 2 * TILE * sizeof(float);  // Q, dO, K, V, P/dS, scratch, L, D

// 64 rows of 64 halves into shared memory, 16 bytes per load; rows past N become zeros
__device__ __forceinline__ void load_tile(half* dst, const half* src, int row0, int N) {
    for (int i = threadIdx.x; i < TILE * D / 8; i += THREADS) {
        const int r = i / (D / 8), c = (i % (D / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (row0 + r < N) v = *reinterpret_cast<const uint4*>(src + size_t(row0 + r) * D + c);
        *reinterpret_cast<uint4*>(dst + r * LDH + c) = v;
    }
}

#if __CUDA_ARCH__ >= 700  // mma.h only declares nvcuda::wmma when compiling for 7.0 and up
namespace wmma = nvcuda::wmma;
using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
using FragAt = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
using FragBt = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// out (16 x 64) = A (16 x 64, row-major at a) times B^T, where B is 64 x 64 row-major at b:
// the scores Q K^T, or dO V^T
__device__ __forceinline__ void mul_abt(FragC (&out)[4], const half* a, const half* b) {
#pragma unroll
    for (int n = 0; n < 4; ++n) wmma::fill_fragment(out[n], 0.0f);
#pragma unroll
    for (int kk = 0; kk < D; kk += 16) {
        FragA fa;
        wmma::load_matrix_sync(fa, a + kk, LDH);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            FragBt fb;  // B^T as a column-major operand is B row-major
            wmma::load_matrix_sync(fb, b + n * 16 * LDH + kk, LDH);
            wmma::mma_sync(out[n], fa, fb, out[n]);
        }
    }
}

__device__ __forceinline__ void store_rows(float* dst, const FragC (&f)[4]) {
#pragma unroll
    for (int n = 0; n < 4; ++n) wmma::store_matrix_sync(dst + n * 16, f[n], LDF, wmma::mem_row_major);
}
#endif

template <bool Causal>
__global__ void __launch_bounds__(THREADS)
forward(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
        float* __restrict__ L, int N, float scale) {
#if __CUDA_ARCH__ >= 700
    extern __shared__ __align__(32) unsigned char smem[];
    half* Qs = reinterpret_cast<half*>(smem);
    half* Ks = Qs + TILE * LDH;
    half* Vs = Ks + TILE * LDH;
    float* scratch = reinterpret_cast<float*>(Vs + TILE * LDH);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float* Sw = scratch + warp * 16 * LDF;                // this warp's 16 x 64 scores
    half* Pw = reinterpret_cast<half*>(Sw);               // and, written over them, its probabilities
    float* Ow = scratch + (WARPS + warp) * 16 * LDF;      // its 16 x 64 output accumulator
    const int q0 = blockIdx.x * TILE;
    const size_t head = size_t(blockIdx.y) * N * D;
    const int r = lane / 2, c0 = (lane % 2) * 32;         // a lane pair per row, 32 columns each
    const int qrow = q0 + warp * 16 + r;
    const float sl2 = scale * LOG2E;                      // softmax in base 2: exp2 is one instruction

    load_tile(Qs, Q + head, q0, N);
    for (int c = c0; c < c0 + 32; ++c) Ow[r * LDF + c] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    const int tiles = (N + TILE - 1) / TILE;
    const int last = Causal ? min(tiles, blockIdx.x + 1) : tiles;  // causal: later keys are all masked
    for (int j = 0; j < last; ++j) {
        __syncthreads();  // everyone is done with the previous K and V
        load_tile(Ks, K + head, j * TILE, N);
        load_tile(Vs, V + head, j * TILE, N);
        __syncthreads();
        FragC s[4];
        mul_abt(s, Qs + warp * 16 * LDH, Ks);
        store_rows(Sw, s);
        __syncwarp();
        float x[32], mx = -INFINITY;
#pragma unroll
        for (int t = 0; t < 32; ++t) {
            const int key = j * TILE + c0 + t;
            const bool ok = key < N && (!Causal || key <= qrow);
            x[t] = ok ? Sw[r * LDF + c0 + t] * sl2 : -INFINITY;
            mx = fmaxf(mx, x[t]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        const float m_new = fmaxf(m, mx);
        const float m_use = m_new == -INFINITY ? 0.0f : m_new;  // a row with nothing unmasked yet
        const float alpha = exp2f(m - m_use);                   // how much the old terms shrink
        float sum = 0.0f;
#pragma unroll
        for (int t = 0; t < 32; ++t) {
            x[t] = exp2f(x[t] - m_use);
            sum += x[t];
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        l = l * alpha + sum;
        m = m_new;
        __syncwarp();  // every lane has read its scores before the probabilities overwrite them
#pragma unroll
        for (int t = 0; t < 32; ++t) Pw[r * LDH + c0 + t] = __float2half(x[t]);
        for (int c = c0; c < c0 + 32; ++c) Ow[r * LDF + c] *= alpha;
        __syncwarp();
        FragC o[4];
#pragma unroll
        for (int n = 0; n < 4; ++n) wmma::load_matrix_sync(o[n], Ow + n * 16, LDF, wmma::mem_row_major);
#pragma unroll
        for (int kk = 0; kk < TILE; kk += 16) {
            FragA p;
            wmma::load_matrix_sync(p, Pw + kk, LDH);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                FragB v;
                wmma::load_matrix_sync(v, Vs + kk * LDH + n * 16, LDH);
                wmma::mma_sync(o[n], p, v, o[n]);
            }
        }
        store_rows(Ow, o);
        __syncwarp();
    }
    if (qrow < N) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (int c = c0; c < c0 + 32; c += 2)
            *reinterpret_cast<half2*>(&O[head + size_t(qrow) * D + c]) =
                __floats2half2_rn(Ow[r * LDF + c] * inv, Ow[r * LDF + c + 1] * inv);
        if (c0 == 0) L[size_t(blockIdx.y) * N + qrow] = m + log2f(l);  // base 2
    }
#endif
}

// D_i = sum_d dO_id O_id, one value per query row
__global__ void row_dot(const half* __restrict__ dO, const half* __restrict__ O, float* __restrict__ Dv, size_t rows) {
    const size_t row = blockIdx.x * size_t(blockDim.y) + threadIdx.y;
    if (row >= rows) return;
    float s = 0.0f;
    for (int c = threadIdx.x; c < D; c += 32) s += __half2float(dO[row * D + c]) * __half2float(O[row * D + c]);
    for (int o = 16; o > 0; o /= 2) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (threadIdx.x == 0) Dv[row] = s;
}

template <bool Causal>
__global__ void __launch_bounds__(THREADS)
backward(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
         const half* __restrict__ dO, const float* __restrict__ L, const float* __restrict__ Dv,
         float* __restrict__ dQacc, half* __restrict__ dK, half* __restrict__ dV, int N, float scale) {
#if __CUDA_ARCH__ >= 700
    extern __shared__ __align__(32) unsigned char smem[];
    half* Qs = reinterpret_cast<half*>(smem);
    half* dOs = Qs + TILE * LDH;
    half* Ks = dOs + TILE * LDH;
    half* Vs = Ks + TILE * LDH;
    half* PdS = Vs + TILE * LDH;                          // P for the tile, then dS over it
    float* scratch = reinterpret_cast<float*>(PdS + TILE * LDH);
    float* Ls = scratch + WARPS * 16 * LDF;
    float* Ds = Ls + TILE;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float* Sw = scratch + warp * 16 * LDF;
    const int k0 = blockIdx.x * TILE;
    const size_t head = size_t(blockIdx.y) * N * D;
    const int r = lane / 2, c0 = (lane % 2) * 32;
    const float sl2 = scale * LOG2E;

    load_tile(Ks, K + head, k0, N);
    load_tile(Vs, V + head, k0, N);
    FragC dk[4], dv[4];  // this warp's 16 keys x 64 dims of dK and dV
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        wmma::fill_fragment(dk[n], 0.0f);
        wmma::fill_fragment(dv[n], 0.0f);
    }
    const int tiles = (N + TILE - 1) / TILE;
    for (int i = Causal ? blockIdx.x : 0; i < tiles; ++i) {  // causal: earlier queries see none of these keys
        const int q0 = i * TILE;
        __syncthreads();
        load_tile(Qs, Q + head, q0, N);
        load_tile(dOs, dO + head, q0, N);
        for (int t = threadIdx.x; t < TILE; t += THREADS) {
            const bool ok = q0 + t < N;
            Ls[t] = ok ? L[size_t(blockIdx.y) * N + q0 + t] : INFINITY;  // P = 0 for rows past N
            Ds[t] = ok ? Dv[size_t(blockIdx.y) * N + q0 + t] : 0.0f;
        }
        __syncthreads();

        // P for this warp's 16 queries, from the scores and the saved log-sum-exp
        const int q = warp * 16 + r, qrow = q0 + q;
        FragC acc[4];
        mul_abt(acc, Qs + warp * 16 * LDH, Ks);
        store_rows(Sw, acc);
        __syncwarp();
        float p[32];
#pragma unroll
        for (int t = 0; t < 32; ++t) {
            const int key = k0 + c0 + t;
            const bool ok = key < N && (!Causal || key <= qrow);
            p[t] = ok ? exp2f(Sw[r * LDF + c0 + t] * sl2 - Ls[q]) : 0.0f;
            PdS[q * LDH + c0 + t] = __float2half(p[t]);
        }
        __syncwarp();
        // dP = dO V^T for the same rows, into the scratch the scores no longer need
        mul_abt(acc, dOs + warp * 16 * LDH, Vs);
        store_rows(Sw, acc);
        __syncthreads();  // all of P is written

        // dV += P^T dO: this warp's keys against all 64 queries
#pragma unroll
        for (int kk = 0; kk < TILE; kk += 16) {
            FragAt pt;  // P^T as a column-major operand is P row-major
            wmma::load_matrix_sync(pt, PdS + kk * LDH + warp * 16, LDH);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                FragB g;
                wmma::load_matrix_sync(g, dOs + kk * LDH + n * 16, LDH);
                wmma::mma_sync(dv[n], pt, g, dv[n]);
            }
        }
        // dS = P (dP - D), kept in registers until every warp has finished reading P
#pragma unroll
        for (int t = 0; t < 32; ++t) p[t] = p[t] * (Sw[r * LDF + c0 + t] - Ds[q]);
        __syncthreads();
#pragma unroll
        for (int t = 0; t < 32; ++t) PdS[q * LDH + c0 + t] = __float2half(p[t]);
        __syncthreads();

        // dK += dS^T Q for this warp's keys
#pragma unroll
        for (int kk = 0; kk < TILE; kk += 16) {
            FragAt st;
            wmma::load_matrix_sync(st, PdS + kk * LDH + warp * 16, LDH);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                FragB qf;
                wmma::load_matrix_sync(qf, Qs + kk * LDH + n * 16, LDH);
                wmma::mma_sync(dk[n], st, qf, dk[n]);
            }
        }
        // dQ for this warp's 16 queries = dS K, added to the fp32 buffer
#pragma unroll
        for (int n = 0; n < 4; ++n) wmma::fill_fragment(acc[n], 0.0f);
#pragma unroll
        for (int kk = 0; kk < TILE; kk += 16) {
            FragA ds;
            wmma::load_matrix_sync(ds, PdS + warp * 16 * LDH + kk, LDH);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                FragB kf;
                wmma::load_matrix_sync(kf, Ks + kk * LDH + n * 16, LDH);
                wmma::mma_sync(acc[n], ds, kf, acc[n]);
            }
        }
        store_rows(Sw, acc);
        __syncwarp();
        if (qrow < N)
            for (int c = c0; c < c0 + 32; ++c)
                atomicAdd(&dQacc[head + size_t(qrow) * D + c], Sw[r * LDF + c] * scale);
    }
    // dK (times the softmax scale) and dV for this warp's 16 keys
    __syncwarp();
    store_rows(Sw, dk);
    __syncwarp();
    const int key = k0 + warp * 16 + r;
    if (key < N)
        for (int c = c0; c < c0 + 32; c += 2)
            *reinterpret_cast<half2*>(&dK[head + size_t(key) * D + c]) =
                __floats2half2_rn(Sw[r * LDF + c] * scale, Sw[r * LDF + c + 1] * scale);
    __syncwarp();
    store_rows(Sw, dv);
    __syncwarp();
    if (key < N)
        for (int c = c0; c < c0 + 32; c += 2)
            *reinterpret_cast<half2*>(&dV[head + size_t(key) * D + c]) =
                __floats2half2_rn(Sw[r * LDF + c], Sw[r * LDF + c + 1]);
#endif
}

__global__ void to_half(const float* __restrict__ x, half* __restrict__ y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        y[i] = __float2half(x[i]);
}

// ---------------------------------------------------------------------------------------
struct Shape {
    int heads;  // batch * heads
    int N;
};

inline float softmax_scale() { return 1.0f / std::sqrt(float(D)); }

// O and L (log-sum-exp in base 2, heads * N floats)
inline void flash_forward(Shape s, bool causal, const half* Q, const half* K, const half* V, half* O, float* L,
                          cudaStream_t st = 0) {
    dim3 grid((s.N + TILE - 1) / TILE, s.heads);
    if (causal) {
        cudaFuncSetAttribute(forward<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kFwdSmem));
        forward<true><<<grid, THREADS, kFwdSmem, st>>>(Q, K, V, O, L, s.N, softmax_scale());
    } else {
        cudaFuncSetAttribute(forward<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kFwdSmem));
        forward<false><<<grid, THREADS, kFwdSmem, st>>>(Q, K, V, O, L, s.N, softmax_scale());
    }
}

// workspace: heads * N floats for D, and heads * N * 64 floats for the dQ accumulator
inline size_t flash_backward_workspace(Shape s) { return size_t(s.heads) * s.N * (1 + D) * sizeof(float); }

inline void flash_backward(Shape s, bool causal, const half* Q, const half* K, const half* V, const half* O,
                           const half* dO, const float* L, half* dQ, half* dK, half* dV, void* workspace,
                           cudaStream_t st = 0) {
    const size_t rows = size_t(s.heads) * s.N;
    float* Dv = static_cast<float*>(workspace);
    float* dQacc = Dv + rows;
    row_dot<<<(rows + 7) / 8, dim3(32, 8), 0, st>>>(dO, O, Dv, rows);
    cudaMemsetAsync(dQacc, 0, rows * D * sizeof(float), st);
    dim3 grid((s.N + TILE - 1) / TILE, s.heads);
    if (causal) {
        cudaFuncSetAttribute(backward<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kBwdSmem));
        backward<true><<<grid, THREADS, kBwdSmem, st>>>(Q, K, V, dO, L, Dv, dQacc, dK, dV, s.N, softmax_scale());
    } else {
        cudaFuncSetAttribute(backward<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kBwdSmem));
        backward<false><<<grid, THREADS, kBwdSmem, st>>>(Q, K, V, dO, L, Dv, dQacc, dK, dV, s.N, softmax_scale());
    }
    to_half<<<512, 256, 0, st>>>(dQacc, dQ, rows * D);
}

}  // namespace flash
