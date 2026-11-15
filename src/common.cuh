#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(call)                                                                          \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,     \
                         cudaGetErrorString(err_));                                       \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { CK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void up(const std::vector<T>& h) { CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

// normal(0, sigma) values rounded to fp16
inline std::vector<half> random_half(size_t n, unsigned seed, float sigma = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, sigma);
    std::vector<half> v(n);
    for (half& x : v) x = __float2half(g(rng));
    return v;
}

inline std::vector<double> to_double(const std::vector<half>& v) {
    std::vector<double> d(v.size());
    for (size_t i = 0; i < v.size(); ++i) d[i] = __half2float(v[i]);
    return d;
}
