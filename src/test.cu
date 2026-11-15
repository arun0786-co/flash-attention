// attn_test: flash and standard attention against a float64 reference on the CPU, for the
// forward output, the log-sum-exp and all three gradients, with and without the causal
// mask, for sequence lengths that are and are not multiples of the 64-row tiles.
//
// Errors are the largest absolute difference divided by the largest absolute reference
// value, or by 1 when that is smaller (for a single key the true dQ and dK are exactly 0, so a
// relative error would divide by nothing). fp16 storage alone gives about 5e-4 (half of 2^-10). Both methods must stay under
// a fixed limit, and flash must be no worse than twice the standard method plus 1e-3: the
// point is that it is exact attention, as accurate as the usual way.

#include <cmath>

#include "common.cuh"
#include "flash.cuh"
#include "standard.cuh"

using flash::D;

struct Ref {
    std::vector<double> O, L, dQ, dK, dV;
};

static Ref reference(int heads, int N, bool causal, const std::vector<double>& Q, const std::vector<double>& K,
                     const std::vector<double>& V, const std::vector<double>& dO) {
    Ref r;
    const size_t nd = size_t(N) * D;
    r.O.assign(heads * nd, 0);
    r.dQ.assign(heads * nd, 0);
    r.dK.assign(heads * nd, 0);
    r.dV.assign(heads * nd, 0);
    r.L.assign(size_t(heads) * N, 0);
    const double scale = 1.0 / std::sqrt(double(D));
    std::vector<double> P(size_t(N) * N), dP(size_t(N) * N);
    for (int h = 0; h < heads; ++h) {
        const double *q = &Q[h * nd], *k = &K[h * nd], *v = &V[h * nd], *g = &dO[h * nd];
        for (int i = 0; i < N; ++i) {
            double mx = -INFINITY;
            for (int j = 0; j < N; ++j) {
                double s = 0;
                for (int c = 0; c < D; ++c) s += q[i * D + c] * k[j * D + c];
                P[size_t(i) * N + j] = (causal && j > i) ? -INFINITY : s * scale;
                mx = std::max(mx, P[size_t(i) * N + j]);
            }
            double sum = 0;
            for (int j = 0; j < N; ++j) sum += P[size_t(i) * N + j] = std::exp(P[size_t(i) * N + j] - mx);
            for (int j = 0; j < N; ++j) P[size_t(i) * N + j] /= sum;
            r.L[size_t(h) * N + i] = (mx + std::log(sum)) / std::log(2.0);
            for (int c = 0; c < D; ++c) {
                double o = 0;
                for (int j = 0; j < N; ++j) o += P[size_t(i) * N + j] * v[j * D + c];
                r.O[h * nd + i * D + c] = o;
            }
        }
        for (int i = 0; i < N; ++i) {
            double Di = 0;
            for (int c = 0; c < D; ++c) Di += g[i * D + c] * r.O[h * nd + i * D + c];
            for (int j = 0; j < N; ++j) {
                double s = 0;
                for (int c = 0; c < D; ++c) s += g[i * D + c] * v[j * D + c];
                dP[size_t(i) * N + j] = P[size_t(i) * N + j] * (s - Di);  // dS
            }
        }
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const double p = P[size_t(i) * N + j], ds = dP[size_t(i) * N + j];
                for (int c = 0; c < D; ++c) {
                    r.dV[h * nd + j * D + c] += p * g[i * D + c];
                    r.dQ[h * nd + i * D + c] += scale * ds * k[j * D + c];
                    r.dK[h * nd + j * D + c] += scale * ds * q[i * D + c];
                }
            }
    }
    return r;
}

template <class T>
static double rel_err(const std::vector<T>& got, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double g;
        if constexpr (sizeof(T) == 2) g = __half2float(got[i]); else g = got[i];
        if (!std::isfinite(g)) return INFINITY;
        num = std::max(num, std::fabs(g - ref[i]));
        den = std::max(den, std::fabs(ref[i]));
    }
    return num / std::max(den, 1.0);  // with one key, dQ and dK are exactly 0: fall back to absolute error
}

static int failures = 0;

static void expect(const char* what, int heads, int N, bool causal, double flash_err, double std_err, double limit) {
    const bool ok = flash_err <= limit && std_err <= limit && flash_err <= 2 * std_err + 1e-3;
    std::printf("%s %-3s N=%5d %-7s flash %.2e  standard %.2e  (limit %.0e)\n", ok ? "ok  " : "FAIL", what, N,
                causal ? "causal" : "full", flash_err, std_err, limit);
    (void)heads;
    failures += !ok;
}

static void run_case(cublasHandle_t h, int heads, int N, bool causal, unsigned seed) {
    const size_t n = size_t(heads) * N * D;
    auto q = random_half(n, seed), k = random_half(n, seed + 1), v = random_half(n, seed + 2), g = random_half(n, seed + 3);
    Ref ref = reference(heads, N, causal, to_double(q), to_double(k), to_double(v), to_double(g));
    Dev<half> Q(n), K(n), V(n), dO(n), O(n), dQ(n), dK(n), dV(n), O2(n), dQ2(n), dK2(n), dV2(n);
    Q.up(q);
    K.up(k);
    V.up(v);
    dO.up(g);
    flash::Shape s{heads, N};
    Dev<float> L(size_t(heads) * N), Dv(size_t(heads) * N);
    Dev<char> fws(flash::flash_backward_workspace(s)), sws(standard::workspace(s));
    flash::flash_forward(s, causal, Q.p, K.p, V.p, O.p, L.p);
    flash::flash_backward(s, causal, Q.p, K.p, V.p, O.p, dO.p, L.p, dQ.p, dK.p, dV.p, fws.p);
    standard::forward(h, s, causal, Q.p, K.p, V.p, O2.p, sws.p);
    standard::backward(h, s, Q.p, K.p, V.p, O2.p, dO.p, dQ2.p, dK2.p, dV2.p, sws.p, Dv.p);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    expect("O", heads, N, causal, rel_err(O.down(), ref.O), rel_err(O2.down(), ref.O), 4e-3);
    expect("dQ", heads, N, causal, rel_err(dQ.down(), ref.dQ), rel_err(dQ2.down(), ref.dQ), 1e-2);
    expect("dK", heads, N, causal, rel_err(dK.down(), ref.dK), rel_err(dK2.down(), ref.dK), 1e-2);
    expect("dV", heads, N, causal, rel_err(dV.down(), ref.dV), rel_err(dV2.down(), ref.dV), 1e-2);
    // the log-sum-exp the backward pass relies on, in absolute terms (base 2)
    auto l = L.down();
    double worst = 0;
    for (size_t i = 0; i < l.size(); ++i) worst = std::max(worst, std::fabs(l[i] - ref.L[i]));
    const bool ok = worst <= 2e-3;
    std::printf("%s L   N=%5d %-7s flash %.2e absolute (limit 2e-03)\n", ok ? "ok  " : "FAIL", N, causal ? "causal" : "full", worst);
    failures += !ok;
}

int main() {
    int major = 0;
    CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    if (major < 7) {
        std::printf("needs tensor cores (compute 7.0 or newer)\n");
        return 1;
    }
    cublasHandle_t h;
    CB(cublasCreate(&h));
    unsigned seed = 1;
    for (int N : {1, 17, 64, 100, 128, 200, 333})
        for (bool causal : {false, true}) run_case(h, 3, N, causal, seed += 5);
    CB(cublasDestroy(h));
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
