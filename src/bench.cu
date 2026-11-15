// attn_bench: time and memory of flash and standard attention as the sequence grows.
//
//   attn_bench --out results/rtx2080ti
//
// 16 heads (for example batch 2 x 8 heads) of dimension 64, fp16. For each sequence length
// and mask: forward and backward times (median of repeated runs after a warm-up), the
// extra memory each method needs beyond inputs, outputs and gradients, and the largest
// difference between the two outputs. Standard attention is skipped once its N x N buffers
// no longer fit.

#include <sys/stat.h>

#include <cctype>
#include <cstring>
#include <string>

#include "common.cuh"
#include "flash.cuh"
#include "standard.cuh"

using flash::D;

template <class F>
static double median_ms(int reps, F&& f) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a));
        f();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms);
    }
    CK(cudaGetLastError());
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static std::string slug(std::string s) {
    std::string out;
    for (char c : s)
        if (std::isalnum(static_cast<unsigned char>(c))) out += static_cast<char>(std::tolower(c));
    for (const char* drop : {"nvidia", "geforce"}) {
        auto p = out.find(drop);
        if (p != std::string::npos) out.erase(p, std::strlen(drop));
    }
    return out;
}

static double max_diff(const Dev<half>& a, const Dev<half>& b) {
    auto x = a.down(), y = b.down();
    double m = 0;
    for (size_t i = 0; i < x.size(); ++i) m = std::max(m, double(std::fabs(__half2float(x[i]) - __half2float(y[i]))));
    return m;
}

int main(int argc, char** argv) {
    std::string out;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::printf("needs tensor cores (compute 7.0 or newer)\n");
        return 1;
    }
    if (out.empty()) out = "results/" + slug(prop.name);
    mkdir("results", 0755);
    mkdir(out.c_str(), 0755);
    FILE* f = std::fopen((out + "/attention.csv").c_str(), "w");
    std::fprintf(f, "method,causal,heads,n,fwd_ms,bwd_ms,fwd_tflops,bwd_tflops,extra_mb,max_diff_o,max_diff_dq\n");
    cublasHandle_t h;
    CB(cublasCreate(&h));
    const int heads = 16;
    for (int N : {256, 512, 1024, 2048, 4096, 8192, 16384}) {
        for (bool causal : {false, true}) {
            const flash::Shape s{heads, N};
            const size_t n = size_t(heads) * N * D;
            Dev<half> Q(n), K(n), V(n), dO(n), O(n), dQ(n), dK(n), dV(n);
            Q.up(random_half(n, 1));
            K.up(random_half(n, 2));
            V.up(random_half(n, 3));
            dO.up(random_half(n, 4));
            // matmul flops: 2 products forward, 5 backward; the causal mask skips about half
            const double fwd_flop = 4.0 * heads * double(N) * N * D * (causal ? 0.5 : 1.0);
            const double bwd_flop = 2.5 * fwd_flop;
            const int reps = N <= 2048 ? 20 : 5;

            Dev<float> L(size_t(heads) * N);
            Dev<char> fws(flash::flash_backward_workspace(s));
            flash::flash_forward(s, causal, Q.p, K.p, V.p, O.p, L.p);  // warm up
            flash::flash_backward(s, causal, Q.p, K.p, V.p, O.p, dO.p, L.p, dQ.p, dK.p, dV.p, fws.p);
            const double ff = median_ms(reps, [&] { flash::flash_forward(s, causal, Q.p, K.p, V.p, O.p, L.p); });
            const double fb = median_ms(reps, [&] {
                flash::flash_backward(s, causal, Q.p, K.p, V.p, O.p, dO.p, L.p, dQ.p, dK.p, dV.p, fws.p);
            });
            const double flash_mb = (L.n * sizeof(float) + fws.n) / 1e6;

            std::string std_row;
            const size_t need = standard::workspace(s);
            size_t free_b = 0, total_b = 0;
            CK(cudaMemGetInfo(&free_b, &total_b));
            double diff_o = -1, diff_dq = -1;
            if (need + (size_t(256) << 20) < free_b) {
                Dev<char> ws(need);
                Dev<float> Dv(size_t(heads) * N);
                Dev<half> O2(n), dQ2(n), dK2(n), dV2(n);
                standard::forward(h, s, causal, Q.p, K.p, V.p, O2.p, ws.p);  // warm up
                standard::backward(h, s, Q.p, K.p, V.p, O2.p, dO.p, dQ2.p, dK2.p, dV2.p, ws.p, Dv.p);
                const double sf = median_ms(reps, [&] { standard::forward(h, s, causal, Q.p, K.p, V.p, O2.p, ws.p); });
                // backward overwrites P with dS, so each timed backward follows an untimed forward
                std::vector<float> tb;
                for (int r = 0; r < reps; ++r) {
                    standard::forward(h, s, causal, Q.p, K.p, V.p, O2.p, ws.p);
                    tb.push_back(float(median_ms(1, [&] {
                        standard::backward(h, s, Q.p, K.p, V.p, O2.p, dO.p, dQ2.p, dK2.p, dV2.p, ws.p, Dv.p);
                    })));
                }
                std::sort(tb.begin(), tb.end());
                const double sb = tb[tb.size() / 2];
                diff_o = max_diff(O, O2);
                diff_dq = max_diff(dQ, dQ2);
                const double std_mb = (need + Dv.n * sizeof(float)) / 1e6;
                char line[256];
                std::snprintf(line, sizeof line, "standard,%d,%d,%d,%.4f,%.4f,%.2f,%.2f,%.1f,,\n", causal, heads, N, sf, sb,
                              fwd_flop / (sf * 1e9), bwd_flop / (sb * 1e9), std_mb);
                std_row = line;
                std::printf("N=%6d %-6s standard fwd %8.3f ms  bwd %8.3f ms  extra %8.1f MB\n", N,
                            causal ? "causal" : "full", sf, sb, std_mb);
            } else {
                char line[256];
                std::snprintf(line, sizeof line, "standard,%d,%d,%d,,,,,%.1f,,\n", causal, heads, N, need / 1e6);
                std_row = line;
                std::printf("N=%6d %-6s standard needs %.1f GB, does not fit\n", N, causal ? "causal" : "full", need / 1e9);
            }
            std::fprintf(f, "flash,%d,%d,%d,%.4f,%.4f,%.2f,%.2f,%.1f,%.5f,%.5f\n", causal, heads, N, ff, fb,
                         fwd_flop / (ff * 1e9), bwd_flop / (fb * 1e9), flash_mb, diff_o, diff_dq);
            std::fputs(std_row.c_str(), f);
            std::fflush(f);
            std::printf("N=%6d %-6s flash    fwd %8.3f ms  bwd %8.3f ms  extra %8.1f MB  |O diff| %.4f\n", N,
                        causal ? "causal" : "full", ff, fb, flash_mb, diff_o);
            std::fflush(stdout);
        }
    }
    std::fclose(f);
    CB(cublasDestroy(h));
    return 0;
}
