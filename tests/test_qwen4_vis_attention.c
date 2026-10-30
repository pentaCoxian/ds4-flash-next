/* Qwen3.8 vision attention (metal/qwen4_vision.metal) against a double
 * precision softmax attention on the CPU, at row counts that end on and off
 * the 32-row query tiles and 32-key blocks, then the GPU time of one layer of
 * a 1024-token image (4096 patches, 16 heads of 72). */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint32_t rng = 7;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (float)((int)(rng % 2001) - 1000) / 400.0f;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int check(uint32_t N, uint32_t H, uint32_t D) {
    const uint32_t W = H * D;
    const size_t n = (size_t)N * W;
    float *q = malloc(n * 4), *k = malloc(n * 4), *v = malloc(n * 4), *got = malloc(n * 4);
    double *s = malloc((size_t)N * sizeof(double));
    for (size_t i = 0; i < n; i++) { q[i] = frand(); k[i] = frand(); v[i] = frand(); }
    ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(n * 4), *gk = ds4_gpu_tensor_alloc(n * 4);
    ds4_gpu_tensor *gv = ds4_gpu_tensor_alloc(n * 4), *go = ds4_gpu_tensor_alloc(n * 4);
    const float scale = 1.0f / sqrtf((float)D);
    int ok = gq && gk && gv && go &&
             ds4_gpu_tensor_write(gq, 0, q, n * 4) && ds4_gpu_tensor_write(gk, 0, k, n * 4) &&
             ds4_gpu_tensor_write(gv, 0, v, n * 4) && ds4_gpu_begin_commands() &&
             ds4_gpu_qwen4_vis_attention_tensor(go, gq, gk, gv, N, H, D, scale) &&
             ds4_gpu_end_commands() && ds4_gpu_tensor_read(go, 0, got, n * 4);
    double worst = 0.0;
    for (uint32_t h = 0; ok && h < H; h++) {
        for (uint32_t r = 0; r < N; r++) {
            double mx = -INFINITY, sum = 0.0;
            for (uint32_t key = 0; key < N; key++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < D; d++) dot += (double)q[(size_t)r * W + h * D + d] * k[(size_t)key * W + h * D + d];
                s[key] = dot * scale;
                if (s[key] > mx) mx = s[key];
            }
            for (uint32_t key = 0; key < N; key++) { s[key] = exp(s[key] - mx); sum += s[key]; }
            for (uint32_t d = 0; d < D; d++) {
                double want = 0.0;
                for (uint32_t key = 0; key < N; key++) want += s[key] * v[(size_t)key * W + h * D + d];
                want /= sum;
                const double err = fabs(want - got[(size_t)r * W + h * D + d]) / (1.0 + fabs(want));
                if (!(err <= worst)) worst = err;   /* NaN is the worst */
            }
        }
    }
    if (ok && !(worst < 2e-5)) ok = 0;
    printf("%s Qwen vision attention rows=%u heads=%u dim=%u: max rel error %.2e\n",
           ok ? "PASS" : "FAIL", N, H, D, worst);
    ds4_gpu_tensor_free(gq); ds4_gpu_tensor_free(gk); ds4_gpu_tensor_free(gv); ds4_gpu_tensor_free(go);
    free(q); free(k); free(v); free(got); free(s);
    return ok;
}

static void timing(uint32_t N, uint32_t H, uint32_t D) {
    const size_t n = (size_t)N * H * D;
    ds4_gpu_tensor *t[4];
    for (int i = 0; i < 4; i++) t[i] = ds4_gpu_tensor_alloc(n * 4);
    float *x = malloc(n * 4);
    for (size_t i = 0; i < n; i++) x[i] = frand();
    for (int i = 0; i < 3; i++) ds4_gpu_tensor_write(t[i], 0, x, n * 4);
    double best = INFINITY;
    for (int rep = 0; rep < 3; rep++) {
        const double t0 = now_s();
        ds4_gpu_begin_commands();
        ds4_gpu_qwen4_vis_attention_tensor(t[3], t[0], t[1], t[2], N, H, D, 1.0f / sqrtf((float)D));
        ds4_gpu_end_commands();
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    printf("Qwen vision attention rows=%u heads=%u dim=%u: %.1f ms per layer\n", N, H, D, best * 1e3);
    for (int i = 0; i < 4; i++) ds4_gpu_tensor_free(t[i]);
    free(x);
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    const uint32_t rows[] = {1, 7, 31, 32, 33, 100, 257, 1000};
    int ok = 1;
    for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]) && ok; i++) ok = check(rows[i], 4, 72);
    for (unsigned i = 0; i < 3 && ok; i++) ok = check(rows[i * 3 + 1], 4, 64);
    if (ok) timing(4096, 16, 72);
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
