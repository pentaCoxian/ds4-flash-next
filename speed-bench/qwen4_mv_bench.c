/* Decode matvec bandwidth of a model's dense weights, by weight type.
 *
 *   qwen4_mv_bench MODEL.gguf TENSORS.tsv [REGEX] [REPS] [TOKENS]
 *
 * TENSORS.tsv is `gguf_mix.py list MODEL.gguf`.  Every 2-D tensor whose name
 * matches REGEX (empty or absent: the per-layer dense projections and the
 * output head) is multiplied by a one-token input, or by TOKENS of them (2
 * and 3 are the MTP verify batches), through ds4_gpu_matmul_quant_tensor,
 * the call the Qwen3.8 decode graph makes, all of them per repetition in one
 * command batch, so the weights come from memory rather than the cache.  The
 * per-type GB/s compares kernel choices (DS4_METAL_MIXED_MV_NT_MAX) on one chip. */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <fcntl.h>
#include <regex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

typedef struct { char name[128], type[16]; uint32_t type_id; uint64_t in, out, off, bytes; } tensor;

static const struct { const char *name; uint32_t id; } TYPES[] = {
    {"F32", 0}, {"F16", 1}, {"Q4_0", 2}, {"Q5_0", 6}, {"Q8_0", 8}, {"Q2_K", 10}, {"Q3_K", 11}, {"Q4_K", 12},
    {"Q5_K", 13}, {"Q6_K", 14}, {"IQ2_XXS", 16}, {"IQ2_XS", 17}, {"IQ3_XXS", 18}, {"IQ4_NL", 20},
    {"IQ3_S", 21}, {"IQ2_S", 22}, {"IQ4_XS", 23}, {"BF16", 30}, {"MXFP4", 39}, {"Q2_0", 42},
};

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf TENSORS.tsv [REGEX] [REPS] [TOKENS]\n", argv[0]);
        return 2;
    }
    const char *pattern = argc > 3 && argv[3][0] ? argv[3] :
        "^(blk\\.[0-9]+\\.(attn_qkv|attn_gate|ssm_out|attn_q|attn_output|attn_k|attn_v|hc_attn_down|hc_ffn_down|"
        "ffn_gate_inp|ffn_up_shexp|ffn_gate_shexp|ffn_down_shexp)\\.weight|output\\.weight)$";
    const int reps = argc > 4 ? atoi(argv[4]) : 20;
    const uint32_t n_tok = argc > 5 ? (uint32_t)atoi(argv[5]) : 1u;
    regex_t re;
    if (regcomp(&re, pattern, REG_EXTENDED | REG_NOSUB) != 0) return 2;
    FILE *tf = fopen(argv[2], "r");
    if (!tf) { perror(argv[2]); return 1; }
    tensor *ts = calloc(4096, sizeof(*ts));
    int n = 0;
    uint64_t lo = UINT64_MAX, max_bytes = 0, max_in = 0, max_out = 0;
    char line[512];
    while (fgets(line, sizeof(line), tf) && n < 4096) {
        if (line[0] == '#') continue;
        tensor t = {0};
        char dims[64];
        if (sscanf(line, "%127s %15s %63s %llu %llu", t.name, t.type, dims, &t.off, &t.bytes) != 5) continue;
        unsigned long long d0 = 0, d1 = 0, d2 = 0;
        if (sscanf(dims, "%llu,%llu,%llu", &d0, &d1, &d2) != 2) continue;   /* 2-D only */
        t.in = d0; t.out = d1;
        if (t.off < lo) lo = t.off;
        if (regexec(&re, t.name, 0, NULL, 0) != 0) continue;
        t.type_id = UINT32_MAX;
        for (size_t i = 0; i < sizeof(TYPES) / sizeof(TYPES[0]); i++)
            if (!strcmp(TYPES[i].name, t.type)) t.type_id = TYPES[i].id;
        if (t.type_id == UINT32_MAX || t.type_id == 0) continue;
        if (t.bytes > max_bytes) max_bytes = t.bytes;
        if (t.in > max_in) max_in = t.in;
        if (t.out > max_out) max_out = t.out;
        ts[n++] = t;
    }
    fclose(tf);
    if (n == 0) { fprintf(stderr, "no tensor matches\n"); return 1; }
    const int fd = open(argv[1], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) { perror(argv[1]); return 1; }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    /* the model's whole tensor range, as the engine maps it */
    for (int i = 0; i < n; i++) if (ts[i].bytes > max_bytes) max_bytes = ts[i].bytes;
    if (!ds4_gpu_init() ||
        !ds4_gpu_set_model_map_range(map, (uint64_t)st.st_size, lo, (uint64_t)st.st_size - lo, 1ull << 30)) {
        fprintf(stderr, "cannot map the model\n");
        return 1;
    }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(max_in * n_tok * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(max_out * n_tok * sizeof(float));
    float *host = malloc(max_in * n_tok * sizeof(float));
    for (uint64_t i = 0; i < max_in * n_tok; i++) host[i] = (float)((int)(i % 97) - 48) / 97.0f;
    ds4_gpu_tensor_write(x, 0, host, max_in * n_tok * sizeof(float));

    /* per type: time all its tensors, reps times, in one batch */
    for (size_t ti = 0; ti < sizeof(TYPES) / sizeof(TYPES[0]); ti++) {
        uint64_t bytes = 0;
        int count = 0;
        for (int i = 0; i < n; i++) if (ts[i].type_id == TYPES[ti].id) { bytes += ts[i].bytes; count++; }
        if (!count) continue;
        double best = 1e30;
        for (int trial = 0; trial < 3; trial++) {
            const double t0 = now_s();
            int ok = ds4_gpu_begin_commands();
            for (int r = 0; r < reps && ok; r++)
                for (int i = 0; i < n && ok; i++)
                    if (ts[i].type_id == TYPES[ti].id)
                        ok = ds4_gpu_matmul_quant_tensor(out, map, (uint64_t)st.st_size, ts[i].off, ts[i].type_id,
                                                         ts[i].in, ts[i].out, x, n_tok);
            ok = ok && ds4_gpu_end_commands();
            const double dt = now_s() - t0;
            if (!ok) { fprintf(stderr, "%s matvec failed\n", TYPES[ti].name); return 1; }
            if (dt < best) best = dt;
        }
        printf("%-8s %4d tensors %8.1f MiB  %7.1f us/rep  %6.1f GB/s\n", TYPES[ti].name, count,
               bytes / 1048576.0, best * 1e6 / reps, (double)bytes * reps / best / 1e9);
    }
    ds4_gpu_cleanup();
    return 0;
}
