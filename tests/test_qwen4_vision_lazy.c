/* The Qwen3.8 encoder maps its views only while it encodes: two encodes of one
 * image through an engine that holds the language model give the same bytes,
 * and the dump it writes must equal tests/test_qwen4_vision's, which maps the
 * encoder once up front (make test-qwen4-vision-lazy compares them).  A short
 * greedy text continuation afterwards prints its tokens, which must not change
 * when DS4_QWEN4_VISION_EVICT_LAYERS takes routed experts out of residency
 * for the encodes: they are back for the language model. */
#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.gguf mmproj.gguf image out.bin\n", argv[0]);
        return 2;
    }
    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.vision_path = argv[2];
    opt.backend = DS4_BACKEND_METAL;
    opt.context_size = 4096;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) {
        fprintf(stderr, "cannot open the engine\n");
        return 1;
    }
    ds4_vision_embedding a = {0}, b = {0};
    char err[256] = {0};
    const double t0 = now_ms();
    int ok = ds4_engine_vision_encode_file(e, argv[3], &a, err, sizeof(err));
    const double t1 = now_ms();
    ok = ok && ds4_engine_vision_encode_file(e, argv[3], &b, err, sizeof(err));
    const double t2 = now_ms();
    if (!ok) fprintf(stderr, "encode failed: %s\n", err);
    else printf("encode: first %.1f ms, second %.1f ms\n", t1 - t0, t2 - t1);
    const uint32_t dim = (uint32_t)ds4_engine_embd_dim(e);
    const size_t n = (size_t)a.token_count * dim;
    if (ok && (a.token_count != b.token_count || memcmp(a.data, b.data, n * sizeof(float)) != 0)) {
        fprintf(stderr, "the second encode differs from the first\n");
        ok = 0;
    }
    FILE *fp = ok ? fopen(argv[4], "wb") : NULL;
    if (ok) {
        const uint32_t hdr[4] = { a.token_count, dim, a.grid_height, a.grid_width };
        ok = fp && fwrite(hdr, sizeof(hdr), 1, fp) == 1 && fwrite(a.data, sizeof(float), n, fp) == n;
        if (fp) fclose(fp);
    }
    if (ok) printf("PASS Qwen vision encodes %u tokens the same twice through transient views\n", a.token_count);
    ds4_session *s = NULL;
    ds4_tokens p = {0};
    if (ok && ds4_session_create(&s, e, 4096) == 0) {
        ds4_encode_chat_prompt(e, "You are a careful assistant.",
                               "List the first ten prime numbers and say why 1 is not one of them.",
                               DS4_THINK_NONE, &p);
        ok = ds4_session_sync(s, &p, err, sizeof(err)) == 0;
        printf("tokens:");
        for (int i = 0; ok && i < 16; i++) {
            const int t = ds4_session_argmax(s);
            printf(" %d", t);
            ok = ds4_session_eval(s, t, err, sizeof(err)) == 0;
        }
        printf("\n");
        if (!ok) fprintf(stderr, "text continuation failed: %s\n", err);
        ds4_tokens_free(&p);
        ds4_session_free(s);
    }
    ds4_vision_embedding_free(&a);
    ds4_vision_embedding_free(&b);
    ds4_engine_close(e);
    return ok ? 0 : 1;
}
