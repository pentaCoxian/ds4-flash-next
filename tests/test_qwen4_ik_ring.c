/* Model-backed check of the Qwen3.8 raw indexer key layouts.  On Metal the
 * keys live in a ring of prefill-chunk + 8 rows and disk checkpoints carry
 * only the last rows; DS4_QWEN4_IK_FULL=1 keeps one row per position, the
 * layout of older checkpoints and of CUDA.  A checkpoint written in either
 * layout must load into a session of either, and continuing from it must end
 * where a session that prefilled the same tokens straight does: in a ring
 * session the whole state matches byte for byte; a full session loading a
 * ring checkpoint lacks key rows nothing reads again, so there the logits
 * and the greedy continuation must match.
 *
 *   DS4_TEST_MODEL=/path/to/qwen38.gguf make test-qwen4-ik-ring
 * with DS4_QWEN_NGRAM_GGUF set as for the server. */

#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CTX 8192
#define TEST_CHUNK 64      /* a 72-row ring */
#define TAIL_TOKENS 150    /* past the ring: the rows before the load are gone */

static int failures;

static void check(bool ok, const char *what) {
    printf("%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}

static void die(const char *what, const char *err) {
    fprintf(stderr, "FAIL: %s: %s\n", what, err ? err : "");
    exit(1);
}

/* a session in the full (one row per position) or the ring layout */
static ds4_session *session(ds4_engine *e, bool full) {
    if (full) setenv("DS4_QWEN4_IK_FULL", "1", 1);
    else unsetenv("DS4_QWEN4_IK_FULL");
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, TEST_CTX) != 0) die("session create", NULL);
    unsetenv("DS4_QWEN4_IK_FULL");
    return s;
}

static void sync_len(ds4_session *s, const ds4_tokens *p, int len) {
    char err[256] = "";
    ds4_tokens head = *p;
    head.len = len;
    if (ds4_session_sync(s, &head, err, sizeof(err)) != 0) die("sync", err);
}

static unsigned char *payload(ds4_session *s, size_t *len) {
    char err[256] = "";
    FILE *fp = tmpfile();
    if (!fp || ds4_session_save_payload(s, fp, err, sizeof(err)) != 0) die("save payload", err);
    *len = (size_t)ftell(fp);
    unsigned char *out = malloc(*len);
    rewind(fp);
    if (!out || fread(out, 1, *len, fp) != *len) die("read payload", NULL);
    fclose(fp);
    return out;
}

static void greedy(ds4_session *s, int *out, int n) {
    char err[256] = "";
    for (int i = 0; i < n; i++) {
        out[i] = ds4_session_argmax(s);
        if (ds4_session_eval(s, out[i], err, sizeof(err)) != 0) die("eval", err);
    }
}

static void load(ds4_session *s, const unsigned char *bytes, size_t len) {
    char err[256] = "";
    FILE *fp = tmpfile();
    if (!fp || fwrite(bytes, 1, len, fp) != len) die("write payload", NULL);
    rewind(fp);
    if (ds4_session_load_payload(s, fp, len, err, sizeof(err)) != 0) die("load payload", err);
    fclose(fp);
}

/* write at `cut` in layout `from`, load into layout `to`, prefill to the end */
static void cross(ds4_engine *e, const ds4_tokens *p, int cut, bool from, bool to) {
    ds4_session *w = session(e, from), *r = session(e, to), *ref = session(e, to);
    sync_len(w, p, cut);
    size_t n = 0, want_n = 0, got_n = 0;
    unsigned char *ck = payload(w, &n);
    load(r, ck, n);
    sync_len(r, p, p->len);
    /* the same chunk boundaries as the writer and the loaded session */
    sync_len(ref, p, cut);
    sync_len(ref, p, p->len);
    char what[160];
    snprintf(what, sizeof(what), "%s checkpoint at %d (%zu bytes) continued in a %s cache", from ? "full" : "ring",
             cut, n, to ? "full" : "ring");
    if (!to) {
        unsigned char *want = payload(ref, &want_n), *got = payload(r, &got_n);
        check(want_n == got_n && memcmp(want, got, want_n) == 0, what);
        free(want); free(got);
    } else {
        const int n_vocab = ds4_engine_vocab_size(e);
        float *want = malloc((size_t)n_vocab * sizeof(float)), *got = malloc((size_t)n_vocab * sizeof(float));
        int want_tok[16], got_tok[16];
        ds4_session_copy_logits(ref, want, n_vocab);
        ds4_session_copy_logits(r, got, n_vocab);
        greedy(ref, want_tok, 16);
        greedy(r, got_tok, 16);
        check(memcmp(want, got, (size_t)n_vocab * sizeof(float)) == 0 &&
              memcmp(want_tok, got_tok, sizeof(want_tok)) == 0, what);
        free(want); free(got);
    }
    free(ck);
    ds4_session_free(w); ds4_session_free(r); ds4_session_free(ref);
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "SKIP: set DS4_TEST_MODEL\n");
        return 0;
    }
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = TEST_CTX,
        .prefill_chunk = TEST_CHUNK,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) die("engine open", model);
    char *text = malloc(64 * 1024);
    text[0] = '\0';
    for (int i = 0; i < 120; i++) {
        char line[160];
        snprintf(line, sizeof(line), "Entry %d keeps the value %d for the indexer to find.\n", i, i * 31);
        strcat(text, line);
    }
    ds4_tokens p = {0};
    ds4_encode_chat_prompt(e, "You are a careful assistant.", text, DS4_THINK_HIGH, &p);
    free(text);
    /* a cut off the 4-token block grid, with the block completed after the load */
    const int cut = p.len - TAIL_TOKENS - (p.len - TAIL_TOKENS) % 4 + 2;
    printf("prompt %d tokens, checkpoint at %d\n", p.len, cut);
    for (int from = 0; from < 2; from++)
        for (int to = 0; to < 2; to++) cross(e, &p, cut, from, to);
    ds4_tokens_free(&p);
    ds4_engine_close(e);
    printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
