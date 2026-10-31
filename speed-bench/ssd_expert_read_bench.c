/* SSD cost of streaming a model's routed experts, measured without loading
 * the model: the reads an expert cache issues when a layer's selection misses.
 *
 *   ssd_expert_read_bench MODEL.gguf TENSORS.tsv [TRIALS]
 *
 * TENSORS.tsv is `gguf_mix.py list MODEL.gguf`.  One expert is the gate, up
 * and down slices of one expert id in one layer (blk.N.ffn_*_exps), read with
 * one pread each into a page-aligned buffer, as ds4_gpu_stream_expert_pread_into
 * fills a cache slot.  Rows of the TSV on stdout:
 *
 *   miss          k experts of one random layer, an F_RDADVISE per slice and
 *                 then 3k preads over T threads of a persistent pool, timed
 *                 from the first advise to the last byte: one decode layer's
 *                 misses through the streaming pool's buffered descriptor.
 *   miss_nocache  the same through an F_NOCACHE descriptor.
 *   warm          the same batches read a second time, now from the page
 *                 cache: what a soft second tier in unwired memory costs.
 *   layer         a whole layer's three tensors in CHUNK reads through the
 *                 F_NOCACHE descriptor: the prefill page-in of one layer.
 *
 * Slices that are already in the page cache (mincore) are skipped for every
 * row but warm, so the numbers are the SSD's even on a machine that has run
 * the model.  If few cold experts are left, run `sudo purge` first. */
#define _DARWIN_C_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#define MAX_LAYERS 256
#define MAX_K 10
#define MAX_THREADS 18

typedef struct { uint64_t off, bytes; } tensor_span;
typedef struct { tensor_span t[3]; uint32_t n_expert; bool ok; } layer_info;  /* gate, up, down */
typedef struct { uint64_t off, len; uint8_t *dst; } read_task;

static layer_info g_layers[MAX_LAYERS];
static uint32_t g_n_layers;
static const uint8_t *g_map;
static uint64_t g_page;
static uint32_t g_rng = 0x2545f491u;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static uint32_t rnd(uint32_t n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng % n;
}

static bool read_full(int fd, uint8_t *dst, uint64_t len, uint64_t off) {
    uint64_t pos = 0;
    while (pos < len) {
        const size_t want = len - pos > SSIZE_MAX ? SSIZE_MAX : (size_t)(len - pos);
        const ssize_t n = pread(fd, dst + pos, want, (off_t)(off + pos));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        pos += (uint64_t)n;
    }
    return true;
}

/* A persistent pool like the streaming one: the workers wait for a batch,
 * take its reads in order and the caller waits for the last of them. */
static struct {
    pthread_mutex_t mu;
    pthread_cond_t start, done;
    pthread_t threads[MAX_THREADS];
    uint32_t n_threads, n_tasks, next, finished;
    uint64_t generation;
    const read_task *tasks;
    int fd;
    bool stop, failed;
} P = { .mu = PTHREAD_MUTEX_INITIALIZER, .start = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER };

static void *pool_worker(void *arg) {
    (void)arg;
    uint64_t seen = 0;
    pthread_mutex_lock(&P.mu);
    for (;;) {
        while (P.generation == seen && !P.stop) pthread_cond_wait(&P.start, &P.mu);
        if (P.stop) break;
        seen = P.generation;
        while (P.next < P.n_tasks) {
            const read_task t = P.tasks[P.next++];
            const int fd = P.fd;
            pthread_mutex_unlock(&P.mu);
            const bool ok = read_full(fd, t.dst, t.len, t.off);
            pthread_mutex_lock(&P.mu);
            if (!ok) P.failed = true;
        }
        if (++P.finished == P.n_threads) pthread_cond_signal(&P.done);
    }
    pthread_mutex_unlock(&P.mu);
    return NULL;
}

static void pool_start(uint32_t n) {
    P.n_threads = n;
    P.stop = false;
    for (uint32_t i = 0; i < n; i++) pthread_create(&P.threads[i], NULL, pool_worker, NULL);
}

static void pool_stop(void) {
    pthread_mutex_lock(&P.mu);
    P.stop = true;
    pthread_cond_broadcast(&P.start);
    pthread_mutex_unlock(&P.mu);
    for (uint32_t i = 0; i < P.n_threads; i++) pthread_join(P.threads[i], NULL);
}

static double pool_run(int fd, const read_task *tasks, uint32_t n, bool advise) {
    const double t0 = now_ms();
    if (advise) {
        for (uint32_t i = 0; i < n; i++) {
            struct radvisory ra = { .ra_offset = (off_t)tasks[i].off, .ra_count = (int)tasks[i].len };
            (void)fcntl(fd, F_RDADVISE, &ra);
        }
    }
    pthread_mutex_lock(&P.mu);
    P.fd = fd;
    P.tasks = tasks;
    P.n_tasks = n;
    P.next = 0;
    P.finished = 0;
    P.generation++;
    pthread_cond_broadcast(&P.start);
    while (P.finished < P.n_threads) pthread_cond_wait(&P.done, &P.mu);
    const bool failed = P.failed;
    pthread_mutex_unlock(&P.mu);
    if (failed) {
        fprintf(stderr, "ssd_expert_read_bench: pread failed\n");
        exit(1);
    }
    return now_ms() - t0;
}

/* True when any page of [off, off + len) is in the page cache. */
static bool resident(uint64_t off, uint64_t len) {
    const uint64_t a = off & ~(g_page - 1), e = (off + len + g_page - 1) & ~(g_page - 1);
    const size_t n = (size_t)((e - a) / g_page);
    char stack[256], *vec = n <= sizeof(stack) ? stack : malloc(n);
    bool any = mincore((caddr_t)(g_map + a), (size_t)(e - a), vec) != 0;
    for (size_t i = 0; i < n && !any; i++) any = (vec[i] & MINCORE_INCORE) != 0;
    if (vec != stack) free(vec);
    return any;
}

static uint64_t slice_bytes(const layer_info *l, int t) { return l->t[t].bytes / l->n_expert; }
static uint64_t slice_off(const layer_info *l, int t, uint32_t e) { return l->t[t].off + e * slice_bytes(l, t); }

static bool expert_cold(const layer_info *l, uint32_t e) {
    for (int t = 0; t < 3; t++) {
        if (resident(slice_off(l, t, e), slice_bytes(l, t))) return false;
    }
    return true;
}

/* k distinct cold experts of one random layer as 3k reads; false when the
 * layer has too few cold experts left. */
static bool pick_batch(uint32_t k, read_task *tasks, uint8_t *buf, uint64_t *bytes) {
    const layer_info *l;
    do l = &g_layers[rnd(g_n_layers)]; while (!l->ok);
    uint32_t chosen[MAX_K], n = 0;
    for (uint32_t tries = 0; n < k && tries < 64 * k; tries++) {
        const uint32_t e = rnd(l->n_expert);
        bool dup = false;
        for (uint32_t i = 0; i < n; i++) dup |= chosen[i] == e;
        if (!dup && expert_cold(l, e)) chosen[n++] = e;
    }
    if (n < k) return false;
    uint64_t pos = 0;
    for (uint32_t i = 0; i < k; i++) {
        for (int t = 0; t < 3; t++) {
            const uint64_t len = slice_bytes(l, t);
            tasks[3 * i + t] = (read_task){ slice_off(l, t, chosen[i]), len, buf + pos };
            pos += (len + g_page - 1) & ~(g_page - 1);
        }
    }
    *bytes = pos;
    return true;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* Batches differ in size with their layer's types, so the throughput is the
 * total read over the total time; the MiB column is the mean batch. */
static void report(const char *mode, uint32_t threads, uint32_t k, double *ms, uint32_t n, double mib_total) {
    if (n == 0) {
        printf("%s\t%u\t%u\t0\t-\t-\t-\t-\n", mode, threads, k);
        return;
    }
    double ms_total = 0.0;
    for (uint32_t i = 0; i < n; i++) ms_total += ms[i];
    qsort(ms, n, sizeof(*ms), cmp_double);
    printf("%s\t%u\t%u\t%u\t%.3f\t%.3f\t%.2f\t%.2f\n", mode, threads, k, n, ms[n / 2], ms[(n * 9) / 10],
           mib_total / n, mib_total / 1024.0 / (ms_total / 1e3));
    fflush(stdout);
}

static bool load_tensors(const char *tsv) {
    FILE *f = fopen(tsv, "r");
    if (!f) return false;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        char name[256], type[32], dims[128];
        unsigned long long off, bytes;
        if (sscanf(line, "%255s\t%31s\t%127s\t%llu\t%llu", name, type, dims, &off, &bytes) != 5) continue;
        unsigned layer;
        char kind[8];
        int end = 0;
        if (sscanf(name, "blk.%u.ffn_%7[a-z]_exps.weight%n", &layer, kind, &end) != 2 ||
            name[end] != '\0' || layer >= MAX_LAYERS) continue;
        const int t = !strcmp(kind, "gate") ? 0 : !strcmp(kind, "up") ? 1 : !strcmp(kind, "down") ? 2 : -1;
        const char *last = strrchr(dims, ',');
        if (t < 0 || !last) continue;
        g_layers[layer].t[t] = (tensor_span){ off, bytes };
        g_layers[layer].n_expert = (uint32_t)strtoul(last + 1, NULL, 10);
        if (layer + 1 > g_n_layers) g_n_layers = layer + 1;
    }
    fclose(f);
    uint32_t ok = 0;
    for (uint32_t i = 0; i < g_n_layers; i++) {
        layer_info *l = &g_layers[i];
        l->ok = l->n_expert != 0 && l->t[0].bytes && l->t[1].bytes && l->t[2].bytes &&
                l->t[0].bytes % l->n_expert == 0 && l->t[1].bytes % l->n_expert == 0 &&
                l->t[2].bytes % l->n_expert == 0;
        ok += l->ok;
    }
    return ok != 0;
}

/* Share of the routed expert bytes that is in the page cache. */
static double routed_cached_percent(void) {
    uint64_t routed = 0, cached = 0;
    for (uint32_t i = 0; i < g_n_layers; i++) {
        const layer_info *l = &g_layers[i];
        if (!l->ok) continue;
        for (int t = 0; t < 3; t++) {
            routed += l->t[t].bytes;
            for (uint64_t p = 0; p < l->t[t].bytes; p += 64ull << 20) {
                const uint64_t n = l->t[t].bytes - p < (64ull << 20) ? l->t[t].bytes - p : 64ull << 20;
                const uint64_t a = (l->t[t].off + p) & ~(g_page - 1);
                const uint64_t z = (l->t[t].off + p + n + g_page - 1) & ~(g_page - 1);
                const size_t np = (size_t)((z - a) / g_page);
                char *vec = malloc(np);
                if (vec && mincore((caddr_t)(g_map + a), (size_t)(z - a), vec) == 0) {
                    for (size_t j = 0; j < np; j++) cached += (vec[j] & MINCORE_INCORE) ? g_page : 0;
                }
                free(vec);
            }
        }
    }
    return routed ? 100.0 * cached / routed : 0.0;
}

static void print_machine(const char *model, uint64_t file_bytes) {
    char brand[128] = "?", hw[64] = "?";
    uint64_t mem = 0;
    size_t len = sizeof(brand);
    sysctlbyname("machdep.cpu.brand_string", brand, &len, NULL, 0);
    len = sizeof(hw);
    sysctlbyname("hw.model", hw, &len, NULL, 0);
    len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    printf("# machine\t%s\t%s\t%.0f GiB\n", brand, hw, mem / 1073741824.0);
    printf("# model\t%s\t%.2f GiB\n", model, file_bytes / 1073741824.0);

    uint64_t routed = 0, min_e = UINT64_MAX, max_e = 0;
    for (uint32_t i = 0; i < g_n_layers; i++) {
        const layer_info *l = &g_layers[i];
        if (!l->ok) continue;
        const uint64_t e = slice_bytes(l, 0) + slice_bytes(l, 1) + slice_bytes(l, 2);
        if (e < min_e) min_e = e;
        if (e > max_e) max_e = e;
        routed += l->t[0].bytes + l->t[1].bytes + l->t[2].bytes;
    }
    printf("# routed experts\t%.2f GiB\t%u layers\t%.2f-%.2f MB per expert\t%.1f%% in the page cache\n",
           routed / 1073741824.0, g_n_layers, min_e / 1e6, max_e / 1e6, routed_cached_percent());
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf TENSORS.tsv [TRIALS]\n", argv[0]);
        return 2;
    }
    const uint32_t trials = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 40;
    if (!load_tensors(argv[2])) {
        fprintf(stderr, "%s: no routed expert tensors (blk.N.ffn_*_exps.weight)\n", argv[2]);
        return 1;
    }
    const int fd = open(argv[1], O_RDONLY), fd_nc = open(argv[1], O_RDONLY);
    struct stat st;
    if (fd < 0 || fd_nc < 0 || fstat(fd, &st) != 0 ||
        fcntl(fd_nc, F_NOCACHE, 1) != 0 || fcntl(fd_nc, F_RDAHEAD, 0) != 0) {
        perror(argv[1]);
        return 1;
    }
    /* Mapped only for mincore: nothing reads through the map. */
    g_map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    g_page = (uint64_t)getpagesize();
    if (g_map == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    for (uint32_t i = 0; i < g_n_layers; i++) {
        const layer_info *l = &g_layers[i];
        if (l->ok && l->t[2].off + l->t[2].bytes > (uint64_t)st.st_size) {
            fprintf(stderr, "%s does not match %s\n", argv[2], argv[1]);
            return 1;
        }
    }
    print_machine(argv[1], (uint64_t)st.st_size);
    printf("mode\tthreads\tk\ttrials\tmedian_ms\tp90_ms\tMiB\tGiB_per_s\n");

    const size_t buf_bytes = 256ull << 20;
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, (size_t)g_page, buf_bytes) != 0) return 1;
    memset(buf, 0, buf_bytes);
    double *ms = calloc(trials > 2 ? trials : 2, sizeof(double));  /* the layer rows take 2 */

    static const uint32_t thread_counts[] = { 1, 4, 9, 18 };
    static const uint32_t ks[] = { 1, 2, 3, 5, 10 };
    /* First, before the buffered reads below leave pages of every layer in
     * the page cache: two random cold layers per setting through the
     * uncached descriptor.  The chunks cycle through buf, which nobody reads. */
    static const uint64_t chunks[] = { 1ull << 20, 8ull << 20 };
    for (size_t ci = 0; ci < sizeof(chunks) / sizeof(*chunks); ci++) {
        const uint64_t chunk = chunks[ci];
        for (size_t ti = 1; ti < sizeof(thread_counts) / sizeof(*thread_counts); ti++) {
            const uint32_t threads = thread_counts[ti];
            pool_start(threads);
            uint32_t n = 0;
            double mib = 0.0;
            for (uint32_t tries = 0; n < 2 && tries < 4 * g_n_layers; tries++) {
                const layer_info *l = &g_layers[rnd(g_n_layers)];
                if (!l->ok || resident(l->t[0].off, l->t[0].bytes) ||
                    resident(l->t[1].off, l->t[1].bytes) || resident(l->t[2].off, l->t[2].bytes)) continue;
                uint32_t n_tasks = 0;
                for (int t = 0; t < 3; t++) n_tasks += (uint32_t)((l->t[t].bytes + chunk - 1) / chunk);
                read_task *tasks = malloc(n_tasks * sizeof(*tasks));
                uint32_t j = 0;
                for (int t = 0; t < 3; t++) {
                    for (uint64_t p = 0; p < l->t[t].bytes; p += chunk, j++) {
                        const uint64_t len = l->t[t].bytes - p < chunk ? l->t[t].bytes - p : chunk;
                        tasks[j] = (read_task){ l->t[t].off + p, len, buf + (j % (buf_bytes / chunk)) * chunk };
                    }
                }
                ms[n++] = pool_run(fd_nc, tasks, n_tasks, false);
                mib += (l->t[0].bytes + l->t[1].bytes + l->t[2].bytes) / 1048576.0;
                free(tasks);
            }
            char mode[32];
            snprintf(mode, sizeof(mode), "layer_%lluMiB", (unsigned long long)(chunk >> 20));
            report(mode, threads, 0, ms, n, mib);
            pool_stop();
        }
    }
    for (size_t ti = 0; ti < sizeof(thread_counts) / sizeof(*thread_counts); ti++) {
        const uint32_t threads = thread_counts[ti];
        pool_start(threads);
        for (int nocache = 0; nocache < 2; nocache++) {
            for (size_t ki = 0; ki < sizeof(ks) / sizeof(*ks); ki++) {
                const uint32_t k = ks[ki];
                read_task tasks[3 * MAX_K];
                uint64_t bytes = 0;
                double mib = 0.0;
                uint32_t n = 0;
                for (uint32_t i = 0; i < trials; i++) {
                    if (!pick_batch(k, tasks, buf, &bytes)) continue;
                    ms[n++] = pool_run(nocache ? fd_nc : fd, tasks, 3 * k, !nocache);
                    mib += bytes / 1048576.0;
                }
                report(nocache ? "miss_nocache" : "miss", threads, k, ms, n, mib);
            }
        }
        /* The second read of a batch comes from the page cache. */
        for (size_t ki = 0; ki < sizeof(ks) / sizeof(*ks); ki++) {
            const uint32_t k = ks[ki];
            read_task tasks[3 * MAX_K];
            uint64_t bytes = 0;
            double mib = 0.0;
            uint32_t n = 0;
            for (uint32_t i = 0; i < trials; i++) {
                if (!pick_batch(k, tasks, buf, &bytes)) continue;
                pool_run(fd, tasks, 3 * k, false);
                ms[n++] = pool_run(fd, tasks, 3 * k, false);
                mib += bytes / 1048576.0;
            }
            report("warm", threads, k, ms, n, mib);
        }
        pool_stop();
    }

    printf("# after\t%.1f%% of the routed experts in the page cache\n", routed_cached_percent());
    free(ms);
    free(buf);
    return 0;
}
