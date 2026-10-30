/* Residency and bandwidth probe for no-copy views of an mmap'd model file.
 *
 *   metal_residency_probe FILE [views] [view_gib]
 *
 * Wraps the first views x view_gib GiB of FILE in no-copy MTLBuffers, as ds4
 * maps model weights, puts them in a residency set and answers what the
 * on-demand expert eviction for vision depends on:
 *
 *   G2  removing one view from the set unwires its pages (wired memory and the
 *       GPU's "In use system memory" drop by about the view's size);
 *   G3  re-wiring it costs at most ~15 ms/GiB from the page cache, and a
 *       refetch after the pages left the cache reaches >= 4 GB/s with 8 CPU
 *       readers.
 *
 * It also prints the GPU read bandwidth of the mapped views and of an
 * ordinary shared buffer, the number the per-kernel bytes/us figures are read
 * against. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const char *kSource =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void probe_read(device const uint4 *p [[buffer(0)]],\n"
    "                       device atomic_uint *sink [[buffer(1)]],\n"
    "                       constant ulong &n [[buffer(2)]],\n"
    "                       uint gid [[thread_position_in_grid]],\n"
    "                       uint gsz [[threads_per_grid]]) {\n"
    "    uint4 acc = 0;\n"
    "    for (ulong i = gid; i < n; i += gsz) acc ^= p[i];\n"
    "    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u)\n"
    "        atomic_fetch_add_explicit(sink, 1u, memory_order_relaxed);\n"
    "}\n";

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pso;
static id<MTLBuffer> sink;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double gib(uint64_t b) { return b / 1073741824.0; }

static uint64_t wired_bytes(void) {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &n) != KERN_SUCCESS) return 0;
    return (uint64_t)vm.wire_count * (uint64_t)vm_page_size;
}

/* The GPU driver's "In use system memory", from ioreg. */
static uint64_t agx_in_use(void) {
    FILE *p = popen("ioreg -r -c AGXAccelerator -d 1 -w0", "r");
    if (!p) return 0;
    char line[8192];
    uint64_t v = 0;
    const char *key = "\"In use system memory\"=";
    while (fgets(line, sizeof(line), p)) {
        char *s = strstr(line, key);
        if (s) { v = strtoull(s + strlen(key), NULL, 10); break; }
    }
    pclose(p);
    return v;
}

static uint64_t base_wired, base_agx;

static void sample(const char *what) {
    const uint64_t w = wired_bytes(), a = agx_in_use();
    printf("  %-44s wired %+7.2f GiB   gpu in use %+7.2f GiB\n", what,
           gib(w) - gib(base_wired), gib(a) - gib(base_agx));
}

/* GPU time to read every byte of the buffers, in seconds. */
static double gpu_read(id<MTLBuffer> __strong *bufs, int n) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:sink offset:0 atIndex:1];
    for (int i = 0; i < n; i++) {
        const uint64_t words = bufs[i].length / 16;
        [enc setBuffer:bufs[i] offset:0 atIndex:0];
        [enc setBytes:&words length:sizeof(words) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(1u << 18, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "probe: command buffer failed: %s\n", cb.error.localizedDescription.UTF8String);
        return -1;
    }
    return cb.GPUEndTime - cb.GPUStartTime;
}

/* As ds4's queue keepalive: after ~3 s without work on the queue the driver
 * drops the process's GPU mappings, so a tiny kernel runs every 250 ms. */
static dispatch_source_t g_keepalive;

static void keepalive_start(void) {
    dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
                                                     dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
    dispatch_source_set_timer(timer, DISPATCH_TIME_NOW, 250 * NSEC_PER_MSEC, 10 * NSEC_PER_MSEC);
    dispatch_source_set_event_handler(timer, ^{
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const uint64_t none = 0;
        [enc setComputePipelineState:pso];
        [enc setBuffer:sink offset:0 atIndex:0];
        [enc setBuffer:sink offset:0 atIndex:1];
        [enc setBytes:&none length:sizeof(none) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
    });
    dispatch_resume(timer);
    g_keepalive = timer;
}

/* Touch one byte per page of [p, p+len) on 8 threads; returns seconds. */
static double cpu_touch(const uint8_t *p, uint64_t len) {
    const uint64_t page = (uint64_t)getpagesize(), part = len / 8;
    __block volatile uint8_t sum = 0;
    const double t0 = now_s();
    dispatch_apply(8, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t k) {
        uint8_t s = 0;
        for (uint64_t o = k * part; o < (k + 1) * part; o += page) s ^= p[o];
        sum ^= s;
    });
    return now_s() - t0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s FILE [views=8] [view_gib=1]\n", argv[0]);
            return 2;
        }
        const int n_views = argc > 2 ? atoi(argv[2]) : 8;
        const uint64_t view_bytes = (uint64_t)(argc > 3 ? atof(argv[3]) : 1.0) * 1073741824ull;
        const uint64_t span = view_bytes * (uint64_t)n_views;
        if (n_views < 2 || n_views > 64 || view_bytes == 0) {
            fprintf(stderr, "probe: views must be 2..64 and view_gib > 0\n");
            return 2;
        }
        const int fd = open(argv[1], O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0 || (uint64_t)st.st_size < span) {
            fprintf(stderr, "probe: %s must exist and hold %.1f GiB\n", argv[1], gib(span));
            return 2;
        }
        uint8_t *map = mmap(NULL, span, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) { perror("mmap"); return 1; }

        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSource) options:nil error:&err];
        pso = lib ? [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe_read"] error:&err] : nil;
        if (!pso) { fprintf(stderr, "probe: %s\n", err.localizedDescription.UTF8String); return 1; }
        sink = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        keepalive_start();
        printf("%s, %d views of %.2f GiB of %s\n", dev.name.UTF8String, n_views, gib(view_bytes), argv[1]);

        /* Read the span once so the views start from the page cache. */
        double t = cpu_touch(map, span);
        printf("page cache fill: %.2f s, %.2f GB/s\n", t, span / t / 1e9);

        base_wired = wired_bytes();
        base_agx = agx_in_use();
        id<MTLBuffer> views[64];
        for (int i = 0; i < n_views; i++) {
            views[i] = [dev newBufferWithBytesNoCopy:map + (uint64_t)i * view_bytes length:view_bytes
                                             options:MTLResourceStorageModeShared deallocator:nil];
            if (!views[i]) { fprintf(stderr, "probe: no-copy view %d failed\n", i); return 1; }
        }
        sample("no-copy views created");

        MTLResidencySetDescriptor *desc = [MTLResidencySetDescriptor new];
        desc.initialCapacity = n_views;
        id<MTLResidencySet> set = [dev newResidencySetWithDescriptor:desc error:&err];
        if (!set) { fprintf(stderr, "probe: %s\n", err.localizedDescription.UTF8String); return 1; }
        for (int i = 0; i < n_views; i++) [set addAllocation:views[i]];
        [set commit];
        t = now_s();
        [set requestResidency];
        [queue addResidencySet:set];
        printf("requestResidency: %.1f ms\n", (now_s() - t) * 1e3);
        sample("residency requested");

        t = gpu_read(views, n_views);
        printf("first GPU read of the views: %.1f ms, %.1f GB/s\n", t * 1e3, span / t / 1e9);
        sample("after the first GPU read");
        t = gpu_read(views, n_views);
        printf("mapped views, GPU read bandwidth: %.1f GB/s\n", span / t / 1e9);

        @autoreleasepool {
            const uint64_t n = 4ull << 30;
            id<MTLBuffer> one[1] = { [dev newBufferWithLength:n options:MTLResourceStorageModeShared] };
            memset(one[0].contents, 1, n);
            gpu_read(one, 1);
            t = gpu_read(one, 1);
            printf("4 GiB shared buffer, GPU read bandwidth: %.1f GB/s\n", n / t / 1e9);
            [one[0] setPurgeableState:MTLPurgeableStateEmpty];
            one[0] = nil;
        }
        sleep(3);   /* let the driver return the shared buffer before G2 */

        const int last = n_views - 1;
        const uint8_t *last_map = map + (uint64_t)last * view_bytes;
        printf("\nG2: unwire view %d (%.2f GiB)\n", last, gib(view_bytes));
        sample("before");
        [set removeAllocation:views[last]];
        [set commit];
        sample("removed from the set");
        const double t_rm = now_s();
        for (int k = 0; k < 4; k++) {
            usleep(250000);
            char what[64];
            snprintf(what, sizeof(what), "+%.2f s", now_s() - t_rm);
            sample(what);
        }
        gpu_read(views, last);
        sample("after a GPU read of the other views");
        usleep(2000000);
        sample("+2 s more");

        printf("\nG3: re-wire from the page cache\n");
        [set addAllocation:views[last]];
        [set commit];
        t = now_s();
        [set requestResidency];
        const double req = now_s() - t;
        t = gpu_read(&views[last], 1);
        printf("  requestResidency %.1f ms, first GPU read %.1f ms: %.1f ms/GiB\n",
               req * 1e3, t * 1e3, (req + t) * 1e3 / gib(view_bytes));
        sample("re-wired");

        printf("\nG3: refetch after the pages left the cache\n");
        [set removeAllocation:views[last]];
        [set commit];
        msync((void *)last_map, view_bytes, MS_INVALIDATE);
        posix_madvise((void *)last_map, view_bytes, POSIX_MADV_DONTNEED);
        fcntl(fd, F_NOCACHE, 1);
        t = cpu_touch(last_map, view_bytes);
        printf("  8 CPU readers: %.2f s, %.2f GB/s (%s)\n", t, view_bytes / t / 1e9,
               view_bytes / t / 1e9 > 20 ? "still cached: run sudo purge first for a cold number" : "from the SSD");
        [set addAllocation:views[last]];
        [set commit];
        [set requestResidency];
        t = gpu_read(&views[last], 1);
        printf("  GPU read after the refetch: %.1f ms\n", t * 1e3);
        sample("re-wired after the refetch");

        [queue removeResidencySet:set];
        [set endResidency];
        munmap(map, span);
        close(fd);
    }
    return 0;
}
