/* Qwen3.8 Flash Next. Included by ds4_cuda.cu so model residency, streams
 * and temporary allocations have the same lifetime as the other CUDA paths. */

#include "ds4_qwen4_vision.h"

namespace qwen4_cuda {

__device__ __forceinline__ float sum(float x);
__device__ __forceinline__ float sigmoid(float x);
template<unsigned TYPE>
__device__ __forceinline__ float dot(const char *row, const float *x, unsigned n,
        const uint64_t *grid = NULL, const uint8_t *signs = NULL);
__device__ float injection(const float *inj, unsigned hc, unsigned stream);

struct rope_args { float freq[32], scale; unsigned nrot; };
static float rope_freq[32];
static float rope_scale = 1;
static bool rope_set;

static rope_args rope(unsigned nrot, float base) {
    rope_args r = {};
    r.nrot = nrot;
    r.scale = rope_set ? rope_scale : 1;
    for (unsigned i = 0; i < nrot / 2; i++)
        r.freq[i] = rope_set ? rope_freq[i] : powf(base, -2.0f * i / nrot);
    return r;
}

__device__ void apply_rope(float *row, const uint32_t *pos, rope_args r) {
    const unsigned i = threadIdx.x;
    if (i < r.nrot / 2) {
        const float theta = (float)pos[i % 3] * r.freq[i];
        const float c = cosf(theta) * r.scale, s = sinf(theta) * r.scale;
        const float a = row[i], b = row[i + r.nrot / 2];
        row[i] = a * c - b * s;
        row[i + r.nrot / 2] = a * s + b * c;
    }
    __syncwarp();
}

/* K/V rows of the attention layers: f16, or (Q8) as on Metal the row's
 * int8 values followed by an f16 scale per 32 of them, 17/16 bytes a value */
static bool g_kv_q8;

__host__ __device__ inline uint64_t kv_row_bytes(uint64_t kv_dim, bool q8) {
    return q8 ? kv_dim + kv_dim / 16 : kv_dim * 2;
}

/* the warp's 32 values at elements off..off+31 of a Q8 row, one a lane:
 * the scale amax/127 rounded to f16, the values rounded to nearest
 * (Metal's qwen4_kv_store) */
__device__ __forceinline__ void kv_store_q8(char *row, uint64_t kv_dim, unsigned off, float x) {
    float a = fabsf(x);
    for (unsigned m = 16; m; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, m));
    const __half d = __float2half_rn(__fdiv_rn(a, 127.0f));
    const float id = __half2float(d) > 0 ? __fdiv_rn(1.0f, __half2float(d)) : 0;
    row[off + (threadIdx.x & 31)] = (char)(int)fminf(fmaxf(rintf(__fmul_rn(x, id)), -127.0f), 127.0f);
    if (!(threadIdx.x & 31)) ((__half *)(row + kv_dim))[off / 32] = d;
}

template<bool Q8>
__global__ void attn_prep(float *qout, float *gate, char *kc, char *vc,
        float *iqout, float *ikc, const float *qg, const float *kp, const float *vp,
        const float *iq, const float *ik, const uint32_t *pos3,
        const float *gq, const float *gk, const float *giq,
        unsigned H, unsigned Hkv, unsigned D, unsigned Hi, unsigned Di,
        unsigned pos0, float eps, rope_args rp) {
    const unsigned slot = blockIdx.x, t = blockIdx.y, pos = pos0 + t, lane = threadIdx.x;
    __shared__ float row[256];
    if (slot == H + Hkv + Hi) {
        for (unsigned i = lane; i < Di; i += 32) ikc[(uint64_t)pos * Di + i] = ik[(uint64_t)t * Di + i];
        return;
    }
    const bool isq = slot < H, isk = !isq && slot < H + Hkv;
    const unsigned h = isq ? slot : isk ? slot - H : slot - H - Hkv;
    const unsigned dim = isq || isk ? D : Di;
    const float *src = isq ? qg + ((uint64_t)t * H + h) * 2 * D :
                      isk ? kp + ((uint64_t)t * Hkv + h) * D : iq + ((uint64_t)t * Hi + h) * Di;
    const float *gamma = isq ? gq : isk ? gk : giq;
    const unsigned npt = dim / 32;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) { const float v = src[lane * npt + i]; ss += v * v; }
    const float inv = rsqrtf(sum(ss) / dim + eps);
    for (unsigned i = lane; i < dim; i += 32) row[i] = src[i] * inv * gamma[i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)pos * 4, rp);
    const uint64_t kv_dim = (uint64_t)Hkv * D, rb = kv_row_bytes(kv_dim, Q8);
    for (unsigned i = lane; i < dim; i += 32) {
        if (isq) {
            qout[((uint64_t)t * H + h) * D + i] = row[i];
            gate[((uint64_t)t * H + h) * D + i] = src[D + i];
        } else if (isk && Q8) {
            kv_store_q8(kc + (uint64_t)pos * rb, kv_dim, h * D + i - lane, row[i]);
            kv_store_q8(vc + (uint64_t)pos * rb, kv_dim, h * D + i - lane, vp[((uint64_t)t * Hkv + h) * D + i]);
        } else if (isk) {
            ((__half *)kc)[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(row[i]);
            ((__half *)vc)[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(vp[((uint64_t)t * Hkv + h) * D + i]);
        } else iqout[((uint64_t)t * Hi + h) * Di + i] = row[i];
    }
}

__global__ void block_key(__half *out, const float *ik, const uint32_t *pos3,
        const float *gamma, unsigned block0, unsigned ratio, unsigned D, float eps, rope_args rp) {
    const unsigned b = block0 + blockIdx.x, lane = threadIdx.x, npt = D / 32;
    __shared__ float row[128];
    float ss = 0, v[4];
    for (unsigned i = 0; i < npt; i++) {
        float a = 0;
        for (unsigned t = 0; t < ratio; t++) a += ik[((uint64_t)b * ratio + t) * D + lane * npt + i];
        v[i] = a / ratio;
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(sum(ss) / D + eps);
    for (unsigned i = 0; i < npt; i++) row[lane * npt + i] = v[i] * inv * gamma[lane * npt + i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)b * ratio * 4, rp);
    for (unsigned i = lane; i < D; i += 32) out[(uint64_t)b * D + i] = __float2half_rn(row[i]);
}

__global__ void idx_score(float *out, const float *q, const __half *key,
        unsigned N, unsigned H, unsigned D, unsigned pos0, unsigned ratio) {
    const unsigned b = blockIdx.x * 4 + threadIdx.x / 32, t = blockIdx.y, lane = threadIdx.x & 31;
    if (b >= N) return;
    float score = 0;
    if (b >= (pos0 + t + 1) / ratio) score = -3e38f;
    else for (unsigned h = 0; h < H; h++) {
        float a = 0;
        for (unsigned i = lane; i < D; i += 32)
            a += q[((uint64_t)t * H + h) * D + i] * __half2float(key[(uint64_t)b * D + i]);
        score += fmaxf(sum(a), 0);
    }
    if (!lane) out[(uint64_t)t * N + b] = score;
}

__global__ void tile_max(unsigned *out, const float *scores, unsigned N, unsigned tiles) {
    const unsigned tile = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (tile >= tiles) return;
    unsigned v = 0;
    for (unsigned i = tile * 8; i < min(N, tile * 8 + 8); i++)
        v = max(v, __float_as_uint(fmaxf(scores[(uint64_t)t * N + i], 0)));
    out[(uint64_t)t * tiles + tile] = v;
}

/* Indexer top-k: the K largest of a row's N block scores (as unsigned keys
 * of the scores clamped at 0), the blocks above the K-th largest score in
 * block order, then as many of those equal to it as K takes, in block order.
 * The K-th largest comes from four 8-bit radix passes; their histograms add
 * a warp's equal bins in one atomic (scores of one magnitude share their top
 * byte), and warp 0 picks the bin.  With N at least 8K the passes
 * run on the maxima of the eight-score tiles first: the K-th largest score
 * is at least the K-th largest tile maximum (K tiles each hold a score that
 * high), so the second search only needs the scores from that bound up, at
 * most cap of them and usually little over K (more ties than cap search the
 * whole row).  The passes over the row keep eight reads a thread in flight.
 * One block a row. */
/* warp 0: the bin, from the top, that holds the need-th key */
__device__ __forceinline__ void radix_pick(const unsigned *hist, unsigned shift, unsigned *threshold, unsigned *need) {
    if (threadIdx.x >= 32) return;
    const unsigned lane = threadIdx.x, nd = *need;
    unsigned c[8], own = 0;
    #pragma unroll
    for (unsigned k = 0; k < 8; k++) { c[k] = hist[255 - 8 * lane - k]; own += c[k]; }
    unsigned upto = own;
    for (unsigned o = 1; o < 32; o <<= 1) {
        const unsigned v = __shfl_up_sync(0xffffffffu, upto, o);
        if (lane >= o) upto += v;
    }
    if (upto - own < nd && upto >= nd) {
        unsigned r = nd - (upto - own);
        for (unsigned k = 0; k < 8; k++) {
            if (c[k] >= r) { *threshold |= (255 - 8 * lane - k) << shift; *need = r; break; }
            r -= c[k];
        }
    }
}

/* the K-th largest of n keys key(i): *threshold its value, *need how many
 * keys equal to it the K largest take */
template<typename KEY>
__device__ void radix_kth(KEY key, unsigned n, unsigned K, unsigned *hist, unsigned *threshold, unsigned *need) {
    const unsigned tid = threadIdx.x, lane = tid & 31;
    if (!tid) { *threshold = 0; *need = K; }
    for (unsigned pass = 0; pass < 4; pass++) {
        const unsigned shift = 24 - 8 * pass, mask = pass ? (0xffffffffu << (shift + 8)) : 0;
        for (unsigned i = tid; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const unsigned th = *threshold;
        for (unsigned i0 = 0; i0 < n; i0 += blockDim.x) {
            const unsigned i = i0 + tid, k = i < n ? key(i) : 0;
            const bool act = i < n && (k & mask) == th;
            const unsigned b = act ? (k >> shift) & 255 : 256u, peers = __match_any_sync(0xffffffffu, b);
            if (act && __ffs(peers) - 1 == lane) atomicAdd(hist + b, __popc(peers));
        }
        __syncthreads();
        radix_pick(hist, shift, threshold, need);
        __syncthreads();
    }
}

/* cap: room for the scores past the tile bound, 0 for no tile search */
__global__ void __launch_bounds__(1024) idx_select(int *out, const float *score, unsigned N, unsigned K, unsigned cap) {
    extern __shared__ unsigned smem[];
    const unsigned tid = threadIdx.x, nth = blockDim.x, lane = tid & 31, warp = tid / 32, nw = nth / 32;
    const unsigned t = blockIdx.x, NT = (N + 7) / 8, lt = (1u << lane) - 1;
    unsigned *tiles = smem, *cand = smem + NT;
    __shared__ unsigned hist[256], bound, threshold, need, count;
    __shared__ uint64_t scan[32];
    const float *row = score + (uint64_t)t * N;
    auto key = [row](unsigned i) { return __float_as_uint(fmaxf(row[i], 0)); };
    bool full = !cap;
    if (cap) {
        /* a warp takes 256 scores in eight coalesced reads; eight lanes share a tile */
        for (unsigned base = warp * 256; base < N; base += nw * 256) {
            unsigned k[8];
            #pragma unroll
            for (unsigned j = 0; j < 8; j++) k[j] = base + lane + 32 * j < N ? key(base + lane + 32 * j) : 0;
            #pragma unroll
            for (unsigned j = 0; j < 8; j++) {
                unsigned v = k[j];
                v = max(v, __shfl_xor_sync(0xffffffffu, v, 1));
                v = max(v, __shfl_xor_sync(0xffffffffu, v, 2));
                v = max(v, __shfl_xor_sync(0xffffffffu, v, 4));
                const unsigned tile = base / 8 + 4 * j + lane / 8;
                if (!(lane & 7) && tile < NT) tiles[tile] = v;
            }
        }
        __syncthreads();
        radix_kth([tiles](unsigned i) { return tiles[i]; }, NT, K, hist, &bound, &need);
        if (!tid) count = 0;
        __syncthreads();
        const unsigned bd = bound;
        for (unsigned i0 = 0; i0 < N; i0 += 8 * nth) {
            unsigned k[8];
            #pragma unroll
            for (unsigned u = 0; u < 8; u++) k[u] = i0 + u * nth + tid < N ? key(i0 + u * nth + tid) : 0;
            #pragma unroll
            for (unsigned u = 0; u < 8; u++) {
                const bool act = i0 + u * nth + tid < N && k[u] >= bd;
                const unsigned ballot = __ballot_sync(0xffffffffu, act);
                if (!ballot) continue;
                unsigned at = 0;
                if (!lane) at = atomicAdd(&count, __popc(ballot));
                at = __shfl_sync(0xffffffffu, at, 0) + __popc(ballot & lt);
                if (act && at < cap) cand[at] = k[u];
            }
        }
        __syncthreads();
        full = count > cap;
        if (!full) radix_kth([cand](unsigned i) { return cand[i]; }, count, K, hist, &threshold, &need);
    }
    if (full) radix_kth(key, N, K, hist, &threshold, &need);
    /* the gather: a warp a contiguous range, 32 scores a step and eight
     * steps in flight, places from ballots and the warps before it */
    const unsigned th = threshold, nd = need;
    const unsigned span = ((N + nw - 1) / nw + 255) / 256 * 256, wb = min(N, warp * span), we = min(N, wb + span);
    unsigned ng = 0, ne = 0;
    for (unsigned i0 = wb; i0 < we; i0 += 256) {
        unsigned k[8];
        #pragma unroll
        for (unsigned u = 0; u < 8; u++) k[u] = i0 + 32 * u + lane < we ? key(i0 + 32 * u + lane) : 0;
        #pragma unroll
        for (unsigned u = 0; u < 8; u++) {
            const bool in = i0 + 32 * u + lane < we;
            ng += __popc(__ballot_sync(0xffffffffu, in && k[u] > th));
            ne += __popc(__ballot_sync(0xffffffffu, in && k[u] == th));
        }
    }
    if (!lane) scan[warp] = (uint64_t)ng << 32 | ne;
    __syncthreads();
    uint64_t before = 0;
    for (unsigned w = 0; w < warp; w++) before += scan[w];
    unsigned g = (unsigned)(before >> 32), e = (unsigned)before;
    int *o = out + (uint64_t)t * K;
    for (unsigned i0 = wb; i0 < we; i0 += 256) {
        unsigned k[8];
        #pragma unroll
        for (unsigned u = 0; u < 8; u++) k[u] = i0 + 32 * u + lane < we ? key(i0 + 32 * u + lane) : 0;
        #pragma unroll
        for (unsigned u = 0; u < 8; u++) {
            const unsigned i = i0 + 32 * u + lane;
            const unsigned gb = __ballot_sync(0xffffffffu, i < we && k[u] > th);
            const unsigned eb = __ballot_sync(0xffffffffu, i < we && k[u] == th);
            if (i < we && k[u] > th) o[g + __popc(gb & lt)] = i;
            else if (i < we && k[u] == th && e + __popc(eb & lt) < nd) o[K - nd + e + __popc(eb & lt)] = i;
            g += __popc(gb);
            e += __popc(eb);
        }
    }
}

__global__ void idx_expand(int *out, unsigned *count, const int *blocks,
        unsigned K, unsigned ratio, unsigned pos0, unsigned stride) {
    const unsigned t = blockIdx.x, pos = pos0 + t, tail = (pos + 1) / ratio * ratio;
    for (unsigned i = threadIdx.x; i < K * ratio; i += blockDim.x)
        out[(uint64_t)t * stride + i] = blocks[(uint64_t)t * K + i / ratio] * ratio + i % ratio;
    for (unsigned i = tail + threadIdx.x; i <= pos; i += blockDim.x)
        out[(uint64_t)t * stride + K * ratio + i - tail] = i;
    if (!threadIdx.x) count[t] = K * ratio + pos + 1 - tail;
}

__device__ __forceinline__ float sbyte_f(uint64_t v, unsigned k);

/* N int8 values of a Q8 K/V row as floats (N of 1, 4 or 8, aligned) */
template<unsigned N>
__device__ __forceinline__ void kv_load_q8(const char *p, float *out) {
    uint64_t v = 0;
    if (N == 8) { const int2 w = *(const int2 *)p; v = (uint32_t)w.x | (uint64_t)(uint32_t)w.y << 32; }
    else if (N == 4) v = *(const uint32_t *)p;
    else v = (uint8_t)*p;
    #pragma unroll
    for (unsigned k = 0; k < N; k++) out[k] = sbyte_f(v, k);
}

/* A warp per head and token.  With f16 rows a lane takes elements lane +
 * 32i; with Q8 rows elements lane*D/32 + i, so its values are consecutive
 * bytes under one scale.  A row's keys split by its own count (up to 64
 * splits of at least 32 keys, the rest of the grid's splits empty), so a
 * row of a verify batch sums its keys as the lone row does. */
template<unsigned D, bool Q8>
__global__ void attention(float *out, float *partial, const float *q, const float *gate,
        const char *kc, const char *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse,
        unsigned splits, unsigned per, float scale) {
    constexpr unsigned NPT = D / 32;
    const unsigned h = blockIdx.x * 4 + threadIdx.x / 32, t = blockIdx.y, split = blockIdx.z;
    if (h >= H) return;
    const unsigned lane = threadIdx.x & 31, kh = h / (H / Hkv), n = sparse ? counts[t] : pos0 + t + 1;
    const unsigned ns = min(splits, (n + 31) / 32), per_row = ns ? (n + ns - 1) / ns : 0;
    (void)per;
    const uint64_t kv_dim = (uint64_t)Hkv * D, rb = kv_row_bytes(kv_dim, Q8);
    const unsigned e0 = Q8 ? lane * NPT : lane, es = Q8 ? 1 : 32;
    float qv[NPT], acc[NPT] = {}, m = -3e38f, denom = 0;
    for (unsigned i = 0; i < NPT; i++) qv[i] = q[((uint64_t)t * H + h) * D + e0 + es * i] * scale;
    for (unsigned j = split * per_row; j < min(n, (split + 1) * per_row); j++) {
        const unsigned p = sparse ? (unsigned)sel[(uint64_t)t * stride + j] : j;
        if (p > pos0 + t) continue;
        float score = 0, x[NPT];
        if (Q8) {
            const char *row = kc + (uint64_t)p * rb;
            kv_load_q8<NPT>(row + kh * D + e0, x);
            for (unsigned i = 0; i < NPT; i++) score += qv[i] * x[i];
            score *= __half2float(((const __half *)(row + kv_dim))[(kh * D + e0) / 32]);
        } else {
            const __half *row = (const __half *)kc + ((uint64_t)p * Hkv + kh) * D;
            for (unsigned i = 0; i < NPT; i++) score += qv[i] * __half2float(row[e0 + es * i]);
        }
        score = sum(score);
        const float nm = fmaxf(m, score), correction = expf(m - nm), w = expf(score - nm);
        denom = denom * correction + w;
        if (Q8) {
            const char *row = vc + (uint64_t)p * rb;
            kv_load_q8<NPT>(row + kh * D + e0, x);
            const float ws = w * __half2float(((const __half *)(row + kv_dim))[(kh * D + e0) / 32]);
            for (unsigned i = 0; i < NPT; i++) acc[i] = acc[i] * correction + ws * x[i];
        } else {
            const __half *row = (const __half *)vc + ((uint64_t)p * Hkv + kh) * D;
            for (unsigned i = 0; i < NPT; i++) acc[i] = acc[i] * correction + w * __half2float(row[e0 + es * i]);
        }
        m = nm;
    }
    if (splits == 1) {
        for (unsigned i = 0; i < NPT; i++) {
            const uint64_t p = ((uint64_t)t * H + h) * D + e0 + es * i;
            out[p] = (denom > 0 ? acc[i] / denom : 0) * sigmoid(gate[p]);
        }
    } else {
        float *dst = partial + (((uint64_t)t * H + h) * splits + split) * (D + 2);
        if (!lane) { dst[0] = m; dst[1] = denom; }
        for (unsigned i = 0; i < NPT; i++) dst[2 + e0 + es * i] = acc[i];
    }
}

__global__ void attn_merge(float *out, const float *partial, const float *gate,
        unsigned H, unsigned D, unsigned splits) {
    const unsigned h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    if (d >= D) return;
    const float *p = partial + ((uint64_t)t * H + h) * splits * (D + 2);
    float m = -3e38f, denom = 0, acc = 0;
    for (unsigned s = 0; s < splits; s++) m = fmaxf(m, p[s * (D + 2)]);
    for (unsigned s = 0; s < splits; s++) {
        const float *row = p + s * (D + 2);
        const float w = row[1] > 0 ? expf(row[0] - m) : 0;
        denom += row[1] * w; acc += row[2 + d] * w;
    }
    const uint64_t i = ((uint64_t)t * H + h) * D + d;
    out[i] = (denom > 0 ? acc / denom : 0) * sigmoid(gate[i]);
}

/* The heads in a KV group share the same selected keys. Keep their output
 * accumulators in registers and use tensor cores for both products. Scaled
 * residual components retain the fine part of the FP32 queries/probabilities;
 * K and V are already half, so neither requires further rounding. */
__global__ void attention_group(float *out, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse, float scale) {
#if __CUDA_ARCH__ >= 800
    const unsigned D = 256, tid = threadIdx.x, lane = tid&31, warp = tid/32;
    const unsigned t = blockIdx.y, kh = blockIdx.x, group = H/Hkv;
    const unsigned qr = tid/16, col = tid%16, h = kh*group+qr;
    const unsigned n = sparse ? counts[t] : pos0+t+1;
    __shared__ __align__(32) __half qh[16][264], ql[16][264], kv[32][264];
    union Scores { float part[2][16][32]; __half prob[2][16][40]; };
    __shared__ __align__(32) Scores scores;
    __shared__ float qs[16], max_score[16], denom[16], correction[16];
    __shared__ unsigned positions[32];
    float mx = 0;
    for (unsigned d = col; d < D; d += 16)
        if (qr < group) mx = fmaxf(mx,fabsf(q[((uint64_t)t*H+h)*D+d]*scale));
    for (unsigned off = 8; off; off /= 2) mx = fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,off,16));
    const int exp = mx > 0 ? max(-120,min(120,(int)((__float_as_uint(mx)>>23)&255)-127)) : 0;
    const float inv = ldexpf(1,-exp);
    if (!col) { qs[qr] = ldexpf(1,exp); max_score[qr] = -3e38f; denom[qr] = 0; }
    for (unsigned d = col; d < D; d += 16) {
        const float v = qr < group ? q[((uint64_t)t*H+h)*D+d]*scale*inv : 0;
        qh[qr][d] = __float2half_rn(v);
        ql[qr][d] = __float2half_rn((v-__half2float(qh[qr][d]))*4096);
    }
    float result[4][4] = {};
    __syncthreads();
    for (unsigned j0 = 0; j0 < n; j0 += 32) {
        if (tid < 32) {
            const unsigned j = j0+tid;
            const unsigned p = j < n ? (sparse ? (unsigned)sel[(uint64_t)t*stride+j] : j) : UINT_MAX;
            positions[tid] = p <= pos0+t ? p : UINT_MAX;
        }
        __syncthreads();
        for (unsigned i = tid*8; i < 32*D; i += 256*8) {
            const unsigned r = i/D, d = i%D, p = positions[r];
            tt_cp_async_16B(&kv[r][d],kc+((uint64_t)(p == UINT_MAX ? 0 : p)*Hkv+kh)*D+d,p != UINT_MAX);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        float hi[4] = {}, lo[4] = {};
        const unsigned split = warp/4, key0 = (warp%4)*8;
        for (unsigned k = split*128; k < (split+1)*128; k += 16) {
            uint32_t ah[4], al[4], b[2];
            tt_ldmatrix_x4(ah,&qh[lane%16][k+(lane/16)*8]);
            tt_ldmatrix_x4(al,&ql[lane%16][k+(lane/16)*8]);
            tt_ldmatrix_x2(b,&kv[key0+lane%8][k+((lane%16)/8)*8]);
            tt_mma_m16n8k16_f16_f32(hi,ah,b);
            tt_mma_m16n8k16_f16_f32(lo,al,b);
        }
        #pragma unroll
        for (unsigned i = 0; i < 4; i++)
            scores.part[split][tt_mma_c_i(lane,i)][key0+tt_mma_c_j(lane,i)] = hi[i]+lo[i]*0x1p-12f;
        __syncthreads();
        float prob[2], peak = max_score[qr];
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const unsigned key = col+i*16;
            prob[i] = positions[key] != UINT_MAX ?
                (scores.part[0][qr][key]+scores.part[1][qr][key])*qs[qr] : -3e38f;
            peak = fmaxf(peak,prob[i]);
        }
        for (unsigned off = 8; off; off /= 2) peak = fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,off,16));
        const float old = expf(max_score[qr]-peak);
        float total = 0;
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            prob[i] = positions[col+i*16] != UINT_MAX ? expf(prob[i]-peak) : 0;
            total += prob[i];
        }
        for (unsigned off = 8; off; off /= 2) total += __shfl_xor_sync(0xffffffff,total,off,16);
        __syncthreads();
        if (!col) {
            correction[qr] = old;
            max_score[qr] = peak;
            denom[qr] = denom[qr]*old+total;
        }
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const __half p = __float2half_rn(prob[i]);
            scores.prob[0][qr][col+i*16] = p;
            scores.prob[1][qr][col+i*16] = __float2half_rn((prob[i]-__half2float(p))*4096);
        }
        for (unsigned i = tid*8; i < 32*D; i += 256*8) {
            const unsigned r = i/D, d = i%D, p = positions[r];
            tt_cp_async_16B(&kv[r][d],vc+((uint64_t)(p == UINT_MAX ? 0 : p)*Hkv+kh)*D+d,p != UINT_MAX);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        #pragma unroll
        for (unsigned tile = 0; tile < 4; tile++) {
            float hi[4] = {}, lo[4] = {};
            #pragma unroll
            for (unsigned k = 0; k < 32; k += 16) {
                uint32_t ah[4], al[4], b[2];
                tt_ldmatrix_x4(ah,&scores.prob[0][lane%16][k+(lane/16)*8]);
                tt_ldmatrix_x4(al,&scores.prob[1][lane%16][k+(lane/16)*8]);
                tt_ldmatrix_x2_trans(b,&kv[k+lane%16][warp*32+tile*8]);
                tt_mma_m16n8k16_f16_f32(hi,ah,b);
                tt_mma_m16n8k16_f16_f32(lo,al,b);
            }
            #pragma unroll
            for (unsigned i = 0; i < 4; i++) result[tile][i] =
                result[tile][i]*correction[tt_mma_c_i(lane,i)]+(hi[i]+lo[i]*0x1p-12f);
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned tile = 0; tile < 4; tile++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = tt_mma_c_i(lane,i), d = warp*32+tile*8+tt_mma_c_j(lane,i);
            if (r < group) {
                const uint64_t dst = ((uint64_t)t*H+kh*group+r)*D+d;
                out[dst] = (denom[r] > 0 ? result[tile][i]/denom[r] : 0)*sigmoid(gate[dst]);
            }
        }
    }
#endif
}

static bool tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

__host__ __device__ static uint64_t row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
    case 0: return n * 4;
    case 1: case 30: return n * 2;
    case 2: case 20: return n % 32 ? 0 : n / 32 * 18;
    case 6: return n % 32 ? 0 : n / 32 * 22;
    case 8: return n % 32 ? 0 : n / 32 * 34;
    case 39: return n % 32 ? 0 : n / 32 * 17;
    case 42: return n % 64 ? 0 : n / 64 * 18;
    case 10: return n % 256 ? 0 : n / 256 * 84;
    case 11: case 21: return n % 256 ? 0 : n / 256 * 110;
    case 12: return n % 256 ? 0 : n / 256 * 144;
    case 13: return n % 256 ? 0 : n / 256 * 176;
    case 14: return n % 256 ? 0 : n / 256 * 210;
    case 16: return n % 256 ? 0 : n / 256 * 66;
    case 17: return n % 256 ? 0 : n / 256 * 74;
    case 18: return n % 256 ? 0 : n / 256 * 98;
    case 22: return n % 256 ? 0 : n / 256 * 82;
    case 23: return n % 256 ? 0 : n / 256 * 136;
    default: return 0;
    }
}

/* The weight types the row kernels instantiate: those of the upstream Qwen
 * files (F32, F16, Q4_0, Q8_0, Q2_K, Q4_K, IQ2_XXS, BF16, MXFP4) and the
 * mixed ones of ISTA's GSQ-RCO files (Q5_0, Q3_K, Q5_K, Q6_K, IQ2_XS,
 * IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, Q2_0). */
#define QWEN_EACH_TYPE(X) X(0); X(1); X(2); X(8); X(10); X(12); X(16); X(30); X(39); \
    X(6); X(11); X(13); X(14); X(17); X(18); X(20); X(21); X(22); X(23); X(42)

static const char *weight(const void *map, uint64_t size, uint64_t off, uint64_t bytes) {
    if (!map || !bytes || off > size || bytes > size - off) return NULL;
    return cuda_resolve_weight_ptr(map, off, bytes, 0, "Qwen weights");
}

static int launched(void) { return cuda_ok(cudaGetLastError(), "Qwen kernel"); }

/* Scratch on the GPU the current layer runs on: with layer placement over
 * several GPUs (ds4.c qwen4_graph_use_tier) every tier has its own. */
static void *tmp_alloc(uint64_t bytes, const char *what) {
    int dev = 0, tier = 0;
    if (g_n_gpus > 1 && cudaGetDevice(&dev) == cudaSuccess)
        for (int t = 0; t < g_n_gpus; t++) if (g_gpu[t].device_id == dev) tier = t;
    return cuda_tmp_alloc_on(tier, bytes, what);
}

__device__ __forceinline__ float sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}

__device__ __forceinline__ float block_sum(float x, float *shared) {
    x = sum(x);
    if (!(threadIdx.x & 31)) shared[threadIdx.x / 32] = x;
    __syncthreads();
    float result = 0;
    for (unsigned i = 0; i < blockDim.x / 32; i++) result += shared[i];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }
__device__ __forceinline__ float softplus(float x) {
    return x > 20 ? x : x < -20 ? expf(x) : log1pf(expf(x));
}

/* ISTA's codebooks in global memory rather than constant memory: the lanes of
 * a row dot each read their own entry, which constant memory serializes. */
#define DS4_IQ_GRID static const __device__
#include "ds4_iq_grids.inc"
#undef DS4_IQ_GRID
static const __device__ int8_t iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

__device__ __forceinline__ float half_at(const uint8_t *p) { return __half2float(*(const __half *)p); }

/* K-quant 6-bit scale and min of 32-weight group g (Q4_K, Q5_K). */
__device__ __forceinline__ void scale_min_k4(const uint8_t *s, unsigned g, unsigned &sc, unsigned &mn) {
    if (g < 4) { sc = s[g] & 63; mn = s[g + 4] & 63; }
    else {
        sc = (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
        mn = (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
    }
}

/* Element j of one 256-weight block of the mixed types, following
 * ds4_quants.c's block layouts (the CPU reference of the Metal kernels). */
template<unsigned TYPE>
__device__ __forceinline__ float value_mixed(const uint8_t *b, unsigned j) {
    const unsigned ib = j / 32, l = j % 32 / 8, k = j % 8;
    if (TYPE == 11) {          /* Q3_K: hmask[32], qs[64], scales[12], d */
        const unsigned h = j / 128, g = j % 128 / 32, e = j % 32, s = 8 * h + 2 * g + e / 16;
        const int sc = (int)(((b[96 + s % 8] >> (4 * (s / 8))) & 15) | (((b[104 + s % 4] >> (2 * (s / 4))) & 3) << 4)) - 32;
        const int q = (int)((b[32 + 32 * h + e] >> (2 * g)) & 3) - ((b[e] >> (4 * h + g)) & 1 ? 0 : 4);
        return half_at(b + 108) * (float)sc * (float)q;
    }
    if (TYPE == 13) {          /* Q5_K: d, dmin, scales[12], qh[32], qs[128] */
        unsigned sc, mn;
        scale_min_k4(b + 4, ib, sc, mn);
        const unsigned q = ((b[48 + 32 * (j / 64) + j % 32] >> (4 * (ib & 1))) & 15) |
                           (((b[16 + j % 32] >> ib) & 1) << 4);
        return half_at(b) * sc * q - half_at(b + 2) * mn;
    }
    if (TYPE == 14) {          /* Q6_K: ql[128], qh[64], int8 scales[16], d */
        const unsigned h = j / 128, g = j % 128 / 32, e = j % 32;
        const int q = (int)(((b[64 * h + e + 32 * (g & 1)] >> (4 * (g >> 1))) & 15) |
                            (((b[128 + 32 * h + e] >> (2 * g)) & 3) << 4)) - 32;
        return half_at(b + 208) * (float)(int8_t)b[192 + 8 * h + e / 16 + 2 * g] * (float)q;
    }
    if (TYPE == 23) {          /* IQ4_XS: d, scales_h, scales_l[4], qs[128] */
        const unsigned hi = (unsigned)b[2] | ((unsigned)b[3] << 8);
        const int ls = (int)(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((hi >> (2 * ib)) & 3) << 4)) - 32;
        return half_at(b) * (float)ls * (float)iq4nl_values[(b[8 + 16 * ib + j % 16] >> (4 * (j % 32 / 16))) & 15];
    }
    unsigned grid, signs;
    float dl;
    if (TYPE == 17) {          /* IQ2_XS: d, u16 qs[32] (9-bit grid, 7-bit signs), scales[8] */
        const unsigned q = (unsigned)b[2 + 8 * ib + 2 * l] | ((unsigned)b[3 + 8 * ib + 2 * l] << 8);
        dl = half_at(b) * (.5f + ((b[66 + ib] >> (4 * (l / 2))) & 15)) * .25f;
        grid = (unsigned)(iq2xs_grid[q & 511] >> (8 * k)) & 255;
        signs = ksigns_iq2xs[q >> 9];
    } else if (TYPE == 22) {   /* IQ2_S: d, qs[32], signs[32], qh[8], scales[8] */
        dl = half_at(b) * (.5f + ((b[74 + ib] >> (4 * (l / 2))) & 15)) * .25f;
        grid = (unsigned)(iq2s_grid[b[2 + 4 * ib + l] | (((b[66 + ib] >> (2 * l)) & 3) << 8)] >> (8 * k)) & 255;
        signs = b[34 + 4 * ib + l];
    } else if (TYPE == 18) {   /* IQ3_XXS: d, qs[64], per 32 weights 4 x 7 sign bits and a scale */
        const uint8_t *a = b + 66 + 4 * ib;
        const unsigned aux = (unsigned)a[0] | ((unsigned)a[1] << 8) | ((unsigned)a[2] << 16) | ((unsigned)a[3] << 24);
        dl = half_at(b) * (.5f + (aux >> 28)) * .5f;
        grid = (iq3xxs_grid[b[2 + 8 * ib + 2 * l + k / 4]] >> (8 * (k % 4))) & 255;
        signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
    } else {                   /* IQ3_S: d, qs[64], qh[8], signs[32], scales[4] */
        dl = half_at(b) * (float)(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15));
        const unsigned idx = b[2 + 8 * ib + 2 * l + k / 4] | (((b[66 + ib] >> (2 * l + k / 4)) & 1) << 8);
        grid = (iq3s_grid[idx] >> (8 * (k % 4))) & 255;
        signs = b[74 + 4 * ib + l];
    }
    return dl * (float)grid * ((signs >> k) & 1 ? -1.f : 1.f);
}

/* Eight bytes from a 2-byte aligned address, in two or three aligned words:
 * the blocks of most formats are only 2-byte aligned (Q6_K's are 210 bytes).
 * The third word may reach two bytes past the eight, inside the allocation. */
__device__ __forceinline__ uint64_t bytes8(const uint8_t *p) {
    const uintptr_t a = (uintptr_t)p;
    const uint32_t *w = (const uint32_t *)(a & ~(uintptr_t)3);
    const uint32_t w0 = __ldg(w), w1 = __ldg(w + 1);
    if (!(a & 3)) return w0 | (uint64_t)w1 << 32;
    const uint32_t w2 = __ldg(w + 2);
    return __funnelshift_r(w0, w1, 16) | (uint64_t)__funnelshift_r(w1, w2, 16) << 32;
}

__device__ __forceinline__ unsigned u16_at(const uint8_t *p) { return __ldg((const uint16_t *)p); }

/* byte k of v */
__device__ __forceinline__ unsigned byte_k(uint64_t v, unsigned k) { return (unsigned)(v >> (8 * k)) & 255; }

/* byte k of v as a float, exactly and at full rate: 0x4B0000qq is the float
 * 8388608 + q (the integer conversion runs at a quarter of the rate on Volta) */
__device__ __forceinline__ float byte_f(uint64_t v, unsigned k) {
    return __int_as_float(__byte_perm((unsigned)(v >> (k >= 4 ? 32 : 0)), 0x4B000000u, 0x7440u | (k & 3))) - 8388608.f;
}

/* four IQ4_NL levels of four nibbles held one per byte: the low three bits
 * of each index, packed as __byte_perm's selector nibbles, pick from the
 * negative and the positive half of the table, the fourth bit between them */
__device__ __forceinline__ int iq4nl4(unsigned n) {
    const unsigned n7 = n & 0x07070707u, x = n7 | (n7 >> 4), sel4 = (x & 0xffu) | ((x >> 8) & 0xff00u);
    const unsigned lo = __byte_perm(0xbfad9881u, 0xf6eaddcfu, sel4);   /* -127 -104 -83 -65 -49 -35 -22 -10 */
    const unsigned hi = __byte_perm(0x26190d01u, 0x71594535u, sel4);   /* 1 13 25 38 53 69 89 113 */
    const unsigned sel = ((n >> 3) & 0x01010101u) * 0xffu;
    return (int)((lo & ~sel) | (hi & sel));
}

/* signed byte k of v as a float */
__device__ __forceinline__ float sbyte_f(uint64_t v, unsigned k) {
    return byte_f(v ^ 0x8080808080808080ull, k) - 128.f;
}

#define QWEN_NIBBLES 0x0f0f0f0f0f0f0f0full

/* Weights 8g..8g+7 of a row, decoded together: the eight share one codebook
 * entry and sign byte (IQ types) or one scale (K-quants, IQ4, Q2_0, Q5_0), so
 * the row dots give each lane whole groups instead of single weights. */
template<unsigned TYPE>
__device__ __forceinline__ void group8(const char *row, unsigned g, float w[8]) {
    if (TYPE == 1 || TYPE == 30) {   /* F16, BF16: one 16-byte load */
        const uint4 v = __ldg((const uint4 *)(row + 16 * g));
        const unsigned u[4] = { v.x, v.y, v.z, v.w };
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            if (TYPE == 1) {
                const float2 f = __half22float2(*(const __half2 *)&u[k]);
                w[2 * k] = f.x; w[2 * k + 1] = f.y;
            } else {
                w[2 * k] = __uint_as_float(u[k] << 16); w[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u);
            }
        }
        return;
    }
    if (TYPE == 8) {           /* Q8_0: 4 groups per 32-weight block */
        const uint8_t *b = (const uint8_t *)row + (g / 4) * 34;
        const float d = half_at(b);
        const uint64_t q = bytes8(b + 2 + 8 * (g % 4));
        #pragma unroll
        for (unsigned k = 0; k < 8; k++) w[k] = sbyte_f(q, k) * d;
        return;
    }
    if (TYPE == 42) {          /* Q2_0: 8 groups per 64-weight block */
        const uint8_t *b = (const uint8_t *)row + (g / 8) * 18;
        const unsigned q = u16_at(b + 2 + (g % 8) * 2);
        const float d = half_at(b);
        #pragma unroll
        for (unsigned k = 0; k < 8; k++) w[k] = (float)((int)((q >> (2 * k)) & 3) - 1) * d;
        return;
    }
    if (TYPE == 20 || TYPE == 6) {   /* IQ4_NL, Q5_0: 4 groups per 32-weight block */
        const unsigned q = g % 4;
        const uint8_t *b = (const uint8_t *)row + (g / 4) * (TYPE == 20 ? 18 : 22);
        const uint64_t lo = (bytes8(b + (TYPE == 20 ? 2 : 6) + 8 * (q & 1)) >> (4 * (q >> 1))) & QWEN_NIBBLES;
        const float d = half_at(b);
        const uint64_t lv = TYPE == 20 ? (uint32_t)iq4nl4((uint32_t)lo) | (uint64_t)(uint32_t)iq4nl4((uint32_t)(lo >> 32)) << 32 : 0;
        #pragma unroll
        for (unsigned k = 0; k < 8; k++) {
            if (TYPE == 20) w[k] = d * sbyte_f(lv, k);
            else w[k] = d * (float)((int)(byte_k(lo, k) | (((b[2 + q] >> k) & 1) << 4)) - 16);
        }
        return;
    }
    const uint8_t *b = (const uint8_t *)row + (g / 32) * row_bytes(TYPE, 256);
    const unsigned gi = g % 32, ib = gi / 4, l = gi % 4, j0 = gi * 8;
    if (TYPE == 12 || TYPE == 13) {  /* Q4_K, Q5_K: a 6-bit scale and min per 32 */
        unsigned sc, mn;
        scale_min_k4(b + 4, ib, sc, mn);
        const float d = half_at(b) * sc, m = half_at(b + 2) * mn;
        uint64_t q = (bytes8(b + (TYPE == 12 ? 16 : 48) + 32 * (j0 / 64) + j0 % 32) >> (4 * (ib & 1))) & QWEN_NIBBLES;
        if (TYPE == 13) q |= ((bytes8(b + 16 + j0 % 32) >> ib) & 0x0101010101010101ull) << 4;
        #pragma unroll
        for (unsigned k = 0; k < 8; k++) w[k] = d * byte_f(q, k) - m;
        return;
    }
    if (TYPE == 14 || TYPE == 11) {  /* Q6_K, Q3_K: a scale per 16 */
        const unsigned h = j0 / 128, gq = j0 % 128 / 32, e = j0 % 32;
        if (TYPE == 14) {
            const float d = half_at(b + 208) * (float)(int8_t)b[192 + 8 * h + e / 16 + 2 * gq];
            const uint64_t q = ((bytes8(b + 64 * h + e + 32 * (gq & 1)) >> (4 * (gq >> 1))) & QWEN_NIBBLES) |
                               (((bytes8(b + 128 + 32 * h + e) >> (2 * gq)) & 0x0303030303030303ull) << 4);
            #pragma unroll
            for (unsigned k = 0; k < 8; k++) w[k] = d * (byte_f(q, k) - 32.f);
        } else {
            const unsigned s = 8 * h + 2 * gq + e / 16;
            const int sc = (int)(((b[96 + s % 8] >> (4 * (s / 8))) & 15) |
                                 (((b[104 + s % 4] >> (2 * (s / 4))) & 3) << 4)) - 32;
            const float d = half_at(b + 108) * (float)sc;
            const uint64_t q = (bytes8(b + 32 + 32 * h + e) >> (2 * gq)) & 0x0303030303030303ull;
            const uint64_t hm = (bytes8(b + e) >> (4 * h + gq)) & 0x0101010101010101ull;
            #pragma unroll
            for (unsigned k = 0; k < 8; k++) w[k] = d * (float)((int)byte_k(q, k) - (byte_k(hm, k) ? 0 : 4));
        }
        return;
    }
    if (TYPE == 23) {          /* IQ4_XS: low nibbles are weights 0-15 of a 32, high 16-31 */
        const unsigned hi = u16_at(b + 2);
        const int ls = (int)(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((hi >> (2 * ib)) & 3) << 4)) - 32;
        const float d = half_at(b) * (float)ls;
        const uint64_t q = (bytes8(b + 8 + 16 * ib + 8 * (l & 1)) >> (4 * (l >> 1))) & QWEN_NIBBLES;
        const uint64_t lv = (uint32_t)iq4nl4((uint32_t)q) | (uint64_t)(uint32_t)iq4nl4((uint32_t)(q >> 32)) << 32;
        #pragma unroll
        for (unsigned k = 0; k < 8; k++) w[k] = d * sbyte_f(lv, k);
        return;
    }
    uint64_t grid;
    unsigned signs;
    float dl;
    if (TYPE == 17) {
        const unsigned q = u16_at(b + 2 + 2 * gi);
        dl = half_at(b) * (.5f + ((b[66 + ib] >> (4 * (l / 2))) & 15)) * .25f;
        grid = iq2xs_grid[q & 511];
        signs = ksigns_iq2xs[q >> 9];
    } else if (TYPE == 22) {
        dl = half_at(b) * (.5f + ((b[74 + ib] >> (4 * (l / 2))) & 15)) * .25f;
        grid = iq2s_grid[b[2 + gi] | (((b[66 + ib] >> (2 * l)) & 3) << 8)];
        signs = b[34 + gi];
    } else if (TYPE == 18) {
        const unsigned aux = u16_at(b + 66 + 4 * ib) | (u16_at(b + 68 + 4 * ib) << 16);
        const unsigned qi = u16_at(b + 2 + 2 * gi);
        dl = half_at(b) * (.5f + (aux >> 28)) * .5f;
        grid = iq3xxs_grid[qi & 255] | ((uint64_t)iq3xxs_grid[qi >> 8] << 32);
        signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
    } else {                   /* IQ3_S */
        dl = half_at(b) * (float)(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15));
        const unsigned qh = b[66 + ib] >> (2 * l), qi = u16_at(b + 2 + 2 * gi);
        grid = iq3s_grid[(qi & 255) | ((qh & 1) << 8)] |
               ((uint64_t)iq3s_grid[(qi >> 8) | (((qh >> 1) & 1) << 8)] << 32);
        signs = b[74 + gi];
    }
    /* the sign as a flipped sign bit gives -(dl*g), the product's own bits */
    #pragma unroll
    for (unsigned k = 0; k < 8; k++)
        w[k] = __int_as_float(__float_as_int(dl * byte_f(grid, k)) ^ (((signs >> k) & 1) << 31));
}

/* the types group8 decodes */
#define QWEN_GROUPED_TYPES(X) X(1) X(6) X(8) X(11) X(12) X(13) X(14) X(17) X(18) X(20) X(21) X(22) X(23) X(30) X(42)
__host__ __device__ constexpr bool grouped(unsigned type) {
#define QWEN_IS(TYPE) type == TYPE ||
    return QWEN_GROUPED_TYPES(QWEN_IS) false;
#undef QWEN_IS
}

/* whole groups of a row of n weights; F16/BF16 rows load 16 bytes at once */
template<unsigned TYPE>
__device__ __forceinline__ bool groups_fit(const char *row, const float *x, unsigned n) {
    return grouped(TYPE) && !(n % 8) && !((uintptr_t)x & 15) &&
           ((TYPE != 1 && TYPE != 30) || !((uintptr_t)row & 15));
}

/* These readers preserve the GGUF values, including padded Q2_K down rows.
 * Templates remove unused formats from each matrix kernel. */
template<unsigned TYPE>
__device__ __forceinline__ float value(const char *row, unsigned i,
        const uint64_t *grid_table = NULL, const uint8_t *sign_table = NULL) {
    if (TYPE == 0) return ((const float *)row)[i];
    if (TYPE == 1) return __half2float(((const __half *)row)[i]);
    if (TYPE == 30) return __bfloat162float(((const __nv_bfloat16 *)row)[i]);
    if (TYPE == 8) {
        const char *b = row + (i / 32) * 34;
        return __half2float(*(const __half *)b) * (float)((const int8_t *)b)[2 + i % 32];
    }
    if (TYPE == 2) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 18;
        return __half2float(*(const __half *)b) *
            (float)((int)((b[2 + i % 16] >> (4 * (i % 32 / 16))) & 15) - 8);
    }
    if (TYPE == 39) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 17;
        const unsigned q = (b[1 + i % 16] >> (4 * (i % 32 / 16))) & 15;
        const float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        const float scale = b[0] == 0 ? 0x1p-127f : __uint_as_float((unsigned)b[0] << 23);
        return (q & 8 ? -levels[q & 7] : levels[q & 7]) * scale;
    }
    if (TYPE == 10) {
        const cuda_block_q2_K *b = (const cuda_block_q2_K *)row + i / 256;
        const unsigned j = i % 256, sc = b->scales[j / 16];
        const unsigned q = (b->qs[j / 128 * 32 + j % 32] >> (2 * (j % 128 / 32))) & 3;
        return dev_f16_to_f32(b->d) * (sc & 15) * q - dev_f16_to_f32(b->dmin) * (sc >> 4);
    }
    if (TYPE == 12) {
        const cuda_block_q4_K *b = (const cuda_block_q4_K *)row + i / 256;
        const unsigned j = i % 256, group = j / 32;
        unsigned sc, mn;
        if (group < 4) { sc = b->scales[group] & 63; mn = b->scales[group + 4] & 63; }
        else {
            sc = (b->scales[group + 4] & 15) | ((b->scales[group - 4] >> 6) << 4);
            mn = (b->scales[group + 4] >> 4) | ((b->scales[group] >> 6) << 4);
        }
        const unsigned q = (b->qs[j / 64 * 32 + j % 32] >> (4 * (group & 1))) & 15;
        return dev_f16_to_f32(b->d) * sc * q - dev_f16_to_f32(b->dmin) * mn;
    }
    if (TYPE == 16) {
        const cuda_block_iq2_xxs *b = (const cuda_block_iq2_xxs *)row + i / 256;
        const unsigned j = i % 256, group = j / 32, sub = j % 32 / 8;
        const uint16_t *p = b->qs + group * 4;
        const unsigned grid_ids = (unsigned)p[0] | ((unsigned)p[1] << 16);
        const unsigned signs_scale = (unsigned)p[2] | ((unsigned)p[3] << 16);
        const unsigned gi = (grid_ids >> (8 * sub)) & 255, si = (signs_scale >> (7 * sub)) & 127;
        const uint64_t grid = grid_table ? grid_table[gi] : cuda_iq2xxs_grid[gi];
        const unsigned signs = sign_table ? sign_table[si] : cuda_ksigns_iq2xs[si];
        float v = (float)((grid >> (8 * (j & 7))) & 255);
        if (signs & (1u << (j & 7))) v = -v;
        return dev_f16_to_f32(b->d) * (.5f + (signs_scale >> 28)) * .25f * v;
    }
    if (TYPE == 6) {           /* Q5_0: d, qh (bit j is weight j's fifth bit), qs[16] */
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 22;
        const unsigned j = i % 32;
        const unsigned q = ((b[6 + j % 16] >> (4 * (j / 16))) & 15) | (((b[2 + j / 8] >> (j % 8)) & 1) << 4);
        return half_at(b) * (float)((int)q - 16);
    }
    if (TYPE == 20) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 18;
        return half_at(b) * (float)iq4nl_values[(b[2 + i % 16] >> (4 * (i % 32 / 16))) & 15];
    }
    if (TYPE == 42) {          /* Q2_0: d, 16 bytes of 2-bit weights, value q - 1 */
        const uint8_t *b = (const uint8_t *)row + (i / 64) * 18;
        const unsigned j = i % 64;
        return (float)((int)((b[2 + j / 4] >> (2 * (j % 4))) & 3) - 1) * half_at(b);
    }
    if (TYPE == 11 || TYPE == 13 || TYPE == 14 || TYPE == 17 || TYPE == 18 ||
        TYPE == 21 || TYPE == 22 || TYPE == 23)
        return value_mixed<TYPE>((const uint8_t *)row + (i / 256) * row_bytes(TYPE, 256), i % 256);
    return 0;
}

/* The shared expert slot, the router gate and the hc injection read rows of
 * a type known only at run time: a warp's row dot, and single weights. */
__device__ __forceinline__ float dot_any(const char *row, const float *x, unsigned n, unsigned type) {
#define QWEN_DOT(TYPE) case TYPE: return dot<TYPE>(row, x, n)
    switch (type) {
    QWEN_EACH_TYPE(QWEN_DOT);
    default: return 0;
    }
#undef QWEN_DOT
}

__device__ __forceinline__ float scalar(const char *row, unsigned i, unsigned type) {
#define QWEN_SCALAR(TYPE) case TYPE: return value<TYPE>(row, i)
    switch (type) {
    QWEN_EACH_TYPE(QWEN_SCALAR);
    default: return 0;
    }
#undef QWEN_SCALAR
}

template<unsigned TYPE>
__device__ __forceinline__ float4 value4(const char *row, unsigned i,
        const uint64_t *grid_table, const uint8_t *sign_table) {
    float4 out;
    float *v = (float *)&out;
    if (TYPE == 1) {
        const uint2 bits = *(const uint2 *)(row+i*2);
        const float2 a = __half22float2(*(__half2 *)&bits.x), b = __half22float2(*(__half2 *)&bits.y);
        out = make_float4(a.x,a.y,b.x,b.y);
    } else if (TYPE == 0) {
        out = *(const float4 *)(row+i*4);
    } else if (TYPE == 8) {
        const char *b = row+(i/32)*34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b+2+i%32);
        out = make_float4((float)(int8_t)q[0]*scale,(float)(int8_t)(q[0]>>8)*scale,
                         (float)(int8_t)q[1]*scale,(float)(int8_t)(q[1]>>8)*scale);
    } else if (TYPE == 39) {
        const uint8_t *b = (const uint8_t *)row+(i/32)*17;
        const float scale = b[0] == 0 ? 0x1p-127f : __uint_as_float((unsigned)b[0]<<23);
        unsigned packed;
        memcpy(&packed,b+1+i%16,4);
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const unsigned q = (packed>>(k*8+(i%32/16)*4))&15, mag = q&7;
            const float level = mag < 2 ? .5f*mag : __uint_as_float(((mag/2+126)<<23)|((mag&1)<<22));
            v[k] = (q&8 ? -level : level)*scale;
        }
    } else if (TYPE == 16) {
        const cuda_block_iq2_xxs *b = (const cuda_block_iq2_xxs *)row+i/256;
        const unsigned j = i%256, sub = (j%32)/8;
        const uint16_t *p = b->qs+(j/32)*4;
        const unsigned ids = (unsigned)p[0]|((unsigned)p[1]<<16);
        const unsigned ss = (unsigned)p[2]|((unsigned)p[3]<<16);
        const uint64_t grid = grid_table[(ids>>(8*sub))&255];
        const unsigned signs = sign_table[(ss>>(7*sub))&127];
        const float scale = dev_f16_to_f32(b->d)*(.5f+(ss>>28))*.25f;
        /* Apply four signs before conversion. IQ2's nonzero magnitudes keep
         * the packed negations from carrying into neighboring bytes. */
        const unsigned offset = j&7;
        const unsigned bits = (((signs>>offset)&15)*0x00204081u)&0x01010101u;
        const unsigned packed = (((unsigned)(grid>>(8*offset)))^(bits*255u))+bits;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            v[k] = scale*(float)(int8_t)(packed>>(8*k));
        }
    } else if (TYPE == 10 || TYPE == 12) {
        const unsigned j = i%256;
        unsigned sc, mn, qs, shift;
        float d, dm;
        if (TYPE == 10) {
            const cuda_block_q2_K *b = (const cuda_block_q2_K *)row+i/256;
            const unsigned s = b->scales[j/16];
            sc = s&15; mn = s>>4;
            d = dev_f16_to_f32(b->d); dm = dev_f16_to_f32(b->dmin);
            qs = *(const unsigned *)(b->qs+(j/128)*32+j%32);
            shift = 2*((j%128)/32);
        } else {
            const cuda_block_q4_K *b = (const cuda_block_q4_K *)row+i/256;
            const unsigned group = j/32;
            if (group < 4) { sc = b->scales[group]&63; mn = b->scales[group+4]&63; }
            else {
                sc = (b->scales[group+4]&15)|((b->scales[group-4]>>6)<<4);
                mn = (b->scales[group+4]>>4)|((b->scales[group]>>6)<<4);
            }
            d = dev_f16_to_f32(b->d); dm = dev_f16_to_f32(b->dmin);
            qs = *(const unsigned *)(b->qs+(j/64)*32+j%32);
            shift = 4*(group&1);
        }
        const float scale = d*sc, offset = dm*mn;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) v[k] = scale*((qs>>(8*k+shift))&(TYPE == 10 ? 3 : 15))-offset;
    } else {
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) v[k] = value<TYPE>(row,i+k,grid_table,sign_table);
    }
    return out;
}

static uint64_t expert_row_bytes(unsigned type, unsigned K) {
    if (type == 10 && K % 32) return 0;
    return row_bytes(type, type == 10 ? ((uint64_t)K + 255) / 256 * 256 : K);
}

/* A token's routing in one warp: each lane holds 16 of the (at most 512)
 * expert probabilities in registers, so the softmax and the NS argmax rounds
 * take shuffles instead of block barriers (26 to 3 us per token on a V100).
 * Ties go to the lower expert, and the weights are normalized in selection
 * order, as before.  A second warp takes the shared expert's gate dot
 * meanwhile, with the row dot's wide loads: its row is cold in memory. */
__global__ void router(int *selected, float *weights, const float *logits,
        const float *x, const char *gate, float *shared_gate,
        unsigned NE, unsigned NS, unsigned K, unsigned type) {
    const unsigned t = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    if (warp) {
        if (K) {
            const float v = dot_any(gate, x + (uint64_t)t * K, K, type);
            if (!lane) shared_gate[t] = v;
        }
        return;
    }
    float p[16], mx = -FLT_MAX;
    #pragma unroll
    for (unsigned i = 0; i < 16; i++) {
        const unsigned e = lane + 32 * i;
        p[i] = e < NE ? logits[(uint64_t)t * NE + e] : -FLT_MAX;
        mx = fmaxf(mx, p[i]);
    }
    for (unsigned d = 16; d; d >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
    float ps = 0;
    #pragma unroll
    for (unsigned i = 0; i < 16; i++) {
        p[i] = lane + 32 * i < NE ? expf(p[i] - mx) : 0;
        ps += p[i];
    }
    const float total = sum(ps);
    #pragma unroll
    for (unsigned i = 0; i < 16; i++) p[i] = lane + 32 * i < NE ? p[i] / total : -1;
    float mine = 0, denom = 0;
    for (unsigned s = 0; s < NS; s++) {
        float best = -1;
        unsigned id = UINT_MAX;
        #pragma unroll
        for (unsigned i = 0; i < 16; i++) if (p[i] > best) { best = p[i]; id = lane + 32 * i; }
        for (unsigned d = 16; d; d >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffff, best, d);
            const unsigned oi = __shfl_xor_sync(0xffffffff, id, d);
            if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
        }
        #pragma unroll
        for (unsigned i = 0; i < 16; i++) if (lane + 32 * i == id) p[i] = -1;
        if (!lane) selected[(uint64_t)t * NS + s] = id;
        if (lane == s) mine = best;
        denom += best;
    }
    if (lane < NS) weights[(uint64_t)t * NS + lane] = mine / denom;
}

/* Decode with int8 activations (decode and verify rows of the routed
 * experts): the activations are rounded to int8 per 32 values, as llama.cpp
 * does, and eight weights become two words of int8 for two dp4a, where the
 * FP32 dot spends a conversion, a scale and an FMA on every weight; the
 * expert rows are as fast as their memory allows only this way on Volta.
 * group8i gives the eight weights of group g and their scale d, and for the
 * K-quants with a minimum m, so that the group's sum is d*q.x - m*sum(x). */
/* the four bytes of g negated where bit k of the nibble bits is set.  The
 * grids hold no zero byte, so the per-byte negation (g ^ ff) + 1 never
 * carries; Volta has no hardware for the SIMD intrinsics (__vcmpne4,
 * __vsub4) that would do it otherwise. */
/* the four bytes a - b, each byte of a below 128 and of b at most 128: the
 * high bit set in each byte of a stops a borrow at the byte (__vsub4 has no
 * hardware on Volta either) */
__device__ __forceinline__ int sub4(unsigned a, unsigned b) {
    return (int)(((a | 0x80808080u) - b) ^ 0x80808080u);
}
__device__ __forceinline__ int signed4(unsigned g, unsigned bits) {
    const unsigned b01 = ((bits & 15) * 0x00204081u) & 0x01010101u;
    return (int)((g ^ (b01 * 0xffu)) + b01);
}

__host__ __device__ constexpr bool grouped_i8(unsigned type) {
    return type == 6 || type == 8 || type == 11 || type == 12 || type == 13 || type == 14 || type == 16 ||
           type == 17 || type == 18 || type == 20 || type == 21 || type == 22 || type == 23 || type == 42;
}

template<unsigned TYPE>
__device__ __forceinline__ void group8i(const char *row, unsigned g, int &w0, int &w1, float &d, float &m,
        const void *grid_table = NULL, const uint8_t *sign_table = NULL) {
    m = 0;
    if (TYPE == 8) {
        const uint8_t *b = (const uint8_t *)row + (g / 4) * 34;
        const uint64_t q = bytes8(b + 2 + 8 * (g % 4));
        w0 = (int)(uint32_t)q; w1 = (int)(uint32_t)(q >> 32); d = half_at(b);
        return;
    }
    if (TYPE == 42) {
        const uint8_t *b = (const uint8_t *)row + (g / 8) * 18;
        const unsigned q = u16_at(b + 2 + (g % 8) * 2);
        const unsigned lo = (q & 3) | ((q >> 2) & 3) << 8 | ((q >> 4) & 3) << 16 | ((q >> 6) & 3) << 24;
        const unsigned hi = ((q >> 8) & 3) | ((q >> 10) & 3) << 8 | ((q >> 12) & 3) << 16 | ((q >> 14) & 3) << 24;
        w0 = sub4(lo, 0x01010101u); w1 = sub4(hi, 0x01010101u); d = half_at(b);
        return;
    }
    if (TYPE == 20 || TYPE == 6) {
        const unsigned q = g % 4;
        const uint8_t *b = (const uint8_t *)row + (g / 4) * (TYPE == 20 ? 18 : 22);
        const uint64_t lo = (bytes8(b + (TYPE == 20 ? 2 : 6) + 8 * (q & 1)) >> (4 * (q >> 1))) & QWEN_NIBBLES;
        d = half_at(b);
        if (TYPE == 20) { w0 = iq4nl4((unsigned)lo); w1 = iq4nl4((unsigned)(lo >> 32)); return; }
        const unsigned h = b[2 + q];   /* the fifth bits of these eight weights */
        const unsigned h0 = ((h & 15) * 0x00204081u) & 0x01010101u, h1 = (((h >> 4) & 15) * 0x00204081u) & 0x01010101u;
        w0 = sub4((unsigned)lo | h0 << 4, 0x10101010u);
        w1 = sub4((unsigned)(lo >> 32) | h1 << 4, 0x10101010u);
        return;
    }
    const uint8_t *b = (const uint8_t *)row + (g / 32) * row_bytes(TYPE, 256);
    const unsigned gi = g % 32, ib = gi / 4, l = gi % 4, j0 = gi * 8;
    if (TYPE == 12 || TYPE == 13) {
        unsigned sc, mn;
        scale_min_k4(b + 4, ib, sc, mn);
        d = half_at(b) * sc; m = half_at(b + 2) * mn;
        uint64_t q = (bytes8(b + (TYPE == 12 ? 16 : 48) + 32 * (j0 / 64) + j0 % 32) >> (4 * (ib & 1))) & QWEN_NIBBLES;
        if (TYPE == 13) q |= ((bytes8(b + 16 + j0 % 32) >> ib) & 0x0101010101010101ull) << 4;
        w0 = (int)(uint32_t)q; w1 = (int)(uint32_t)(q >> 32);
        return;
    }
    if (TYPE == 14) {
        const unsigned h = j0 / 128, gq = j0 % 128 / 32, e = j0 % 32;
        d = half_at(b + 208) * (float)(int8_t)b[192 + 8 * h + e / 16 + 2 * gq];
        const uint64_t q = ((bytes8(b + 64 * h + e + 32 * (gq & 1)) >> (4 * (gq >> 1))) & QWEN_NIBBLES) |
                           (((bytes8(b + 128 + 32 * h + e) >> (2 * gq)) & 0x0303030303030303ull) << 4);
        w0 = sub4((uint32_t)q, 0x20202020u); w1 = sub4((uint32_t)(q >> 32), 0x20202020u);
        return;
    }
    if (TYPE == 11) {
        const unsigned h = j0 / 128, gq = j0 % 128 / 32, e = j0 % 32, s = 8 * h + 2 * gq + e / 16;
        d = half_at(b + 108) * (float)((int)(((b[96 + s % 8] >> (4 * (s / 8))) & 15) |
                                            (((b[104 + s % 4] >> (2 * (s / 4))) & 3) << 4)) - 32);
        const uint64_t q = (bytes8(b + 32 + 32 * h + e) >> (2 * gq)) & 0x0303030303030303ull;
        const uint64_t lo4 = (~(bytes8(b + e) >> (4 * h + gq)) & 0x0101010101010101ull) << 2;   /* 4 where the high bit is clear */
        w0 = sub4((uint32_t)q, (uint32_t)lo4); w1 = sub4((uint32_t)(q >> 32), (uint32_t)(lo4 >> 32));
        return;
    }
    if (TYPE == 23) {
        const unsigned hi = u16_at(b + 2);
        d = half_at(b) * (float)((int)(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((hi >> (2 * ib)) & 3) << 4)) - 32);
        const uint64_t q = (bytes8(b + 8 + 16 * ib + 8 * (l & 1)) >> (4 * (l >> 1))) & QWEN_NIBBLES;
        w0 = iq4nl4((uint32_t)q); w1 = iq4nl4((uint32_t)(q >> 32));
        return;
    }
    uint64_t grid;
    unsigned signs;
    if (TYPE == 16) {          /* IQ2_XXS: 8 grid indices and 4 x 7 sign bits with a scale per 32 */
        const unsigned ids = u16_at(b + 2 + 8 * ib) | (u16_at(b + 4 + 8 * ib) << 16);
        const unsigned ss = u16_at(b + 6 + 8 * ib) | (u16_at(b + 8 + 8 * ib) << 16);
        d = half_at(b) * (.5f + (ss >> 28)) * .25f;
        /* the constant-memory tables serialize a warp's different entries */
        grid = grid_table ? ((const uint64_t *)grid_table)[(ids >> (8 * l)) & 255] : cuda_iq2xxs_grid[(ids >> (8 * l)) & 255];
        signs = sign_table ? sign_table[(ss >> (7 * l)) & 127] : cuda_ksigns_iq2xs[(ss >> (7 * l)) & 127];
    } else if (TYPE == 17) {
        const unsigned q = u16_at(b + 2 + 2 * gi);
        d = half_at(b) * (.5f + ((b[66 + ib] >> (4 * (l / 2))) & 15)) * .25f;
        grid = grid_table ? ((const uint64_t *)grid_table)[q & 511] : iq2xs_grid[q & 511];
        signs = sign_table ? sign_table[q >> 9] : ksigns_iq2xs[q >> 9];
    } else if (TYPE == 22) {
        d = half_at(b) * (.5f + ((b[74 + ib] >> (4 * (l / 2))) & 15)) * .25f;
        const unsigned gi2 = b[2 + gi] | (((b[66 + ib] >> (2 * l)) & 3) << 8);
        grid = grid_table ? ((const uint64_t *)grid_table)[gi2] : iq2s_grid[gi2];
        signs = b[34 + gi];
    } else if (TYPE == 18) {
        const unsigned aux = u16_at(b + 66 + 4 * ib) | (u16_at(b + 68 + 4 * ib) << 16);
        const unsigned qi = u16_at(b + 2 + 2 * gi);
        d = half_at(b) * (.5f + (aux >> 28)) * .5f;
        const uint32_t *t = grid_table ? (const uint32_t *)grid_table : iq3xxs_grid;
        grid = t[qi & 255] | ((uint64_t)t[qi >> 8] << 32);
        signs = sign_table ? sign_table[(aux >> (7 * l)) & 127] : ksigns_iq2xs[(aux >> (7 * l)) & 127];
    } else {                   /* IQ3_S */
        d = half_at(b) * (float)(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15));
        const unsigned qh = b[66 + ib] >> (2 * l), qi = u16_at(b + 2 + 2 * gi);
        const uint32_t *t = grid_table ? (const uint32_t *)grid_table : iq3s_grid;
        grid = t[(qi & 255) | ((qh & 1) << 8)] | ((uint64_t)t[(qi >> 8) | (((qh >> 1) & 1) << 8)] << 32);
        signs = b[74 + gi];
    }
    w0 = signed4((uint32_t)grid, signs); w1 = signed4((uint32_t)(grid >> 32), signs >> 4);
}

/* group32i: the 32 weights of sub-block s of a row (groups 4s to 4s+3) as
 * eight words of int8, with the scale d0 of words 0-3 and d1 of words 4-7,
 * for the types whose sub-blocks of 32 share their fields: each is loaded
 * once, where four group8i load it four times. */
__device__ __forceinline__ unsigned u32_at2(const uint8_t *p) {   /* 4 bytes at a 2-aligned address */
    return u16_at(p) | (u16_at(p + 2) << 16);
}

__host__ __device__ constexpr bool grouped_i32(unsigned type) {
    return type == 8 || type == 12 || type == 13 || type == 14 || type == 16 || type == 17 || type == 18 ||
           type == 20 || type == 21 || type == 22 || type == 23 || type == 42;
}

/* m: the K-quants' minimum, which the sub-block's sum takes m * sum(x) off */
template<unsigned TYPE>
__device__ __forceinline__ void group32i(const char *row, unsigned s, int (&w)[8], float &d0, float &d1, float &m,
        const void *grid_table = NULL, const uint8_t *sign_table = NULL) {
    m = 0;
    if (TYPE == 8) {           /* Q8_0: one 34-byte block of 32 */
        const uint8_t *b = (const uint8_t *)row + s * 34;
        d0 = d1 = half_at(b);
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const uint64_t q = bytes8(b + 2 + 8 * k);
            w[2 * k] = (int)(uint32_t)q; w[2 * k + 1] = (int)(uint32_t)(q >> 32);
        }
        return;
    }
    if (TYPE == 12 || TYPE == 13) {   /* Q4_K, Q5_K: the nibbles of 32 bytes, Q5_K's fifth bits */
        const uint8_t *b = (const uint8_t *)row + (s / 8) * (TYPE == 12 ? 144 : 176);
        const unsigned ib = s % 8, sh = 4 * (ib & 1);
        unsigned sc, mn;
        scale_min_k4(b + 4, ib, sc, mn);
        d0 = d1 = half_at(b) * sc;
        m = half_at(b + 2) * mn;
        const uint8_t *qs = b + (TYPE == 12 ? 16 : 48) + 32 * (ib / 2);
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const uint64_t q = bytes8(qs + 8 * k);
            unsigned lo = ((uint32_t)q >> sh) & 0x0f0f0f0fu, hi = ((uint32_t)(q >> 32) >> sh) & 0x0f0f0f0fu;
            if (TYPE == 13) {
                const uint64_t h = bytes8(b + 16 + 8 * k);
                lo |= (((uint32_t)h >> ib) & 0x01010101u) << 4;
                hi |= (((uint32_t)(h >> 32) >> ib) & 0x01010101u) << 4;
            }
            w[2 * k] = (int)lo; w[2 * k + 1] = (int)hi;
        }
        return;
    }
    if (TYPE == 14) {          /* Q6_K: 32 weights of a half, a scale per 16 */
        const uint8_t *b = (const uint8_t *)row + (s / 8) * 210;
        const unsigned sb = s % 8, h = sb / 4, gq = sb % 4;
        const float d = half_at(b + 208);
        d0 = d * (float)(int8_t)b[192 + 8 * h + 2 * gq];
        d1 = d * (float)(int8_t)b[193 + 8 * h + 2 * gq];
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const uint64_t q = ((bytes8(b + 64 * h + 32 * (gq & 1) + 8 * k) >> (4 * (gq >> 1))) & QWEN_NIBBLES) |
                               (((bytes8(b + 128 + 32 * h + 8 * k) >> (2 * gq)) & 0x0303030303030303ull) << 4);
            w[2 * k] = sub4((uint32_t)q, 0x20202020u); w[2 * k + 1] = sub4((uint32_t)(q >> 32), 0x20202020u);
        }
        return;
    }
    if (TYPE == 20) {          /* IQ4_NL: one 18-byte block of 32 */
        const uint8_t *b = (const uint8_t *)row + s * 18;
        const uint64_t lo = bytes8(b + 2), hi = bytes8(b + 10);
        d0 = d1 = half_at(b);
        w[0] = iq4nl4((unsigned)lo & 0x0f0f0f0fu); w[1] = iq4nl4((unsigned)(lo >> 32) & 0x0f0f0f0fu);
        w[2] = iq4nl4((unsigned)hi & 0x0f0f0f0fu); w[3] = iq4nl4((unsigned)(hi >> 32) & 0x0f0f0f0fu);
        w[4] = iq4nl4((unsigned)(lo >> 4) & 0x0f0f0f0fu); w[5] = iq4nl4((unsigned)(lo >> 36) & 0x0f0f0f0fu);
        w[6] = iq4nl4((unsigned)(hi >> 4) & 0x0f0f0f0fu); w[7] = iq4nl4((unsigned)(hi >> 36) & 0x0f0f0f0fu);
        return;
    }
    if (TYPE == 42) {          /* Q2_0: half of an 18-byte block of 64 */
        const uint8_t *b = (const uint8_t *)row + (s / 2) * 18;
        const uint64_t q = bytes8(b + 2 + 8 * (s % 2));
        d0 = d1 = half_at(b);
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const unsigned v = (unsigned)(q >> (16 * k));
            const unsigned lo = (v & 3) | ((v >> 2) & 3) << 8 | ((v >> 4) & 3) << 16 | ((v >> 6) & 3) << 24;
            const unsigned hi = ((v >> 8) & 3) | ((v >> 10) & 3) << 8 | ((v >> 12) & 3) << 16 | ((v >> 14) & 3) << 24;
            w[2 * k] = sub4(lo, 0x01010101u); w[2 * k + 1] = sub4(hi, 0x01010101u);
        }
        return;
    }
    const uint8_t *b = (const uint8_t *)row + (s / 8) * row_bytes(TYPE, 256);
    const unsigned ib = s % 8;
    if (TYPE == 23) {          /* IQ4_XS */
        const unsigned hs = u16_at(b + 2);
        d0 = d1 = half_at(b) * (float)((int)(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((hs >> (2 * ib)) & 3) << 4)) - 32);
        const uint64_t lo = bytes8(b + 8 + 16 * ib), hi = bytes8(b + 16 + 16 * ib);
        w[0] = iq4nl4((unsigned)lo & 0x0f0f0f0fu); w[1] = iq4nl4((unsigned)(lo >> 32) & 0x0f0f0f0fu);
        w[2] = iq4nl4((unsigned)hi & 0x0f0f0f0fu); w[3] = iq4nl4((unsigned)(hi >> 32) & 0x0f0f0f0fu);
        w[4] = iq4nl4((unsigned)(lo >> 4) & 0x0f0f0f0fu); w[5] = iq4nl4((unsigned)(lo >> 36) & 0x0f0f0f0fu);
        w[6] = iq4nl4((unsigned)(hi >> 4) & 0x0f0f0f0fu); w[7] = iq4nl4((unsigned)(hi >> 36) & 0x0f0f0f0fu);
        return;
    }
    const float d = half_at(b);
    if (TYPE == 21) {          /* IQ3_S */
        d0 = d1 = d * (float)(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15));
        const unsigned qh = b[66 + ib], sg = u32_at2(b + 74 + 4 * ib);
        const uint64_t qs = bytes8(b + 2 + 8 * ib);
        const uint32_t *t = grid_table ? (const uint32_t *)grid_table : iq3s_grid;
        #pragma unroll
        for (unsigned l = 0; l < 4; l++) {
            const unsigned lo = byte_k(qs, 2 * l) | ((qh >> (2 * l)) & 1) << 8;
            const unsigned hi = byte_k(qs, 2 * l + 1) | ((qh >> (2 * l + 1)) & 1) << 8;
            const unsigned sl = sg >> (8 * l);
            w[2 * l] = signed4(t[lo], sl); w[2 * l + 1] = signed4(t[hi], sl >> 4);
        }
        return;
    }
    if (TYPE == 18) {          /* IQ3_XXS */
        const unsigned aux = u32_at2(b + 66 + 4 * ib);
        d0 = d1 = d * (.5f + (aux >> 28)) * .5f;
        const uint64_t qs = bytes8(b + 2 + 8 * ib);
        const uint32_t *t = grid_table ? (const uint32_t *)grid_table : iq3xxs_grid;
        #pragma unroll
        for (unsigned l = 0; l < 4; l++) {
            const unsigned sl = sign_table ? sign_table[(aux >> (7 * l)) & 127] : ksigns_iq2xs[(aux >> (7 * l)) & 127];
            w[2 * l] = signed4(t[byte_k(qs, 2 * l)], sl);
            w[2 * l + 1] = signed4(t[byte_k(qs, 2 * l + 1)], sl >> 4);
        }
        return;
    }
    if (TYPE == 16) {          /* IQ2_XXS */
        const uint64_t q = bytes8(b + 2 + 8 * ib);
        const unsigned ids = (unsigned)q, ss = (unsigned)(q >> 32);
        d0 = d1 = d * (.5f + (ss >> 28)) * .25f;
        #pragma unroll
        for (unsigned l = 0; l < 4; l++) {
            const uint64_t g = grid_table ? ((const uint64_t *)grid_table)[(ids >> (8 * l)) & 255]
                                          : cuda_iq2xxs_grid[(ids >> (8 * l)) & 255];
            const unsigned sl = sign_table ? sign_table[(ss >> (7 * l)) & 127] : cuda_ksigns_iq2xs[(ss >> (7 * l)) & 127];
            w[2 * l] = signed4((unsigned)g, sl); w[2 * l + 1] = signed4((unsigned)(g >> 32), sl >> 4);
        }
        return;
    }
    if (TYPE == 17) {          /* IQ2_XS: a scale per 16 */
        const unsigned sc = b[66 + ib];
        d0 = d * (.5f + (sc & 15)) * .25f;
        d1 = d * (.5f + (sc >> 4)) * .25f;
        const uint64_t q = bytes8(b + 2 + 8 * ib);
        #pragma unroll
        for (unsigned l = 0; l < 4; l++) {
            const unsigned v = (unsigned)(q >> (16 * l)) & 0xffff;
            const uint64_t g = grid_table ? ((const uint64_t *)grid_table)[v & 511] : iq2xs_grid[v & 511];
            const unsigned sl = sign_table ? sign_table[v >> 9] : ksigns_iq2xs[v >> 9];
            w[2 * l] = signed4((unsigned)g, sl); w[2 * l + 1] = signed4((unsigned)(g >> 32), sl >> 4);
        }
        return;
    }
    if (TYPE == 22) {          /* IQ2_S: a scale per 16 */
        const unsigned sc = b[74 + ib], qh = b[66 + ib];
        d0 = d * (.5f + (sc & 15)) * .25f;
        d1 = d * (.5f + (sc >> 4)) * .25f;
        const unsigned qs = u32_at2(b + 2 + 4 * ib), sg = u32_at2(b + 34 + 4 * ib);
        #pragma unroll
        for (unsigned l = 0; l < 4; l++) {
            const unsigned gi = ((qs >> (8 * l)) & 255) | (((qh >> (2 * l)) & 3) << 8);
            const uint64_t g = grid_table ? ((const uint64_t *)grid_table)[gi] : iq2s_grid[gi];
            const unsigned sl = sg >> (8 * l);
            w[2 * l] = signed4((unsigned)g, sl); w[2 * l + 1] = signed4((unsigned)(g >> 32), sl >> 4);
        }
        return;
    }
}

/* The int8 sums with every rounding spelled out: fast math may contract a
 * product and a sum into an FMA in one instantiation of a kernel and not in
 * another (a row alone, or among a verify's), and a verify row would no
 * longer get plain decode's bits.  A group's term is d*dot, less m*sum(x)
 * for the K-quants' minimum. */
template<unsigned TYPE>
__device__ __forceinline__ float q8_term(float d, int dot, float m, int sx) {
    if (TYPE == 12 || TYPE == 13) return __fmaf_rn(d, (float)dot, __fmul_rn(-m, (float)sx));
    return __fmul_rn(d, (float)dot);
}
__device__ __forceinline__ float q8_halves(float d0, int p0, float d1, int p1) {
    return __fmaf_rn(d0, (float)p0, __fmul_rn(d1, (float)p1));
}
/* a sub-block of 32: the halves' terms, less m * sum(x) for the K-quants */
template<unsigned TYPE>
__device__ __forceinline__ float q8_sub32(float d0, int p0, float d1, int p1, float m, int sx) {
    if (TYPE == 12 || TYPE == 13) return __fmaf_rn(-m, (float)sx, q8_halves(d0, p0, d1, p1));
    return q8_halves(d0, p0, d1, p1);
}
__device__ __forceinline__ int sum32(int4 a, int4 b) {   /* the sum of 32 int8 */
    return __dp4a(0x01010101, a.x, __dp4a(0x01010101, a.y, __dp4a(0x01010101, a.z, __dp4a(0x01010101, a.w,
           __dp4a(0x01010101, b.x, __dp4a(0x01010101, b.y, __dp4a(0x01010101, b.z, __dp4a(0x01010101, b.w, 0))))))));
}

/* Round rows of n activations (n a multiple of 32) to int8 with a scale per
 * 32: one warp per 32 values.  The divisions are IEEE ones, which fast math
 * would not give: a value then rounds the same on every GPU and in the
 * tests' CPU reference. */
__global__ void quantize_q8(int8_t *q, float *d, const float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    float amax = fabsf(v);
    for (unsigned o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float s = __fdiv_rn(amax, 127.f);
    q[i] = (int8_t)__float2int_rn(s > 0 ? __fdiv_rn(v, s) : 0.f);
    if (!(threadIdx.x & 31)) d[i / 32] = s;
}

/* Decode and verify rows read int8 activations (DS4_QWEN4_F32_ACT=1 keeps
 * FP32 for A/B): Volta decodes the weights four to a dp4a instead of one
 * FMA each, and a verify row's activations are a quarter of the bytes. */
static bool act_q8() {
    static const bool on = getenv("DS4_QWEN4_F32_ACT") == NULL;
    return on;
}

/* ds4_gpu_qwen4_q8_input's rows: an activation rounded once for every
 * product that reads it next, one per device. */
static struct { const void *x; unsigned T, K; int8_t *q; float *d; uint64_t cap; } g_q8_input[DS4_MAX_GPUS];

static int current_tier() {
    int dev = 0, tier = 0;
    if (g_n_gpus > 1 && cudaGetDevice(&dev) == cudaSuccess)
        for (int t = 0; t < g_n_gpus; t++) if (g_gpu[t].device_id == dev) tier = t;
    return tier;
}

/* Activations their producer rounds to int8 as it writes them (the hc
 * norm's xn, the GDN output), a few per device by tensor.  A decode-sized
 * writer of such a tensor that does not round it drops its entry. */
struct q8_entry { const void *x; unsigned T, K; int8_t *q; float *d; uint64_t cap; };
static q8_entry g_q8_out[DS4_MAX_GPUS][4];

/* the buffers a producer of x's T rows of K fills, now x's entry */
static bool q8_out(const void *x, unsigned T, unsigned K, int8_t **q, float **d) {
    if (!act_q8() || !T || T > 8 || !K || K % 32) return false;
    q8_entry *e = g_q8_out[current_tier()];
    unsigned i = 0;
    while (i < 4 && e[i].x != x) i++;
    if (i == 4) for (i = 0; i < 3 && e[i].x; i++) {}
    const uint64_t n = (uint64_t)T * K, qbytes = (n + 15) / 16 * 16, bytes = qbytes + n / 32 * 4;
    e[i].x = NULL;
    if (e[i].cap < bytes) {
        if (e[i].q) (void)cudaFree(e[i].q);
        e[i].q = NULL;
        e[i].cap = 0;
        if (cudaMalloc(&e[i].q, bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        e[i].cap = bytes;
    }
    e[i].x = x; e[i].T = T; e[i].K = K;
    e[i].d = (float *)(e[i].q + qbytes);
    *q = e[i].q;
    *d = e[i].d;
    return true;
}

static void q8_out_drop(const void *x) {
    for (q8_entry &e : g_q8_out[current_tier()]) if (e.x == x) e.x = NULL;
}

/* x's T rows of K rounded to int8: the registered ones when x is the
 * registered input or a producer rounded it, else rounded now into the
 * scratch */
static bool q8_rows(const float *x, unsigned T, unsigned K, const int8_t **xq, const float **xd) {
    const int tier = current_tier();
    if (g_q8_input[tier].x == x && g_q8_input[tier].T == T && g_q8_input[tier].K == K) {
        *xq = g_q8_input[tier].q;
        *xd = g_q8_input[tier].d;
        return true;
    }
    for (const q8_entry &e : g_q8_out[tier]) {
        if (e.x == x && e.T == T && e.K == K) {
            *xq = e.q;
            *xd = e.d;
            return true;
        }
    }
    const uint64_t n = (uint64_t)T * K, qbytes = (n + 15) / 16 * 16;
    int8_t *q = (int8_t *)tmp_alloc(qbytes + n / 32 * 4, "Qwen int8 activations");
    if (!q) return false;
    quantize_q8<<<(unsigned)((n + 255) / 256),256,0,cuda_decode_stream()>>>(q, (float *)(q + qbytes), x, n);
    *xq = q;
    *xd = (float *)(q + qbytes);
    return launched();
}

__host__ __device__ constexpr bool iq_codebook(unsigned type) {
    return type == 16 || type == 17 || type == 18 || type == 20 || type == 21 || type == 22 || type == 23;
}

/* NR weight rows against the rows of up to ROWS tokens with int8
 * activations: each group's weights are decoded once for every token in
 * mask, whose activations are rows xr[j] of xq.  A lane takes CH groups (CH
 * * 8 consecutive weights) a step.  A token's sum is the same whichever
 * others share the pass. */
template<unsigned TYPE, unsigned ROWS, unsigned NR, unsigned CH>
__device__ __forceinline__ void doti_rows(float (&acc)[NR][ROWS], const char *const (&r)[NR],
        const int8_t *xq, const float *xd, const unsigned (&xr)[ROWS], unsigned mask, unsigned K,
        const void *grid_table = NULL, const uint8_t *sign_table = NULL) {
    for (unsigned c = threadIdx.x & 31; c < K / (8 * CH); c += 32) {
        if constexpr (CH == 4 && grouped_i32(TYPE)) {
            int q[NR][8];
            float d0[NR], d1[NR], mm[NR];
            #pragma unroll
            for (unsigned i = 0; i < NR; i++) group32i<TYPE>(r[i], c, q[i], d0[i], d1[i], mm[i], grid_table, sign_table);
            #pragma unroll
            for (unsigned j = 0; j < ROWS; j++) if (mask >> j & 1) {
                const int8_t *xt = xq + (uint64_t)xr[j] * K + 32 * c;
                const int4 xa = *(const int4 *)xt, xb = *(const int4 *)(xt + 16);
                const float s = xd[(uint64_t)xr[j] * (K / 32) + c];
                const int sx = TYPE == 12 || TYPE == 13 ? sum32(xa, xb) : 0;
                #pragma unroll
                for (unsigned i = 0; i < NR; i++) {
                    const int p0 = __dp4a(q[i][0], xa.x, __dp4a(q[i][1], xa.y, __dp4a(q[i][2], xa.z, __dp4a(q[i][3], xa.w, 0))));
                    const int p1 = __dp4a(q[i][4], xb.x, __dp4a(q[i][5], xb.y, __dp4a(q[i][6], xb.z, __dp4a(q[i][7], xb.w, 0))));
                    acc[i][j] = __fmaf_rn(s, q8_sub32<TYPE>(d0[i], p0, d1[i], p1, mm[i], sx), acc[i][j]);
                }
            }
        } else {
            int w0[NR][CH], w1[NR][CH];
            float d[NR][CH], m[NR][CH];
            #pragma unroll
            for (unsigned i = 0; i < NR; i++)
                #pragma unroll
                for (unsigned o = 0; o < CH; o++) group8i<TYPE>(r[i], CH * c + o, w0[i][o], w1[i][o], d[i][o], m[i][o],
                                                              grid_table, sign_table);
            #pragma unroll
            for (unsigned j = 0; j < ROWS; j++) if (mask >> j & 1) {
                const int8_t *xt = xq + (uint64_t)xr[j] * K + 8 * CH * c;
                const float s = xd[((uint64_t)xr[j] * K + 8 * CH * c) / 32];
                int2 x[CH];
                int sx[CH];
                #pragma unroll
                for (unsigned o = 0; o < CH; o++) {
                    x[o] = *(const int2 *)(xt + 8 * o);
                    sx[o] = TYPE == 12 || TYPE == 13 ? __dp4a(0x01010101, x[o].x, __dp4a(0x01010101, x[o].y, 0)) : 0;
                }
                #pragma unroll
                for (unsigned i = 0; i < NR; i++) {
                    float a = 0;
                    #pragma unroll
                    for (unsigned o = 0; o < CH; o++)
                        a = __fadd_rn(a, q8_term<TYPE>(d[i][o], __dp4a(w0[i][o], x[o].x, __dp4a(w1[i][o], x[o].y, 0)),
                                                       m[i][o], sx[o]));
                    acc[i][j] = __fmaf_rn(s, a, acc[i][j]);
                }
            }
        }
    }
}

/* the expert rows' geometry: the gate and up rows of one output together,
 * the codebook types' 32 weights a lane and step; a down row of 640
 * weights four lanes, eight rows a warp (moe_quad_q8).  None depends on
 * the batch. */
template<unsigned TYPE> struct moe_geometry {
    static constexpr unsigned CH = iq_codebook(TYPE) ? 4 : 1;
};

template<unsigned ROWS>
__device__ __forceinline__ void moe_rows_q8_any(float (&acc)[1][ROWS], const char *const (&r)[1], const int8_t *xq,
        const float *xd, const unsigned (&xr)[ROWS], unsigned mask, unsigned K, unsigned type) {
#define QWEN_ROWS_Q8(TYPE) case TYPE: doti_rows<TYPE, ROWS, 1, moe_geometry<TYPE>::CH>(acc, r, xq, xd, xr, mask, K); break;
    switch (type) {
    QWEN_ROWS_Q8(6) QWEN_ROWS_Q8(8) QWEN_ROWS_Q8(11) QWEN_ROWS_Q8(12) QWEN_ROWS_Q8(13) QWEN_ROWS_Q8(14)
    QWEN_ROWS_Q8(16) QWEN_ROWS_Q8(17) QWEN_ROWS_Q8(18) QWEN_ROWS_Q8(20) QWEN_ROWS_Q8(21) QWEN_ROWS_Q8(22)
    QWEN_ROWS_Q8(23) QWEN_ROWS_Q8(42)
    default: break;
    }
#undef QWEN_ROWS_Q8
}

/* A down row with four lanes, each every fourth sub-block of 32 weights
 * (five of a 640-weight row): a warp of 32 lanes over one such row leaves
 * a third of them idle and gives each a short chain of loads; with eight
 * rows a warp each lane decodes five independent sub-blocks (Q2_0 down rows
 * of a three-row verify: 41 us to 34 in isolation).  The caller adds the
 * four lanes' sums. */
template<unsigned TYPE, unsigned ROWS>
__device__ __forceinline__ void moe_quad_q8(float (&acc)[ROWS], const char *r, const int8_t *xq, const float *xd,
        const unsigned (&xr)[ROWS], unsigned mask, unsigned K,
        const void *grid_table = NULL, const uint8_t *sign_table = NULL) {
    for (unsigned c = threadIdx.x & 3; c < K / 32; c += 4) {
        int q[8];
        float d[4], m[4] = {};
        if constexpr (grouped_i32(TYPE)) {
            group32i<TYPE>(r, c, q, d[0], d[2], m[0], grid_table, sign_table);
            d[1] = d[0];
            d[3] = d[2];
            m[1] = m[2] = m[3] = m[0];
        } else {
            #pragma unroll
            for (unsigned o = 0; o < 4; o++) group8i<TYPE>(r, 4 * c + o, q[2 * o], q[2 * o + 1], d[o], m[o], grid_table, sign_table);
        }
        #pragma unroll
        for (unsigned j = 0; j < ROWS; j++) if (mask >> j & 1) {
            const int8_t *xt = xq + (uint64_t)xr[j] * K + 32 * c;
            const int4 xa = *(const int4 *)xt, xb = *(const int4 *)(xt + 16);
            const int x[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
            float a = 0;
            #pragma unroll
            for (unsigned o = 0; o < 4; o++) {
                const int sx = TYPE == 12 || TYPE == 13 ? __dp4a(0x01010101, x[2 * o], __dp4a(0x01010101, x[2 * o + 1], 0)) : 0;
                a = __fadd_rn(a, q8_term<TYPE>(d[o], __dp4a(q[2 * o], x[2 * o], __dp4a(q[2 * o + 1], x[2 * o + 1], 0)), m[o], sx));
            }
            acc[j] = __fmaf_rn(xd[(uint64_t)xr[j] * (K / 32) + c], a, acc[j]);
        }
    }
}

template<unsigned ROWS>
__device__ __forceinline__ void moe_quad_q8_any(float (&acc)[ROWS], const char *r, const int8_t *xq, const float *xd,
        const unsigned (&xr)[ROWS], unsigned mask, unsigned K, unsigned type) {
#define QWEN_QUAD_Q8(TYPE) case TYPE: moe_quad_q8<TYPE, ROWS>(acc, r, xq, xd, xr, mask, K); break;
    switch (type) {
    QWEN_QUAD_Q8(6) QWEN_QUAD_Q8(8) QWEN_QUAD_Q8(11) QWEN_QUAD_Q8(12) QWEN_QUAD_Q8(13) QWEN_QUAD_Q8(14)
    QWEN_QUAD_Q8(16) QWEN_QUAD_Q8(17) QWEN_QUAD_Q8(18) QWEN_QUAD_Q8(20) QWEN_QUAD_Q8(21) QWEN_QUAD_Q8(22)
    QWEN_QUAD_Q8(23) QWEN_QUAD_Q8(42)
    default: break;
    }
#undef QWEN_QUAD_Q8
}

/* The expert kernels stage the IQ types' codebooks in shared memory: a
 * warp's lanes look up 32 random entries at once, which a table in global
 * memory serves through L1 a line at a time (IQ3_S gate and up rows of a
 * three-row verify: 62 us against 70). */
__host__ __device__ constexpr unsigned iq_grid_bytes(unsigned type) {
    return type == 16 ? 2048 : type == 17 ? 4096 : type == 22 ? 8192 : type == 18 ? 1024 : type == 21 ? 2048 : 0;
}
__host__ __device__ constexpr bool iq_grid_signs(unsigned type) { return type == 16 || type == 17 || type == 18; }

template<unsigned TYPE>
__device__ __forceinline__ void stage_iq_grid(uint8_t *grid, uint8_t *signs) {
    if (TYPE == 16) for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) ((uint64_t *)grid)[i] = cuda_iq2xxs_grid[i];
    if (TYPE == 17) for (unsigned i = threadIdx.x; i < 512; i += blockDim.x) ((uint64_t *)grid)[i] = iq2xs_grid[i];
    if (TYPE == 22) for (unsigned i = threadIdx.x; i < 1024; i += blockDim.x) ((uint64_t *)grid)[i] = iq2s_grid[i];
    if (TYPE == 18) for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) ((uint32_t *)grid)[i] = iq3xxs_grid[i];
    if (TYPE == 21) for (unsigned i = threadIdx.x; i < 512; i += blockDim.x) ((uint32_t *)grid)[i] = iq3s_grid[i];
    if (iq_grid_signs(TYPE))
        for (unsigned i = threadIdx.x; i < 128; i += blockDim.x) signs[i] = TYPE == 16 ? cuda_ksigns_iq2xs[i] : ksigns_iq2xs[i];
}

/* moe_mv with int8 activations (xq/xd hold the rows the FP32 kernel reads
 * from x, rounded per 32), and an expert's rows read once for all the
 * tokens of a batch that picked it: a three-row MTP verify of iq3m8 picks
 * 21.6 distinct experts for its 30 (token, slot) pairs on code.  The block
 * of (rows, slot, token) finds the pairs of its expert with a ballot over
 * the selection; only the first token's block computes them, the others
 * leave at once.  The shared expert's block of token 0 computes every
 * token.  A gate/up warp takes one row of each, a down warp eight rows.
 * 64 registers (a few spilled in the shared slot's switch) keep 32 warps
 * an SM; at 96 to 136 the rows waited on memory 10% longer. */
template<unsigned TYPE, bool DOWN, unsigned ROWS>
__global__ void __launch_bounds__(128, 8) moe_mv_q8(float *out, const int8_t *xq, const float *xd, const int *selected,
        const char *w0, const char *w1, const char *sh0, const char *sh1,
        unsigned shared_type, unsigned shared_up_type, unsigned NE, unsigned T, unsigned NS, unsigned K, unsigned M,
        uint64_t rb, uint64_t srb, uint64_t surb) {
    constexpr unsigned RW = DOWN ? 8 : 1;
    const unsigned lane = threadIdx.x & 31, row0 = (blockIdx.x * 4 + threadIdx.x / 32) * RW, slot = blockIdx.y, t = blockIdx.z;
    const bool shared = slot == NS;
    const unsigned stride = NS + (shared_type != UINT_MAX), n = T * NS;
    /* one load of the selection serves the ballot and this block's expert */
    const int mine = lane < n ? selected[lane] : -1;
    const int e = shared ? -1 : t * NS + slot < 32 ? __shfl_sync(0xffffffff, mine, t * NS + slot)
                                                   : selected[(uint64_t)t * NS + slot];
    const bool valid = shared || (e >= 0 && (unsigned)e < NE);
    unsigned mask = 1u << t, slots = slot << (4 * t);   /* the tokens computed here, and their slots */
    if (shared) {
        if (t) return;
        mask = (1u << T) - 1;
        slots = NS * 0x11111111u;
    } else if (valid) {
        mask = slots = 0;
        for (unsigned base = 0; base < n; base += 32) {
            const int v = base ? (base + lane < n ? selected[base + lane] : -1) : mine;
            for (unsigned hit = __ballot_sync(0xffffffff, v == e); hit; hit &= hit - 1) {
                const unsigned i = base + __ffs(hit) - 1, tok = i / NS;
                if (tok < t) return;
                mask |= 1u << tok;
                slots |= (i - tok * NS) << (4 * tok);
            }
        }
    }
    constexpr unsigned GB = iq_grid_bytes(TYPE);
    __shared__ __align__(16) uint8_t grid_table[GB ? GB : 16];
    __shared__ uint8_t sign_table[iq_grid_signs(TYPE) ? 128 : 1];
    if constexpr (GB > 0) {
        stage_iq_grid<TYPE>(grid_table, sign_table);
        __syncthreads();
    }
    if (row0 >= M) return;
    unsigned xr[ROWS];
    #pragma unroll
    for (unsigned j = 0; j < ROWS; j++) xr[j] = DOWN ? j * stride + ((slots >> (4 * j)) & 15) : j;
    if constexpr (DOWN) {
        /* rows past M (M % 8) repeat the last one and are not stored */
        const unsigned mine_row = row0 + lane / 4, row = min(mine_row, M - 1);
        float a[ROWS] = {};
        if (shared) moe_quad_q8_any<ROWS>(a, sh0 + row * srb, xq, xd, xr, mask, K, shared_type);
        else if (valid) moe_quad_q8<TYPE, ROWS>(a, w0 + ((uint64_t)e * M + row) * rb, xq, xd, xr, mask, K,
                                                grid_table, sign_table);
        #pragma unroll
        for (unsigned j = 0; j < ROWS; j++) if (mask >> j & 1) {
            float v = a[j];
            v += __shfl_xor_sync(0xffffffff, v, 1);
            v += __shfl_xor_sync(0xffffffff, v, 2);
            if (!(lane & 3) && mine_row < M) out[((uint64_t)j * stride + ((slots >> (4 * j)) & 15)) * M + mine_row] = v;
        }
    } else {
        float a[2][ROWS] = {};
        if (shared) {
            /* the shared gate and up rows may differ in type */
            float g[1][ROWS] = {}, u[1][ROWS] = {};
            const char *rg[1] = { sh0 + row0 * srb }, *ru[1] = { sh1 + row0 * surb };
            moe_rows_q8_any<ROWS>(g, rg, xq, xd, xr, mask, K, shared_type);
            moe_rows_q8_any<ROWS>(u, ru, xq, xd, xr, mask, K, shared_up_type);
            #pragma unroll
            for (unsigned j = 0; j < ROWS; j++) { a[0][j] = g[0][j]; a[1][j] = u[0][j]; }
        } else if (valid) {
            const uint64_t off = ((uint64_t)e * M + row0) * rb;
            const char *r[2] = { w0 + off, w1 + off };
            doti_rows<TYPE, ROWS, 2, moe_geometry<TYPE>::CH>(a, r, xq, xd, xr, mask, K, grid_table, sign_table);
        }
        #pragma unroll
        for (unsigned j = 0; j < ROWS; j++) if (mask >> j & 1) {
            const float g = sum(a[0][j]), u = sum(a[1][j]);
            if (!lane) out[((uint64_t)j * stride + ((slots >> (4 * j)) & 15)) * M + row0] = silu(g) * u;
        }
    }
}

/* The shared expert's gate and up rows may differ in type (31 of iq3m's 48
 * layers); its slot reads them as shared_type and shared_up_type. */
template<unsigned TYPE, bool DOWN>
__global__ void moe_mv(float *out, const float *x, const int *selected,
        const char *w0, const char *w1, const char *sh0, const char *sh1,
        unsigned shared_type, unsigned shared_up_type, unsigned NE, unsigned NS, unsigned K, unsigned M,
        uint64_t rb, uint64_t srb, uint64_t surb) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, slot = blockIdx.y, t = blockIdx.z;
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = threadIdx.x; i < 128; i += blockDim.x) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    if (row >= M) return;
    const bool shared = slot == NS;
    const unsigned stride = NS + (shared_type != UINT_MAX);
    const uint64_t pair = (uint64_t)t * stride + slot;
    const float *xt = x + (DOWN ? pair : t) * K;
    float a = 0, b = 0;
    if (shared) {
        a = dot_any(sh0 + row * srb, xt, K, shared_type);
        if (!DOWN) b = dot_any(sh1 + row * surb, xt, K, shared_up_type);
    } else {
        const int e = selected[(uint64_t)t * NS + slot];
        if (e >= 0 && (unsigned)e < NE) {
            const uint64_t off = ((uint64_t)e * M + row) * rb;
            if ((TYPE == 16 || TYPE == 10 || TYPE == 12 || TYPE == 39) && !((uintptr_t)xt&15)) {
                for (unsigned i = (threadIdx.x&31)*4; i < K; i += 128) {
                    const float4 xv = *(const float4 *)(xt+i);
                    const float4 av = value4<TYPE>(w0+off,i,grid_table,sign_table);
                    a += av.x*xv.x; a += av.y*xv.y; a += av.z*xv.z; a += av.w*xv.w;
                    if (!DOWN) {
                        const float4 bv = value4<TYPE>(w1+off,i,grid_table,sign_table);
                        b += bv.x*xv.x; b += bv.y*xv.y; b += bv.z*xv.z; b += bv.w*xv.w;
                    }
                }
                a = sum(a); b = sum(b);
            } else {
                a = dot<TYPE>(w0+off,xt,K,grid_table,sign_table);
                if (!DOWN) b = dot<TYPE>(w1+off,xt,K,grid_table,sign_table);
            }
        }
    }
    if (!(threadIdx.x & 31)) out[pair * M + row] = DOWN ? a : silu(a) * b;
}

static int moe_mv_dispatch(float *out, const float *x, const int *sel,
        const char *w0, const char *w1, const char *s0, const char *s1,
        unsigned type, unsigned st, unsigned sut, unsigned NE, unsigned T, unsigned NS, unsigned K, unsigned M, bool down) {
    const uint64_t rb = expert_row_bytes(type, K), srb = row_bytes(st, K), surb = row_bytes(sut, K);
    const dim3 grid((M + 3) / 4, NS + (st != UINT_MAX), T);
    /* int8 activations where every weight type of the layer has an int8
     * decoder; a down warp takes eight rows */
    if (act_q8() && T <= 8 && NS < 16 && grouped_i8(type) && (st == UINT_MAX || (grouped_i8(st) && grouped_i8(sut))) &&
        !(K % 32)) {
        const int8_t *xq;
        const float *xd;
        if (!q8_rows(x, down ? T * (NS + (st != UINT_MAX)) : T, K, &xq, &xd)) return 0;
        const dim3 grid_down((M + 31) / 32, grid.y, grid.z);
#define QWEN_MOE_Q8_ROWS(TYPE, R) \
        if (down) moe_mv_q8<TYPE, true, R><<<grid_down,128,0,cuda_decode_stream()>>>(out,xq,xd,sel,w0,w1,s0,s1,st,sut,NE,T,NS,K,M,rb,srb,surb); \
        else moe_mv_q8<TYPE, false, R><<<grid,128,0,cuda_decode_stream()>>>(out,xq,xd,sel,w0,w1,s0,s1,st,sut,NE,T,NS,K,M,rb,srb,surb)
#define QWEN_MOE_Q8(TYPE) case TYPE: \
        if (T == 1) { QWEN_MOE_Q8_ROWS(TYPE, 1); } else if (T <= 4) { QWEN_MOE_Q8_ROWS(TYPE, 4); } else { QWEN_MOE_Q8_ROWS(TYPE, 8); } break;
        switch (type) {
        QWEN_MOE_Q8(6) QWEN_MOE_Q8(8) QWEN_MOE_Q8(11) QWEN_MOE_Q8(12) QWEN_MOE_Q8(13) QWEN_MOE_Q8(14) QWEN_MOE_Q8(16)
        QWEN_MOE_Q8(17) QWEN_MOE_Q8(18) QWEN_MOE_Q8(20) QWEN_MOE_Q8(21) QWEN_MOE_Q8(22) QWEN_MOE_Q8(23) QWEN_MOE_Q8(42)
        default: return 0;
        }
#undef QWEN_MOE_Q8
#undef QWEN_MOE_Q8_ROWS
        return launched();
    }
#define QWEN_MOE(TYPE) case TYPE: \
    if (down) moe_mv<TYPE, true><<<grid,128,0,cuda_decode_stream()>>>(out,x,sel,w0,w1,s0,s1,st,sut,NE,NS,K,M,rb,srb,surb); \
    else moe_mv<TYPE, false><<<grid,128,0,cuda_decode_stream()>>>(out,x,sel,w0,w1,s0,s1,st,sut,NE,NS,K,M,rb,srb,surb); break
    switch (type) {
    QWEN_EACH_TYPE(QWEN_MOE);
    default: return 0;
    }
#undef QWEN_MOE
    return launched();
}

__global__ void moe_reduce(float *out, float *R, const float *inj, const float *part,
        const float *weights, const float *gate, const float *shared,
        unsigned NS, unsigned stride, unsigned D, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    __shared__ float inject[4];
    if (threadIdx.x < hc) inject[threadIdx.x] = injection(inj + (uint64_t)t * hc * hc * 8, hc, threadIdx.x);
    __syncthreads();
    if (d >= D) return;
    float v = 0;
    for (unsigned s = 0; s < NS; s++) v += weights[(uint64_t)t * NS + s] * part[((uint64_t)t * stride + s) * D + d];
    if (gate) v += sigmoid(gate[t]) * (shared ? shared[(uint64_t)t * D + d] : part[((uint64_t)t * stride + NS) * D + d]);
    out[(uint64_t)t * D + d] = v;
    for (unsigned s = 0; s < hc; s++) R[((uint64_t)t * hc + s) * D + d] += inject[s] * v;
}

__global__ void expert_lists(int *lists, int *counts, const int *selected,
        unsigned pairs, unsigned NE, unsigned cap) {
    __shared__ unsigned counters[512];
    for (unsigned e = threadIdx.x; e < NE; e += blockDim.x) counters[e] = 0;
    __syncthreads();
    for (unsigned p = threadIdx.x; p < pairs; p += blockDim.x) {
        const unsigned e = (unsigned)selected[p];
        if (e < NE) {
            const unsigned i = atomicAdd(counters + e, 1u);
            if (i < cap) lists[(uint64_t)e * cap + i] = p;
        }
    }
    __syncthreads();
    for (unsigned e = threadIdx.x; e < NE; e += blockDim.x) counts[e] = min(counters[e], cap);
}

/* Tiles share dequantized weights across 16 tokens without requantizing the
 * activations. Expert counts determine the work; no expert list is truncated. */
template<unsigned TYPE, bool DOWN, bool EXPERT>
__global__ void matrix(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, unsigned T, unsigned NS, unsigned NO,
        unsigned K, unsigned M, unsigned cap, uint64_t rb) {
    const unsigned r = threadIdx.x % 16, c = threadIdx.x / 16, e = blockIdx.y;
    const unsigned row = blockIdx.x * 16 + r;
    const unsigned count = EXPERT ? (unsigned)counts[e] : T;
    __shared__ float a[16][33], b[16][33], v[16][33];
    for (unsigned t0 = blockIdx.z * 16; t0 < count; t0 += gridDim.z * 16) {
        const unsigned item = t0 + c;
        const int pair = EXPERT && item < count ? lists[(uint64_t)e * cap + item] : (int)item;
        const unsigned tok = EXPERT ? pair / NS : item, slot = EXPERT ? pair % NS : 0;
        float acc = 0, up = 0;
        for (unsigned k0 = 0; k0 < K; k0 += 32) {
            for (unsigned j = threadIdx.x; j < 16 * 32; j += 256) {
                const unsigned rr = j / 32, kk = j % 32, global_row = blockIdx.x * 16 + rr;
                const unsigned logical = k0 + kk;
                const char *wr = w0 + ((uint64_t)(EXPERT ? e : 0) * M + global_row) * rb;
                a[rr][kk] = global_row < M && logical < K ? value<TYPE>(wr, logical) : 0;
                if (EXPERT && !DOWN) {
                    const char *ur = w1 + ((uint64_t)e * M + global_row) * rb;
                    b[rr][kk] = global_row < M && logical < K ? value<TYPE>(ur, logical) : 0;
                }
                const unsigned ti = t0 + rr;
                const int pp = EXPERT && ti < count ? lists[(uint64_t)e * cap + ti] : (int)ti;
                const unsigned tt = EXPERT ? pp / NS : ti, ss = EXPERT ? pp % NS : 0;
                const uint64_t xx = DOWN && EXPERT ? (uint64_t)tt * NO + ss : tt;
                v[rr][kk] = ti < count && logical < K ? x[xx * K + logical] : 0;
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k = 0; k < 32; k++) {
                acc += a[r][k] * v[c][k];
                if (EXPERT && !DOWN) up += b[r][k] * v[c][k];
            }
            __syncthreads();
        }
        if (row < M && item < count) {
            const uint64_t op = EXPERT ? (uint64_t)tok * NO + slot : item;
            out[op * M + row] = EXPERT && !DOWN ? silu(acc) * up : acc;
        }
    }
}

__global__ void expert_tiles(unsigned *prefix, const int *counts, unsigned NE, unsigned nt) {
    unsigned n = 0;
    prefix[0] = 0;
    for (unsigned e = 0; e < NE; e++) {
        n += ((unsigned)counts[e]+nt-1)/nt;
        prefix[e+1] = n;
    }
}

/* Two TF32 components retain the fine part of each FP32 operand. Keep the
 * correction products in their own accumulator: repeatedly adding them to
 * the much larger leading product loses their precision on tensor cores. */
template<unsigned TYPE>
__global__ void matrix_tc(float *out, const float *x, const char *w0,
        unsigned T, unsigned K, unsigned M, uint64_t rb) {
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32, r0 = blockIdx.x*64;
    /* Padding separates the banks of adjacent rows in the TF32 fragment
     * loads. The leading planes also hold the completed output tiles. */
    __shared__ __align__(32) float ah[64][36], al[64][36];
    __shared__ __align__(32) float bh[16][36], bl[16][36];
    for (unsigned t0 = blockIdx.z*16; t0 < T; t0 += gridDim.z*16) {
        wm::fragment<wm::accumulator,16,16,8,float> acc, corr, total;
        wm::fill_fragment(acc,0);
        wm::fill_fragment(corr,0);
        wm::fill_fragment(total,0);
        for (unsigned k0 = 0; k0 < K; k0 += 32) {
            for (unsigned j = tid; j < 64*32; j += 128) {
                const unsigned r = j/32, kk = j%32, row = r0+r, col = k0+kk;
                const uint64_t off = (uint64_t)row*rb;
                const float a = row < M && col < K ? value<TYPE>(w0+off,col) : 0;
                ah[r][kk] = wm::__float_to_tf32(a);
                al[r][kk] = wm::__float_to_tf32(a-ah[r][kk]);
            }
            for (unsigned j = tid; j < 16*32; j += 128) {
                const unsigned item = t0+j/32, kk = j%32, col = k0+kk;
                const float b = item < T && col < K ? x[(uint64_t)item*K+col] : 0;
                bh[j/32][kk] = wm::__float_to_tf32(b);
                bl[j/32][kk] = wm::__float_to_tf32(b-bh[j/32][kk]);
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k = 0; k < 32; k += 8) {
                wm::fragment<wm::matrix_a,16,16,8,wm::precision::tf32,wm::row_major> a, alo;
                wm::fragment<wm::matrix_b,16,16,8,wm::precision::tf32,wm::col_major> b, blo;
                wm::load_matrix_sync(a,&ah[warp*16][k],36);
                wm::load_matrix_sync(alo,&al[warp*16][k],36);
                wm::load_matrix_sync(b,&bh[0][k],36);
                wm::load_matrix_sync(blo,&bl[0][k],36);
                wm::mma_sync(corr,alo,blo,corr);
                wm::mma_sync(corr,alo,b,corr);
                wm::mma_sync(corr,a,blo,corr);
                wm::mma_sync(acc,a,b,acc);
            }
            __syncthreads();
            /* Bound tensor-core accumulation depth. CUDA FP32 adds the
             * partial sums, avoiding drift on long HC projection rows. */
            if ((k0+32)%256 == 0 || k0+32 >= K) {
                for (unsigned i = 0; i < acc.num_elements; i++) {
                    total.x[i] += acc.x[i] + corr.x[i];
                }
                wm::fill_fragment(acc,0);
                wm::fill_fragment(corr,0);
            }
        }
        wm::store_matrix_sync(&ah[warp*16][0],total,36,wm::mem_row_major);
        __syncthreads();
        for (unsigned j = tid; j < 64*16; j += 128) {
            const unsigned row = r0+j/16, item = t0+j%16;
            if (row < M && item < T) out[(uint64_t)item*M+row] = ah[j/16][j%16];
        }
        __syncthreads();
    }
#endif
}

__device__ __forceinline__ void mma_tf32(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

template<unsigned TYPE, bool DOWN, unsigned NC>
__global__ void matrix_reg(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb) {
#if __CUDA_ARCH__ >= 800
    const unsigned lane = threadIdx.x&31, warp = threadIdx.x/32;
    const unsigned row_tiles = (M+31)/32, job = blockIdx.x/row_tiles;
    if (job >= tiles[NE]) return;
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = threadIdx.x; i < 128; i += blockDim.x) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    unsigned e = 0, end = NE;
    while (e < end) {
        const unsigned mid = (e+end)/2;
        if (tiles[mid+1] <= job) e = mid+1;
        else end = mid;
    }
    const unsigned count = counts[e], t0 = (job-tiles[e])*16*NC;
    const unsigned r0 = (blockIdx.x%row_tiles)*32+(warp%2)*16, c0 = t0+(warp/2)*8*NC;
    if (r0 >= M || c0 >= count) return;
    float acc[NC][4] = {}, corr[NC][4] = {}, total[NC][4] = {};
    float uacc[NC][4] = {}, ucorr[NC][4] = {}, utotal[NC][4] = {};
    const float *xp[NC];
    #pragma unroll
    for (unsigned nc = 0; nc < NC; nc++) {
        const unsigned item = c0+nc*8+lane/4;
        const unsigned pair = item < count ? lists[(uint64_t)e*cap+item] : 0;
        xp[nc] = x+(DOWN ? (uint64_t)(pair/NS)*NO+pair%NS : pair/NS)*K;
    }
    for (unsigned k0 = 0; k0 < K; k0 += 8) {
        unsigned a[4], al[4], u[4], ul[4], b[2], bl[2];
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = r0+lane/4+(i%2)*8, k = k0+lane%4+(i/2)*4;
            const uint64_t off = ((uint64_t)e*M+r)*rb;
            const float v = r < M && k < K ? value<TYPE>(w0+off,k,grid_table,sign_table) : 0;
            const float hi = nvcuda::wmma::__float_to_tf32(v);
            a[i] = __float_as_uint(hi);
            al[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            if (!DOWN) {
                const float v = r < M && k < K ? value<TYPE>(w1+off,k,grid_table,sign_table) : 0;
                const float hi = nvcuda::wmma::__float_to_tf32(v);
                u[i] = __float_as_uint(hi);
                ul[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            }
        }
        #pragma unroll
        for (unsigned nc = 0; nc < NC; nc++) {
            if (c0+nc*8 >= count) continue;
            const unsigned item = c0+nc*8+lane/4;
            #pragma unroll
            for (unsigned i = 0; i < 2; i++) {
                const unsigned k = k0+lane%4+i*4;
                const float v = item < count && k < K ? xp[nc][k] : 0;
                const float hi = nvcuda::wmma::__float_to_tf32(v);
                b[i] = __float_as_uint(hi);
                bl[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            }
            mma_tf32(corr[nc],al,bl);
            mma_tf32(corr[nc],al,b);
            mma_tf32(corr[nc],a,bl);
            mma_tf32(acc[nc],a,b);
            if (!DOWN) {
                mma_tf32(ucorr[nc],ul,bl);
                mma_tf32(ucorr[nc],ul,b);
                mma_tf32(ucorr[nc],u,bl);
                mma_tf32(uacc[nc],u,b);
            }
            if ((k0+8)%256 == 0 || k0+8 >= K) {
                #pragma unroll
                for (unsigned i = 0; i < 4; i++) {
                    total[nc][i] += acc[nc][i]+corr[nc][i];
                    acc[nc][i] = corr[nc][i] = 0;
                    if (!DOWN) { utotal[nc][i] += uacc[nc][i]+ucorr[nc][i]; uacc[nc][i] = ucorr[nc][i] = 0; }
                }
            }
        }
    }
    #pragma unroll
    for (unsigned nc = 0; nc < NC; nc++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned row = r0+lane/4+(i/2)*8, item = c0+nc*8+(lane%4)*2+(i%2);
            if (row < M && item < count) {
                const unsigned pair = lists[(uint64_t)e*cap+item];
                out[((uint64_t)(pair/NS)*NO+pair%NS)*M+row] = DOWN ? total[nc][i] : silu(total[nc][i])*utotal[nc][i];
            }
        }
    }
#endif
}

/* Match the half operands of the production Metal expert tiles. Products
 * accumulate in FP32; only the matrix operands are rounded to half. */
template<unsigned TYPE, bool DOWN, unsigned NT, unsigned NR = 64>
__global__ void matrix_half_tile(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb) {
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32, nr = (M+NR-1)/NR, job = blockIdx.x/nr;
    if (job >= tiles[NE]) return;
    unsigned e = 0, end = NE;
    while (e < end) {
        const unsigned mid = (e+end)/2;
        if (tiles[mid+1] <= job) e = mid+1; else end = mid;
    }
    const unsigned count = counts[e], t0 = (job-tiles[e])*NT, r0 = (blockIdx.x%nr)*NR;
    const unsigned wr = warp%(NR/16), wc = warp/(NR/16);
    constexpr unsigned NC = NT/(8/(NR/16));
    union Tile {
        __half ab[(NR+(DOWN ? 0 : NR)+NT)*72];
        float c[NR][NT];
    };
    __shared__ __align__(32) Tile tile;
    __half (*a)[72] = (__half (*)[72])tile.ab;
    __half (*u)[72] = a+NR;
    __half (*b)[72] = a+NR+(DOWN ? 0 : NR);
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = tid; i < 256; i += 256) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = tid; i < 128; i += 256) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    wm::fragment<wm::accumulator,16,16,16,float> acc[NC/16], up[NC/16];
    #pragma unroll
    for (unsigned j = 0; j < NC/16; j++) {
        wm::fill_fragment(acc[j],0);
        if (!DOWN) wm::fill_fragment(up[j],0);
    }
    for (unsigned k0 = 0; k0 < K; k0 += 64) {
        for (unsigned i = tid*4; i < NR*64; i += 256*4) {
            const unsigned row = r0+i/64, k = k0+i%64;
            const uint64_t off = ((uint64_t)e*M+row)*rb;
            const float4 av = row < M && k < K ? value4<TYPE>(w0+off,k,grid_table,sign_table) : make_float4(0,0,0,0);
            *(__half2 *)&a[i/64][i%64] = __floats2half2_rn(av.x,av.y);
            *(__half2 *)&a[i/64][i%64+2] = __floats2half2_rn(av.z,av.w);
            if (!DOWN) {
                const float4 uv = row < M && k < K ? value4<TYPE>(w1+off,k,grid_table,sign_table) : make_float4(0,0,0,0);
                *(__half2 *)&u[i/64][i%64] = __floats2half2_rn(uv.x,uv.y);
                *(__half2 *)&u[i/64][i%64+2] = __floats2half2_rn(uv.z,uv.w);
            }
        }
        for (unsigned i = tid; i < NT*64; i += 256) {
            const unsigned item = t0+i/64, k = k0+i%64;
            const unsigned pair = item < count ? lists[(uint64_t)e*cap+item] : 0;
            const uint64_t row = DOWN ? (uint64_t)(pair/NS)*NO+pair%NS : pair/NS;
            b[i/64][i%64] = __float2half_rn(item < count && k < K ? x[row*K+k] : 0);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 64; k += 16) {
            wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> af, uf;
            wm::load_matrix_sync(af,&a[wr*16][k],72);
            if (!DOWN) wm::load_matrix_sync(uf,&u[wr*16][k],72);
            #pragma unroll
            for (unsigned j = 0; j < NC/16; j++) {
                wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major> bf;
                wm::load_matrix_sync(bf,&b[wc*NC+j*16][k],72);
                wm::mma_sync(acc[j],af,bf,acc[j]);
                if (!DOWN) wm::mma_sync(up[j],uf,bf,up[j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned j = 0; j < NC/16; j++) {
        if (!DOWN) for (unsigned i = 0; i < acc[j].num_elements; i++)
            acc[j].x[i] = silu(acc[j].x[i])*up[j].x[i];
        wm::store_matrix_sync(&tile.c[wr*16][wc*NC+j*16],acc[j],NT,wm::mem_row_major);
    }
    __syncthreads();
    for (unsigned i = tid; i < NR*NT; i += 256) {
        const unsigned row = r0+i/NT, item = t0+i%NT;
        if (row < M && item < count) {
            const unsigned pair = lists[(uint64_t)e*cap+item];
            out[((uint64_t)(pair/NS)*NO+pair%NS)*M+row] = tile.c[i/NT][i%NT];
        }
    }
#endif
}

__global__ void half_copy(__half *out, const float *x, uint64_t n4) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const float4 v = ((const float4 *)x)[i];
    ((__half2 *)out)[2 * i] = __floats2half2_rn(v.x, v.y);
    ((__half2 *)out)[2 * i + 1] = __floats2half2_rn(v.z, v.w);
}

/* weights 8g..8g+7 of a row as half, into 16 bytes of shared memory */
template<unsigned TYPE>
__device__ __forceinline__ void half8(__half *dst, const char *row, unsigned g,
        const uint64_t *grid_table, const uint8_t *sign_table) {
    float w[8];
    if (grouped(TYPE)) group8<TYPE>(row, g, w);
    else {
        const float4 lo = value4<TYPE>(row, 8 * g, grid_table, sign_table), hi = value4<TYPE>(row, 8 * g + 4, grid_table, sign_table);
        w[0] = lo.x; w[1] = lo.y; w[2] = lo.z; w[3] = lo.w; w[4] = hi.x; w[5] = hi.y; w[6] = hi.z; w[7] = hi.w;
    }
    __half2 h[4];
    #pragma unroll
    for (unsigned k = 0; k < 4; k++) h[k] = __floats2half2_rn(w[2 * k], w[2 * k + 1]);
    *(uint4 *)dst = *(const uint4 *)h;
}

/* Volta's expert tiles: the half operands of the half tiles above on the
 * first tensor cores, with FP32 sums. A block takes 64 rows of one expert by
 * 64 of its tokens, decoding each weight once, and the activations arrive as
 * half. A warp owns 32 rows by 16 tokens and skips its products when those
 * tokens lie past the expert's count, so a part-filled tile costs tensor
 * time in 16-token steps. */
template<unsigned TYPE, bool DOWN>
__global__ void __launch_bounds__(256) expert_tile_h(float *out, const __half *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb) {
#if __CUDA_ARCH__ >= 700
    namespace wm = nvcuda::wmma;
    constexpr unsigned NR = 64, NT = 64, KC = 64, LD = KC + 8;
    const unsigned tid = threadIdx.x, warp = tid / 32, nr = M / NR, job = blockIdx.x / nr;
    if (job >= tiles[NE]) return;
    unsigned e = 0, end = NE;
    while (e < end) {
        const unsigned mid = (e + end) / 2;
        if (tiles[mid + 1] <= job) e = mid + 1; else end = mid;
    }
    const unsigned count = counts[e], t0 = (job - tiles[e]) * NT, r0 = (blockIdx.x % nr) * NR;
    union Tile {
        __half ab[(NR * (DOWN ? 1 : 2) + NT) * LD];
        float c[NT][NR + 4];
    };
    __shared__ __align__(32) Tile tile;
    __shared__ unsigned xrow[NT];
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    __half (*a)[LD] = (__half (*)[LD])tile.ab, (*u)[LD] = a + NR, (*b)[LD] = a + NR * (DOWN ? 1 : 2);
    if (TYPE == 16) {
        for (unsigned i = tid; i < 256; i += 256) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = tid; i < 128; i += 256) sign_table[i] = cuda_ksigns_iq2xs[i];
    }
    /* the gate and up tiles read the token's row, the down tile its slot's */
    if (tid < NT) {
        const unsigned item = t0 + tid, pair = item < count ? lists[(uint64_t)e * cap + item] : 0;
        xrow[tid] = item < count ? (DOWN ? (pair / NS) * NO + pair % NS : pair / NS) : UINT_MAX;
    }
    __syncthreads();
    const unsigned g = tid % 8, wr = tid / 8, tc = warp % 4, rh = warp / 4;
    const char *ra[2], *ru[2];
    #pragma unroll
    for (unsigned j = 0; j < 2; j++) {
        ra[j] = w0 + ((uint64_t)e * M + r0 + wr + 32 * j) * rb;
        ru[j] = DOWN ? NULL : w1 + ((uint64_t)e * M + r0 + wr + 32 * j) * rb;
    }
    const bool busy = t0 + tc * 16 < count;
    wm::fragment<wm::accumulator,16,16,16,float> acc[2], up[2];
    #pragma unroll
    for (unsigned j = 0; j < 2; j++) {
        wm::fill_fragment(acc[j], 0);
        wm::fill_fragment(up[j], 0);
    }
    for (unsigned k0 = 0; k0 < K; k0 += KC) {
        #pragma unroll
        for (unsigned j = 0; j < 2; j++) {
            half8<TYPE>(&a[wr + 32 * j][8 * g], ra[j], k0 / 8 + g, grid_table, sign_table);
            if (!DOWN) half8<TYPE>(&u[wr + 32 * j][8 * g], ru[j], k0 / 8 + g, grid_table, sign_table);
        }
        #pragma unroll
        for (unsigned i = tid; i < NT * KC / 8; i += 256) {
            const unsigned t = i / 8, q = i % 8;
            *(uint4 *)&b[t][8 * q] = xrow[t] != UINT_MAX ?
                *(const uint4 *)(x + (uint64_t)xrow[t] * K + k0 + 8 * q) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (busy) {
            #pragma unroll
            for (unsigned k = 0; k < KC; k += 16) {
                wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major> bf;
                wm::load_matrix_sync(bf, &b[tc * 16][k], LD);
                #pragma unroll
                for (unsigned j = 0; j < 2; j++) {
                    wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> af;
                    wm::load_matrix_sync(af, &a[rh * 32 + j * 16][k], LD);
                    wm::mma_sync(acc[j], af, bf, acc[j]);
                    if (!DOWN) {
                        wm::load_matrix_sync(af, &u[rh * 32 + j * 16][k], LD);
                        wm::mma_sync(up[j], af, bf, up[j]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned j = 0; j < 2; j++) {
        if (!DOWN) for (unsigned i = 0; i < acc[j].num_elements; i++)
            acc[j].x[i] = silu(acc[j].x[i]) * up[j].x[i];
        wm::store_matrix_sync(&tile.c[tc * 16][rh * 32 + j * 16], acc[j], NR + 4, wm::mem_col_major);
    }
    __syncthreads();
    for (unsigned i = tid; i < NT * NR; i += 256) {
        const unsigned t = i / NR, r = i % NR;
        if (xrow[t] == UINT_MAX) continue;
        const unsigned pair = lists[(uint64_t)e * cap + t0 + t];
        out[((uint64_t)(pair / NS) * NO + pair % NS) * M + r0 + r] = tile.c[t][r];
    }
#endif
}

static bool volta(void) {
    static int major = -1;
    if (major < 0) {
        int device = 0;
        cudaDeviceProp prop;
        major = cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess ? prop.major : 0;
        (void)cudaGetLastError();
    }
    return major == 7 && !getenv("DS4_QWEN4_NO_VOLTA_TILES");
}

/* the types half8 decodes: group8's, and value4's IQ2_XXS, Q2_K and MXFP4 */
static bool volta_tiles(unsigned type, unsigned K, unsigned M) {
    return volta() && !g_quality_mode && !getenv("DS4_CUDA_NO_TF32") && !(K % 64) && !(M % 64) &&
        (grouped(type) || type == 10 || type == 16 || type == 39);
}

static int matrix_dispatch(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, unsigned type, unsigned NE, unsigned T,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, bool down) {
    const dim3 grid((M + 15) / 16, lists ? NE : 1, std::min(8u, (T + 15) / 16));
    const uint64_t rb = expert_row_bytes(type, K);
    if (lists && volta_tiles(type, K, M)) {
        const uint64_t rows = down ? (uint64_t)T * NO : T, n = rows * K;
        const uint64_t jobs = ((uint64_t)T * NS + 63) / 64 + NE, blocks = jobs * (M / 64);
        if (blocks > INT_MAX) return 0;
        unsigned *tiles = (unsigned *)tmp_alloc(n * 2 + ((uint64_t)NE + 1) * 4, "Qwen expert half tiles");
        if (!tiles) return 0;
        __half *xh = (__half *)(tiles + (NE + 1 + 3) / 4 * 4);
        expert_tiles<<<1,1,0,cuda_decode_stream()>>>(tiles, counts, NE, 64);
        half_copy<<<(unsigned)((n / 4 + 255) / 256),256,0,cuda_decode_stream()>>>(xh, x, n / 4);
        if (!launched()) return 0;
#define QWEN_VOLTA(TYPE) case TYPE: \
        if (down) expert_tile_h<TYPE,true><<<(unsigned)blocks,256,0,cuda_decode_stream()>>>(out,xh,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
        else expert_tile_h<TYPE,false><<<(unsigned)blocks,256,0,cuda_decode_stream()>>>(out,xh,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
        break;
        switch (type) {
        QWEN_GROUPED_TYPES(QWEN_VOLTA) QWEN_VOLTA(10) QWEN_VOLTA(16) QWEN_VOLTA(39)
        default: return 0;
        }
#undef QWEN_VOLTA
        return launched();
    }
    if (ds4_cuda_attn_tokentile_arch_ok() && !getenv("DS4_CUDA_NO_TF32")) {
        if (lists && !g_quality_mode && (type == 16 || type == 10 || type == 12 || type == 39)) {
            const unsigned nt = T >= 2048 ? 64 : 32;
            unsigned *tiles = (unsigned *)tmp_alloc(((uint64_t)NE+1)*4,"Qwen expert tiles");
            if (!tiles) return 0;
            expert_tiles<<<1,1,0,cuda_decode_stream()>>>(tiles,counts,NE,nt);
            if (!launched()) return 0;
            /* Wider rows reuse activations; Q2_K down is faster at 64 rows. */
            const unsigned nr = nt == 64 && type != 10 ? 128 : 64;
            const uint64_t blocks = (((uint64_t)T*NS+nt-1)/nt+NE)*((M+nr-1)/nr);
            if (blocks > INT_MAX) return 0;
#define QWEN_HALF(TYPE, DOWN) \
            if (nt == 64) matrix_half_tile<TYPE,DOWN,64,(TYPE == 10 ? 64 : 128)><<<blocks,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
            else matrix_half_tile<TYPE,DOWN,32><<<blocks,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb)
#define QWEN_HALF_TYPE(TYPE) case TYPE: if (down) { QWEN_HALF(TYPE,true); } else { QWEN_HALF(TYPE,false); } break
            switch (type) { QWEN_HALF_TYPE(16); QWEN_HALF_TYPE(10); QWEN_HALF_TYPE(12); QWEN_HALF_TYPE(39); }
#undef QWEN_HALF_TYPE
#undef QWEN_HALF
            return launched();
        }
        unsigned *tiles = NULL;
        dim3 tcgrid((M+63)/64,1,std::min(8u,(T+15)/16));
        if (lists) {
            tiles = (unsigned *)tmp_alloc(((uint64_t)NE+1)*4,"Qwen expert tile schedule");
            if (!tiles) return 0;
            expert_tiles<<<1,1,0,cuda_decode_stream()>>>(tiles,counts,NE,32);
            if (!launched()) return 0;
            const uint64_t jobs = ((uint64_t)T*NS+31)/32+NE;
            const uint64_t blocks = jobs*((M+31)/32);
            if (blocks > INT_MAX) return 0;
            tcgrid = dim3((unsigned)blocks,1,1);
        }
#define QWEN_TC(TYPE) case TYPE: \
        if (!lists) matrix_tc<TYPE><<<tcgrid,128,0,cuda_decode_stream()>>>(out,x,w0,T,K,M,rb); \
        else if (down) matrix_reg<TYPE,true,2><<<tcgrid,128,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
        else matrix_reg<TYPE,false,2><<<tcgrid,128,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); break
        switch (type) {
        QWEN_EACH_TYPE(QWEN_TC);
        default: return 0;
        }
#undef QWEN_TC
        return launched();
    }
#define QWEN_MM(TYPE) case TYPE: \
    if (!lists) matrix<TYPE,false,false><<<grid,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); \
    else if (down) matrix<TYPE,true,true><<<grid,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); \
    else matrix<TYPE,false,true><<<grid,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); break
    switch (type) {
    QWEN_EACH_TYPE(QWEN_MM);
    default: return 0;
    }
#undef QWEN_MM
    return launched();
}

template<unsigned TYPE>
__device__ __forceinline__ float dot(const char *row, const float *x, unsigned n,
        const uint64_t *grid, const uint8_t *signs) {
    float acc = 0;
    if (groups_fit<TYPE>(row,x,n)) {
        for (unsigned g = threadIdx.x&31; g < n/8; g += 32) {
            float w[8];
            group8<TYPE>(row,g,w);
            const float4 a = *(const float4 *)(x+8*g), b = *(const float4 *)(x+8*g+4);
            acc += w[0]*a.x; acc += w[1]*a.y; acc += w[2]*a.z; acc += w[3]*a.w;
            acc += w[4]*b.x; acc += w[5]*b.y; acc += w[6]*b.z; acc += w[7]*b.w;
        }
    } else if (TYPE == 1 && !(n%4) && !((uintptr_t)x&15) && !((uintptr_t)row&7)) {
        for (unsigned i = (threadIdx.x&31)*4; i < n; i += 128) {
            const float4 w = value4<TYPE>(row,i,grid,signs), v = *(const float4 *)(x+i);
            acc += w.x*v.x; acc += w.y*v.y; acc += w.z*v.z; acc += w.w*v.w;
        }
    } else {
        for (unsigned i = threadIdx.x & 31; i < n; i += 32) acc += value<TYPE>(row, i, grid, signs) * x[i];
    }
    return sum(acc);
}

template<unsigned TYPE>
__global__ void matvec(float *out, const char *w, const float *x,
                       unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, tok = blockIdx.y;
    if (row >= M) return;
    const float v = dot<TYPE>(w + row * stride, x + (uint64_t)tok * K, K);
    if (!(threadIdx.x & 31)) out[(uint64_t)tok * M + row] = v;
}

template<unsigned TYPE, unsigned ROWS>
__global__ void matvec_rows(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x&31;
    if (row >= M) return;
    float a[ROWS] = {};
    const char *wr = w+(uint64_t)row*stride;
    if (groups_fit<TYPE>(wr,x,K)) {
        for (unsigned g = lane; g < K/8; g += 32) {
            float v[8];
            group8<TYPE>(wr,g,v);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float4 x0 = *(const float4 *)(x+(uint64_t)t*K+8*g), x1 = *(const float4 *)(x+(uint64_t)t*K+8*g+4);
                a[t] += v[0]*x0.x; a[t] += v[1]*x0.y; a[t] += v[2]*x0.z; a[t] += v[3]*x0.w;
                a[t] += v[4]*x1.x; a[t] += v[5]*x1.y; a[t] += v[6]*x1.z; a[t] += v[7]*x1.w;
            }
        }
    } else if (TYPE == 1 && !(K%4) && !((uintptr_t)x&15) && !((uintptr_t)wr&7)) {
        for (unsigned i = lane*4; i < K; i += 128) {
            const float4 v = value4<TYPE>(wr,i,NULL,NULL);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float4 xv = *(const float4 *)(x+(uint64_t)t*K+i);
                a[t] += v.x*xv.x; a[t] += v.y*xv.y; a[t] += v.z*xv.z; a[t] += v.w*xv.w;
            }
        }
    } else {
        for (unsigned i = lane; i < K; i += 32) {
            const float v = value<TYPE>(wr,i);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) a[t] += v*x[(uint64_t)t*K+i];
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(a[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

template<unsigned ROWS>
__global__ void matvec_q8(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x&31;
    if (row >= M) return;
    const char *wr = w+(uint64_t)row*stride;
    float acc[ROWS] = {};
    for (unsigned i = lane*4; i < K; i += 128) {
        const char *b = wr+(i/32)*34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b+2+i%32);
        const unsigned q01 = q[0], q23 = q[1];
        #pragma unroll
        for (unsigned t = 0; t < ROWS; t++) if (t < T) {
            const float4 v = *(const float4 *)(x+(uint64_t)t*K+i);
            float part = (float)(int8_t)q01*v.x;
            part += (float)(int8_t)(q01>>8)*v.y;
            part += (float)(int8_t)q23*v.z;
            part += (float)(int8_t)(q23>>8)*v.w;
            acc[t] += part*scale;
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(acc[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

/* Decode and verify rows (T <= ROWS) with each weight row shared by W
 * warps (8/W rows per block), each warp a contiguous W-th of the row's
 * groups, summed in a fixed order.  W depends on the matrix alone, so a
 * token's sums do not depend on how many rows the batch has: an MTP verify
 * row gets the bits of plain decode.  Short wide matrices (the hc down
 * projections' 320 rows of 10240, the router's 512, the GDN gates' 48)
 * otherwise give each SM a few warps that wait on memory one row at a time. */
template<unsigned TYPE, unsigned W, unsigned ROWS>
__global__ void matvec_split(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    __shared__ float part[8][ROWS];
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const unsigned row = blockIdx.x * (8 / W) + warp / W, piece = warp % W;
    const unsigned G = K / 8, g0 = G * piece / W, g1 = G * (piece + 1) / W;
    float acc[ROWS] = {};
    if (row < M) {
        const char *wr = w + (uint64_t)row * stride;
        for (unsigned g = g0 + lane; g < g1; g += 32) {
            float v[8];
            group8<TYPE>(wr, g, v);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float4 a = *(const float4 *)(x+(uint64_t)t*K+8*g), b = *(const float4 *)(x+(uint64_t)t*K+8*g+4);
                acc[t] += v[0]*a.x; acc[t] += v[1]*a.y; acc[t] += v[2]*a.z; acc[t] += v[3]*a.w;
                acc[t] += v[4]*b.x; acc[t] += v[5]*b.y; acc[t] += v[6]*b.z; acc[t] += v[7]*b.w;
            }
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) {
        const float v = sum(acc[t]);
        if (!lane) part[warp][t] = v;
    }
    __syncthreads();
    if (row < M && !piece && lane < T) {
        float v = 0;
        for (unsigned p = 0; p < W; p++) v += part[warp + p][lane];
        out[(uint64_t)lane * M + row] = v;
    }
}

/* matvec_split with int8 activations.  With a warp per row (W = 1) the
 * types with a 32-weight decoder take a sub-block of 32 weights per lane and
 * step (CH = 4 groups, group32i), which keeps four times the bytes in flight
 * and loads the sub-block's fields once; their decode is otherwise the
 * step's latency.  CH depends on the matrix alone. */
template<unsigned TYPE, unsigned W, unsigned ROWS, unsigned CH>
__global__ void matvec_split_q8(float *out, const char *w, const int8_t *xq, const float *xd,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    __shared__ float part[8][ROWS];
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const unsigned row = blockIdx.x * (8 / W) + warp / W, piece = warp % W;
    const unsigned C = K / (8 * CH), c0 = C * piece / W, c1 = C * (piece + 1) / W;
    float acc[ROWS] = {};
    if (row < M) {
        const char *wr = w + (uint64_t)row * stride;
        for (unsigned c = c0 + lane; c < c1; c += 32) {
            if constexpr (CH == 4 && grouped_i32(TYPE)) {
                int q[8];
                float d0, d1, mm;
                group32i<TYPE>(wr, c, q, d0, d1, mm);
                #pragma unroll
                for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                    const int8_t *xt = xq + (uint64_t)t * K + 32 * c;
                    const int4 xa = *(const int4 *)xt, xb = *(const int4 *)(xt + 16);
                    const int p0 = __dp4a(q[0], xa.x, __dp4a(q[1], xa.y, __dp4a(q[2], xa.z, __dp4a(q[3], xa.w, 0))));
                    const int p1 = __dp4a(q[4], xb.x, __dp4a(q[5], xb.y, __dp4a(q[6], xb.z, __dp4a(q[7], xb.w, 0))));
                    const int sx = TYPE == 12 || TYPE == 13 ? sum32(xa, xb) : 0;
                    acc[t] = __fmaf_rn(xd[(uint64_t)t * (K / 32) + c], q8_sub32<TYPE>(d0, p0, d1, p1, mm, sx), acc[t]);
                }
            } else {
                int w0[CH], w1[CH];
                float d[CH], m[CH];
                #pragma unroll
                for (unsigned o = 0; o < CH; o++) group8i<TYPE>(wr, CH * c + o, w0[o], w1[o], d[o], m[o]);
                #pragma unroll
                for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                    const int8_t *xt = xq + (uint64_t)t * K + 8 * CH * c;
                    const float s = xd[((uint64_t)t * K + 8 * CH * c) / 32];
                    float a = 0;
                    #pragma unroll
                    for (unsigned o = 0; o < CH; o++) {
                        const int2 x = *(const int2 *)(xt + 8 * o);
                        const int sx = TYPE == 12 || TYPE == 13 ? __dp4a(0x01010101, x.x, __dp4a(0x01010101, x.y, 0)) : 0;
                        a = __fadd_rn(a, q8_term<TYPE>(d[o], __dp4a(w0[o], x.x, __dp4a(w1[o], x.y, 0)), m[o], sx));
                    }
                    acc[t] = __fmaf_rn(s, a, acc[t]);
                }
            }
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) {
        const float v = sum(acc[t]);
        if (!lane) part[warp][t] = v;
    }
    __syncthreads();
    if (row < M && !piece && lane < T) {
        float v = 0;
        for (unsigned p = 0; p < W; p++) v += part[warp + p][lane];
        out[(uint64_t)lane * M + row] = v;
    }
}

static int matvec_dispatch(float *out, const char *w, const float *x,
                           unsigned type, unsigned T, unsigned K, unsigned M) {
    const dim3 grid((M + 3) / 4, T);
    const uint64_t stride = row_bytes(type, K);
    if (!stride) return 0;
    /* about 2560 warps in flight over the V100's 80 SMs, and at least four
     * per row of 4096 weights or more: their warps otherwise wait on memory
     * through 16 and more groups one after the other */
    unsigned W = M >= 2560 ? 1 : M >= 1280 ? 2 : M >= 640 ? 4 : 8;
    if (K >= 4096 && W < 4) W = 4;
    if (T <= 8 && act_q8() && grouped_i8(type) && !(K % 32)) {
        const int8_t *xq;
        const float *xd;
        if (!q8_rows(x, T, K, &xq, &xd)) return 0;
#define QWEN_SPLIT_Q8_ROWS(TYPE, R) \
        if (W == 1) { \
            if (grouped_i32(TYPE)) matvec_split_q8<TYPE,1,R,4><<<(M+7)/8,256,0,cuda_decode_stream()>>>(out,w,xq,xd,T,K,M,stride); \
            else matvec_split_q8<TYPE,1,R,1><<<(M+7)/8,256,0,cuda_decode_stream()>>>(out,w,xq,xd,T,K,M,stride); \
        } \
        else if (W == 2) matvec_split_q8<TYPE,2,R,1><<<(M+3)/4,256,0,cuda_decode_stream()>>>(out,w,xq,xd,T,K,M,stride); \
        else if (W == 4) matvec_split_q8<TYPE,4,R,1><<<(M+1)/2,256,0,cuda_decode_stream()>>>(out,w,xq,xd,T,K,M,stride); \
        else matvec_split_q8<TYPE,8,R,1><<<M,256,0,cuda_decode_stream()>>>(out,w,xq,xd,T,K,M,stride)
#define QWEN_SPLIT_Q8(TYPE) case TYPE: \
        if (T == 1) { QWEN_SPLIT_Q8_ROWS(TYPE,1); } \
        else if (T <= 4) { QWEN_SPLIT_Q8_ROWS(TYPE,4); } \
        else { QWEN_SPLIT_Q8_ROWS(TYPE,8); } \
        return launched();
        switch (type) {
        QWEN_SPLIT_Q8(6) QWEN_SPLIT_Q8(8) QWEN_SPLIT_Q8(11) QWEN_SPLIT_Q8(12) QWEN_SPLIT_Q8(13) QWEN_SPLIT_Q8(14)
        QWEN_SPLIT_Q8(16) QWEN_SPLIT_Q8(17) QWEN_SPLIT_Q8(18) QWEN_SPLIT_Q8(20) QWEN_SPLIT_Q8(21) QWEN_SPLIT_Q8(22)
        QWEN_SPLIT_Q8(23) QWEN_SPLIT_Q8(42)
        default: break;
        }
#undef QWEN_SPLIT_Q8
#undef QWEN_SPLIT_Q8_ROWS
    }
    if (T <= 8 && grouped(type) && !(K % 8) && !((uintptr_t)x & 15) &&
        ((type != 1 && type != 30) || (!(stride % 16) && !((uintptr_t)w & 15)))) {
#define QWEN_SPLIT_ROWS(TYPE, R) \
        if (W == 1) matvec_split<TYPE,1,R><<<(M+7)/8,256,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
        else if (W == 2) matvec_split<TYPE,2,R><<<(M+3)/4,256,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
        else if (W == 4) matvec_split<TYPE,4,R><<<(M+1)/2,256,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
        else matvec_split<TYPE,8,R><<<M,256,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride)
#define QWEN_SPLIT(TYPE) case TYPE: \
        if (T == 1) { QWEN_SPLIT_ROWS(TYPE,1); } \
        else if (T <= 4) { QWEN_SPLIT_ROWS(TYPE,4); } \
        else { QWEN_SPLIT_ROWS(TYPE,8); } \
        return launched();
        switch (type) {
        QWEN_GROUPED_TYPES(QWEN_SPLIT)
        default: break;
        }
#undef QWEN_SPLIT
#undef QWEN_SPLIT_ROWS
    }
    if (type == 8 && T <= 8 && !((uintptr_t)x&15)) {
#define QWEN_Q8_ROWS(N) matvec_q8<N><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride)
        if (T == 1) { QWEN_Q8_ROWS(1); }
        else if (T == 2) { QWEN_Q8_ROWS(2); }
        else if (T <= 4) { QWEN_Q8_ROWS(4); }
        else { QWEN_Q8_ROWS(8); }
#undef QWEN_Q8_ROWS
        return launched();
    }
#define QWEN_MV(TYPE) case TYPE: \
    if (T == 2) matvec_rows<TYPE,2><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
    else if (T > 2 && T <= 4) matvec_rows<TYPE,4><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
    else if (T > 4 && T <= 8) matvec_rows<TYPE,8><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
    else matvec<TYPE><<<grid,128,0,cuda_decode_stream()>>>(out,w,x,K,M,stride); break
    switch (type) {
    QWEN_EACH_TYPE(QWEN_MV);
    default: return 0;
    }
#undef QWEN_MV
    return launched();
}

template<unsigned TYPE>
__global__ void unpack(float *out, const char *w, unsigned K, unsigned M, uint64_t rb) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (uint64_t)K*M) out[i] = value<TYPE>(w+(i/K)*rb,i%K);
}

/* Power-of-two row scaling protects half range. Preserve the residual too,
 * so dense projections retain substantially more than half precision. */
template<unsigned TYPE>
__global__ void pack_half_components(float *scales, __half *hi, __half *lo,
        const char *x, unsigned K, uint64_t rb) {
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    __shared__ float maxima[256], inv;
    float mx = 0;
    for (unsigned k = tid; k < K; k += 256) mx = fmaxf(mx,fabsf(value<TYPE>(x+(uint64_t)row*rb,k)));
    maxima[tid] = mx;
    __syncthreads();
    for (unsigned d = 128; d; d /= 2) {
        if (tid < d) maxima[tid] = fmaxf(maxima[tid],maxima[tid+d]);
        __syncthreads();
    }
    if (!tid) {
        const int e = maxima[0] > 0 ? max(-120,min(120,(int)((__float_as_uint(maxima[0])>>23)&255)-127)) : 0;
        inv = ldexpf(1,-e); scales[row] = ldexpf(1,e);
    }
    __syncthreads();
    for (unsigned k = tid; k < K; k += 256) {
        const float v = value<TYPE>(x+(uint64_t)row*rb,k)*inv;
        const __half h = __float2half_rn(v);
        hi[(uint64_t)row*K+k] = h;
        if (lo) lo[(uint64_t)row*K+k] = __float2half_rn((v-__half2float(h))*4096);
    }
}

__global__ void dense_rescale(float *out, const float *scales,
        unsigned M, unsigned N) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*N) out[i] *= scales[i/M];
}

/* Short projections reuse the weights for both activation components.
 * Keep separate FP32 sums and the same scaled residual as the cuBLAS path. */
__global__ void dense_f16_components(float *out, const __half *xh, const __half *xl,
        const __half *w, const float *scales, unsigned T, unsigned K, unsigned M) {
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32;
    const unsigned m0 = blockIdx.x*64, t0 = blockIdx.y*64;
    union Tile { __half data[3][64][72]; float result[64][64]; };
    __shared__ __align__(32) Tile tile;
    wm::fragment<wm::accumulator,16,16,16,float> hi[2], lo[2];
    for (unsigned j = 0; j < 2; j++) {
        wm::fill_fragment(hi[j],0);
        wm::fill_fragment(lo[j],0);
    }
    for (unsigned k0 = 0; k0 < K; k0 += 64) {
        for (unsigned i = tid*8; i < 4096; i += 2048) {
            const unsigned r = i/64, k = i%64;
            const bool valid_w = m0+r < M, valid_x = t0+r < T;
            tt_cp_async_16B(&tile.data[0][r][k],w+((uint64_t)(valid_w ? m0+r : 0))*K+k0+k,valid_w);
            tt_cp_async_16B(&tile.data[1][r][k],xh+((uint64_t)(valid_x ? t0+r : 0))*K+k0+k,valid_x);
            tt_cp_async_16B(&tile.data[2][r][k],xl+((uint64_t)(valid_x ? t0+r : 0))*K+k0+k,valid_x);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        for (unsigned k = 0; k < 64; k += 16) {
            wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> a;
            wm::load_matrix_sync(a,&tile.data[0][(warp%4)*16][k],72);
            for (unsigned j = 0; j < 2; j++) {
                wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major> b, c;
                wm::load_matrix_sync(b,&tile.data[1][(warp/4)*32+j*16][k],72);
                wm::load_matrix_sync(c,&tile.data[2][(warp/4)*32+j*16][k],72);
                wm::mma_sync(hi[j],a,b,hi[j]);
                wm::mma_sync(lo[j],a,c,lo[j]);
            }
        }
        __syncthreads();
    }
    for (unsigned j = 0; j < 2; j++) {
        for (unsigned i = 0; i < hi[j].num_elements; i++) hi[j].x[i] += lo[j].x[i]*0x1p-12f;
        wm::store_matrix_sync(&tile.result[(warp%4)*16][(warp/4)*32+j*16],hi[j],64,wm::mem_row_major);
    }
    __syncthreads();
    for (unsigned i = tid; i < 4096; i += 256) {
        const unsigned m = m0+i%64, t = t0+i/64;
        if (m < M && t < T) out[(uint64_t)t*M+m] = tile.result[i%64][i/64]*scales[t];
    }
#endif
}

static int dense_f16_blas(float *out, const float *x, const __half *w,
        unsigned T, unsigned K, unsigned M) {
    const uint64_t xn = (uint64_t)T*K;
    __half *hi = (__half *)tmp_alloc((xn+T)*4, "Qwen F16 projection scratch");
    if (!hi) return 0;
    __half *lo = hi+xn;
    float *scales = (float *)(lo+xn);
    pack_half_components<0><<<T,256,0,cuda_decode_stream()>>>(scales,hi,lo,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    if (K <= 512 && K%64 == 0 && T >= 32 && ds4_cuda_attn_tokentile_arch_ok()) {
        dense_f16_components<<<dim3((M+63)/64,(T+63)/64),256,0,cuda_decode_stream()>>>(out,hi,lo,w,scales,T,K,M);
        return launched();
    }
    const float zero = 0, one = 1, low = 0x1p-12f;
    if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                M,T,K,&low,w,CUDA_R_16F,K,lo,CUDA_R_16F,K,&zero,out,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen F16 residual projection") ||
        !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                M,T,K,&one,w,CUDA_R_16F,K,hi,CUDA_R_16F,K,&one,out,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen F16 leading projection")) return 0;
    dense_rescale<<<((uint64_t)M*T+255)/256,256,0,cuda_decode_stream()>>>(out,scales,M,T);
    return launched();
}

__global__ void dense_rescale2(float *out, const float *xs, const float *ws,
        unsigned M, unsigned T, unsigned stride) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*T) {
        const uint64_t dst = (i/M)*stride+i%M;
        out[dst] = out[dst]*xs[i/M]*ws[i%M];
    }
}

/* Any weight type on the tensor cores: the weight rows are packed to half
 * per tile, each weight and activation row under a power-of-two scale that
 * keeps it in half's range, and one GEMM sums in FP32.  Half rounds both
 * operands by at most 2^-11, under the int8 activations of the decode
 * path; DS4_CUDA_NO_TF32 and the quality mode keep FP32 GEMMs. */
static int dense_half_blas(float *out, const float *x, const char *w, unsigned type,
        unsigned T, unsigned K, unsigned M) {
    const unsigned bound = (unsigned)std::max(UINT64_C(1),(UINT64_C(64)<<20)/((uint64_t)K*2));
    const unsigned tile = std::min(M,bound >= 64 ? bound/64*64 : bound);
    const uint64_t xn = (uint64_t)T*K, wn = (uint64_t)tile*K;
    __half *xh = (__half *)tmp_alloc((xn+wn)*2+((uint64_t)T+tile)*4,"Qwen half projection scratch");
    if (!xh) return 0;
    __half *wh = xh+xn;
    float *xs = (float *)(wh+wn), *ws = xs+T;
    pack_half_components<0><<<T,256,0,cuda_decode_stream()>>>(xs,xh,NULL,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    const float zero = 0, one = 1;
    const uint64_t rb = row_bytes(type,K);
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
#define QWEN_PACK(TYPE) case TYPE: pack_half_components<TYPE><<<n,256,0,cuda_decode_stream()>>>(ws,wh,NULL,w+(uint64_t)r*rb,K,rb); break
        switch (type) {
        QWEN_EACH_TYPE(QWEN_PACK);
        default: return 0;
        }
#undef QWEN_PACK
        if (!launched() ||
            !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&one,wh,CUDA_R_16F,K,xh,CUDA_R_16F,K,&zero,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen half projection")) return 0;
        dense_rescale2<<<((uint64_t)n*T+255)/256,256,0,cuda_decode_stream()>>>(out+r,xs,ws,n,T,M);
        if (!launched()) return 0;
    }
    return 1;
}

/* Bound the temporary weight expansion instead of retaining an FP32 copy
 * of each projection. cuBLAS uses FP32 math, including FP32 activations. */
static int dense_blas(float *out, const float *x, const char *w,
        unsigned type, unsigned T, unsigned K, unsigned M) {
    const unsigned tile = std::min(M, (unsigned)std::max(UINT64_C(1), (UINT64_C(64)*1024*1024)/((uint64_t)K*4)));
    float *scratch = type ? (float *)tmp_alloc((uint64_t)tile*K*4, "Qwen dense FP32 tile") : NULL;
    if (type && !scratch) return 0;
    const uint64_t rb = row_bytes(type,K);
    const float alpha = 1, beta = 0;
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        const float *wf = type ? scratch : (const float *)w+(uint64_t)r*K;
        if (type) {
#define QWEN_UNPACK(TYPE) case TYPE: unpack<TYPE><<<((uint64_t)n*K+255)/256,256,0,cuda_decode_stream()>>>(scratch,w+(uint64_t)r*rb,K,n,rb); break
            switch (type) {
            QWEN_EACH_TYPE(QWEN_UNPACK);
            default: return 0;
            }
#undef QWEN_UNPACK
            if (!launched()) return 0;
        }
        if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&alpha,wf,CUDA_R_32F,K,x,CUDA_R_32F,K,&beta,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT), "Qwen FP32 projection")) return 0;
    }
    return 1;
}

__global__ void mtp_stage(float *cat, const float *e, const float *R,
        const float *ge, const float *gh, unsigned E, unsigned hc, float eps) {
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    const bool emb = row == 0;
    const unsigned n = emb ? E : E * hc;
    const float *rs = emb ? e : R;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned i = tid; i < n; i += blockDim.x) ss += rs[i] * rs[i];
    const float inv = rsqrtf(block_sum(ss, red) / n + eps);
    const float *src = emb ? e : R + (uint64_t)(row - 1) * E;
    const float *g = emb ? ge : gh + (uint64_t)(row - 1) * E;
    float *o = cat + (uint64_t)row * 2 * E;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        o[(emb ? 0 : E) + i] = src[i] * inv * g[i];
        o[(emb ? E : 0) + i] = 0;
    }
}

__global__ void mtp_combine(float *out, const float *proj, unsigned E, unsigned hc) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E * hc) out[i] = proj[i % E] + proj[E + i];
}

/* The argmax and its softmax probability, which the MTP depth policy reads:
 * every partial keeps its maximum and the sum of exp(x - maximum), rescaled
 * when a larger maximum joins.  NaN logits take no part. */
struct max_pair { float value; int index; float sum; };

__device__ __forceinline__ max_pair max_merge(max_pair a, const max_pair b) {
    if ((__float_as_uint(b.value) & 0x7fffffffu) > 0x7f800000u) return a;
    if (b.value > a.value) {
        a.sum = a.sum * expf(a.value - b.value) + b.sum;
        a.value = b.value; a.index = b.index;
    } else {
        a.sum += b.sum * expf(b.value - a.value);
        if (b.value == a.value && b.index < a.index) a.index = b.index;
    }
    return a;
}

template<bool FINISH>
__global__ void argmax(int *out, max_pair *scratch, const float *logits, unsigned n) {
    const unsigned tid = threadIdx.x;
    __shared__ max_pair best[256];
    max_pair v = {-1e30f, 0, 0};
    for (unsigned i = (FINISH ? 0 : blockIdx.x * 4096) + tid;
         i < (FINISH ? n : min(n, (blockIdx.x + 1) * 4096)); i += 256)
        v = max_merge(v, FINISH ? scratch[i] : max_pair{logits[i], (int)i, 1});
    best[tid] = v;
    __syncthreads();
    for (unsigned s = 128; s; s /= 2) {
        if (tid < s) best[tid] = max_merge(best[tid], best[tid + s]);
        __syncthreads();
    }
    if (!tid) {
        if (FINISH) { out[0] = best[0].index; out[1] = __float_as_int(1.0f / best[0].sum); }
        else scratch[blockIdx.x] = best[0];
    }
}

__global__ void vis_patch(float *x, const float *a, const float *b, const float *bias,
        const float *pos, unsigned N, unsigned E) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint64_t)N * E) x[i] = a[i] + b[i] + bias[i % E] + pos[i];
}

__global__ void vis_norm(float *out, const float *x, const float *w, const float *b, unsigned E, float eps) {
    const unsigned tid = threadIdx.x;
    const uint64_t base = (uint64_t)blockIdx.x * E;
    __shared__ float red[32];
    float v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) v += x[base + i];
    const float mean = block_sum(v, red) / E;
    v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { const float d = x[base + i] - mean; v += d*d; }
    const float inv = rsqrtf(block_sum(v, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) out[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];
}

__global__ void vis_qkv(float *q, float *k, float *v, const float *qkv, const float *bias,
        unsigned H, unsigned D, unsigned grid_w) {
    const unsigned row = blockIdx.x, tid = threadIdx.x, E = H*D, hd = D/2, qd = D/4;
    const unsigned blk = row/4, within = row%4, w2 = grid_w/2;
    const float hp = (blk/w2)*2 + within/2, wp = (blk%w2)*2 + within%2;
    const float *src = qkv + (uint64_t)row*3*E;
    const uint64_t off = (uint64_t)row*E;
    for (unsigned idx = tid; idx < 2*H*hd; idx += blockDim.x) {
        const unsigned part = idx/(H*hd), rem = idx%(H*hd), h = rem/hd, i = rem%hd;
        const unsigned base = part*E + h*D;
        const float x0 = src[base+i] + bias[base+i], x1 = src[base+i+hd] + bias[base+i+hd];
        const unsigned j = i < qd ? i : i-qd;
        const float theta = (i < qd ? hp : wp) * powf(10000.0f, -2.0f*j/hd);
        float s, c; sincosf(theta, &s, &c);
        float *dst = part ? k : q;
        dst[off+h*D+i] = x0*c - x1*s;
        dst[off+h*D+i+hd] = x0*s + x1*c;
    }
    for (unsigned i = tid; i < E; i += blockDim.x) v[off+i] = src[2*E+i] + bias[2*E+i];
}

__global__ void vis_attention(float *out, const float *q, const float *k, const float *v,
        unsigned N, unsigned H, unsigned D) {
    const unsigned row = blockIdx.x, h = blockIdx.y, lane = threadIdx.x, E = H*D;
    const uint64_t qb = (uint64_t)row*E + h*D;
    float query[3] = {}, acc[3] = {};
    for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) query[j] = q[qb+lane+j*32];
    float mx = -INFINITY, denom = 0;
    for (unsigned p = 0; p < N; p++) {
        const uint64_t kb = (uint64_t)p*E + h*D;
        float score = 0;
        for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) score += query[j]*k[kb+lane+j*32];
        score = sum(score) * rsqrtf((float)D);
        const float nm = fmaxf(mx, score), old = expf(mx-nm), prob = expf(score-nm);
        denom = denom*old + prob;
        for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) acc[j] = acc[j]*old + prob*v[kb+lane+j*32];
        mx = nm;
    }
    for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) out[qb+lane+j*32] = acc[j]/denom;
}

__global__ void vis_add(float *x, const float *add, const float *bias, unsigned N, unsigned E, unsigned mode) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (uint64_t)N*E) return;
    if (add) { x[i] += add[i] + bias[i%E]; return; }
    float t = x[i] + bias[i%E];
    if (!mode) t *= .5f * (1 + tanhf(fminf(30, fmaxf(-30, .7978845608f*(t+.044715f*t*t*t)))));
    else if (mode == 1) t *= .5f * (1 + erff(t*.70710678f));
    x[i] = t;
}

/* Decode-sized hyper-connection norm (T <= 8) and its inject partials, the
 * inject rows' type a template argument so a thread's loads issue together
 * (read one by one through scalar(), the cold rows cost 13 us a call).  With
 * COMBINE it first adds the block output weighted by the previous injection
 * to R and writes the sum to next: hc_combine and the norm in one pass.
 * With xq it also rounds xn to int8 for the down projection (E a multiple
 * of 256, so a warp's values are whole groups of 32). */
template<unsigned TYPE, bool COMBINE>
__global__ void hc_norm_rows(float *xn, float *inj, float *next, const float *R, const float *blk,
                             const float *oldinj, const float *gamma, const char *wi, unsigned E,
                             unsigned hc, unsigned ni, float eps, int8_t *xq, float *xd) {
    const unsigned stream = blockIdx.x / 8, chunk = blockIdx.x % 8, tok = blockIdx.y;
    const unsigned tid = threadIdx.x, dim = E * hc;
    const uint64_t base = ((uint64_t)tok * hc + stream) * E;
    const float *b = COMBINE ? blk + (uint64_t)tok * E : NULL;
    const float w = COMBINE ? injection(oldinj + (uint64_t)tok * hc * hc * 8, hc, stream) : 0;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        const float r = COMBINE ? R[base + i] + w * b[i] : R[base + i];
        ss += r * r;
    }
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    float acc[4] = {};
    const unsigned per = (E + 7) / 8, end = min(E, (chunk + 1) * per);
    for (unsigned i = chunk * per + tid; i < end; i += blockDim.x) {
        const float r = COMBINE ? R[base + i] + w * b[i] : R[base + i];
        if (COMBINE) next[base + i] = r;
        const float v = r * inv * gamma[stream * E + i];
        xn[base + i] = v;
        if (xq) {   /* a warp's 32 values are a group: quantize_q8's rounding */
            float a = fabsf(v);
            for (unsigned o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
            const float sc = __fdiv_rn(a, 127.f);
            xq[base + i] = (int8_t)__float2int_rn(sc > 0 ? __fdiv_rn(v, sc) : 0.f);
            if (!(tid & 31)) xd[(base + i) / 32] = sc;
        }
        for (unsigned j = 0; j < ni; j++) acc[j] += value<TYPE>(wi, j * dim + stream * E + i) * v;
    }
    for (unsigned j = 0; j < ni; j++) {
        const float v = block_sum(acc[j], red);
        if (!tid) inj[((uint64_t)tok * hc * 8 + stream * 8 + chunk) * ni + j] = v;
    }
}

/* Prefill has enough tokens to share one normalization across all eight
 * injection chunks. Each warp preserves the four partial sums of the
 * decode kernel, including their order, without rereading R eight times. */
template<unsigned TYPE>
__global__ void hc_norm_prefill(float *xn, float *inj, const float *R,
        const float *gamma, const char *wi, unsigned E, unsigned hc,
        unsigned ni, float eps) {
    const unsigned stream = blockIdx.x, tok = blockIdx.y;
    const unsigned tid = threadIdx.x, lane = tid & 31, chunk = tid / 32;
    const unsigned dim = E * hc, per = (E + 7) / 8;
    const uint64_t base = ((uint64_t)tok * hc + stream) * E;
    __shared__ float red[32];
    float ss = 0;
    if (tid < 128) for (unsigned i = tid; i < E; i += 128)
        ss += R[base+i] * R[base+i];
    const float inv = rsqrtf(block_sum(ss,red) / E + eps);
    float acc[4][4] = {};
    const unsigned end = min(E,(chunk+1)*per);
    #pragma unroll
    for (unsigned part = 0; part < 4; part++) {
        for (unsigned i = chunk*per+lane+part*32; i < end; i += 128) {
            const float v = R[base+i] * inv * gamma[stream*E+i];
            xn[base+i] = v;
            #pragma unroll
            for (unsigned j = 0; j < 4; j++) if (j < ni)
                acc[j][part] += value<TYPE>(wi,j*dim+stream*E+i) * v;
        }
    }
    #pragma unroll
    for (unsigned j = 0; j < 4; j++) if (j < ni) {
        float v = 0;
        #pragma unroll
        for (unsigned part = 0; part < 4; part++) v += sum(acc[j][part]);
        if (!lane) inj[((uint64_t)tok*hc*8+stream*8+chunk)*ni+j] = v;
    }
}

template<unsigned TYPE>
__global__ void hc_mix(float *out, const float *xn, const float *lo,
                       const char *up, unsigned E, unsigned hc, unsigned rank, uint64_t rb) {
    const unsigned d = blockIdx.x * 4 + threadIdx.x / 32, tok = blockIdx.y;
    __shared__ __align__(16) float activated[512];
    if (rank <= 512) {
        for (unsigned r = threadIdx.x; r < rank; r += blockDim.x)
            activated[r] = silu(lo[(uint64_t)tok*rank+r]/hc);
        __syncthreads();
    }
    if (d >= E) return;
    const unsigned lane = threadIdx.x & 31, stream = lane / 8, l = lane % 8;
    float a = 0;
    if (stream < hc) {
        const char *row = up+((uint64_t)stream*E+d)*rb;
        if (!(rank%4) && rank <= 512 &&
            !(TYPE == 0 ? (uintptr_t)row&15 : TYPE == 1 ? (uintptr_t)row&7 : 0)) {
            for (unsigned r = l*4; r < rank; r += 32) {
                const float4 w = value4<TYPE>(row,r,NULL,NULL), v = *(const float4 *)(activated+r);
                a += w.x*v.x; a += w.y*v.y; a += w.z*v.z; a += w.w*v.w;
            }
        } else {
            for (unsigned r = l; r < rank; r += 8)
                a += value<TYPE>(row,r) *
                    (rank <= 512 ? activated[r] : silu(lo[(uint64_t)tok * rank + r] / hc));
        }
    }
    for (unsigned off = 1; off <= 4; off *= 2) a += __shfl_xor_sync(0xffffffff, a, off);
    float v = stream < hc ? sigmoid(a) * xn[((uint64_t)tok * hc + stream) * E + d] : 0;
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 16);
    if (!lane) out[(uint64_t)tok * E + d] = v / hc;
}

/* The gate mix of up to ROWS decode or verify rows: a lane decodes its slice
 * of the hc up rows once and applies it to every row's activations, where
 * hc_mix's per-row blocks read the 3.5 MB of up rows again for each row.
 * Each row's sums keep hc_mix's order, so a verify row gets decode's bits. */
template<unsigned TYPE, unsigned ROWS>
__global__ void hc_mix_rows(float *out, const float *xn, const float *lo, const char *up,
                            unsigned T, unsigned E, unsigned hc, unsigned rank, uint64_t rb) {
    const unsigned d = blockIdx.x * 4 + threadIdx.x / 32;
    __shared__ __align__(16) float activated[ROWS][512];
    for (unsigned i = threadIdx.x; i < T * rank; i += blockDim.x) activated[i / rank][i % rank] = silu(lo[i] / hc);
    __syncthreads();
    if (d >= E) return;
    const unsigned lane = threadIdx.x & 31, stream = lane / 8, l = lane % 8;
    float a[ROWS] = {};
    if (stream < hc) {
        const char *row = up + ((uint64_t)stream * E + d) * rb;
        for (unsigned r = l * 4; r < rank; r += 32) {
            const float4 w = value4<TYPE>(row, r, NULL, NULL);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float4 v = *(const float4 *)(activated[t] + r);
                a[t] += w.x*v.x; a[t] += w.y*v.y; a[t] += w.z*v.z; a[t] += w.w*v.w;
            }
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) {
        if (t >= T) break;
        float at = a[t];
        for (unsigned off = 1; off <= 4; off *= 2) at += __shfl_xor_sync(0xffffffff, at, off);
        float v = stream < hc ? sigmoid(at) * xn[((uint64_t)t * hc + stream) * E + d] : 0;
        v += __shfl_xor_sync(0xffffffff, v, 8);
        v += __shfl_xor_sync(0xffffffff, v, 16);
        if (!lane) out[(uint64_t)t * E + d] = v / hc;
    }
}

__device__ float injection(const float *inj, unsigned hc, unsigned stream) {
    float v = 0;
    for (unsigned i = 0; i < hc * 8; i++) v += inj[i * hc + stream];
    return 2 * sigmoid(v / hc);
}

__global__ void hc_combine(float *R, const float *out, const float *inj, unsigned E, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    __shared__ float weights[4];
    if (threadIdx.x < hc) weights[threadIdx.x] = injection(inj + (uint64_t)tok * hc * hc * 8, hc, threadIdx.x);
    __syncthreads();
    if (d >= E) return;
    for (unsigned s = 0; s < hc; s++) R[((uint64_t)tok * hc + s) * E + d] += weights[s] * out[(uint64_t)tok * E + d];
}

__global__ void hc_lo(float *dst, const float *src, uint64_t n, unsigned hc) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = silu(src[i] / hc);
}

__global__ void hc_rows(float *dst, const float *up, const float *xn, unsigned E, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (d >= E) return;
    float v = 0;
    for (unsigned s = 0; s < hc; s++) {
        const uint64_t i = ((uint64_t)t * hc + s) * E + d;
        v += sigmoid(up[i]) * xn[i];
    }
    dst[(uint64_t)t * E + d] = v / hc;
}

} // namespace qwen4_cuda

namespace qwen4_cuda {

__global__ void conv(float *x, float *history, const float *w, unsigned T,
                     unsigned C, unsigned K, bool activate,
                     float *snap, unsigned snap_t, float *snap2, unsigned snap2_t) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float win[3], taps[4];
    for (unsigned i = 0; i < K - 1; i++) win[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = w[c * K + i];
    for (unsigned t = 0; t < T; t++) {
        const uint64_t pos = (uint64_t)t * C + c;
        const float raw = x[pos];
        float v = taps[K - 1] * raw;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * win[i];
        for (unsigned i = 0; i + 2 < K; i++) win[i] = win[i + 1];
        win[K - 2] = raw;
        x[pos] = activate ? silu(v) : v;
        if (snap && t == snap_t) for (unsigned i = 0; i < K - 1; i++) snap[(uint64_t)i * C + c] = win[i];
        if (snap2 && t == snap2_t) for (unsigned i = 0; i < K - 1; i++) snap2[(uint64_t)i * C + c] = win[i];
    }
    for (unsigned i = 0; i < K - 1; i++) history[(uint64_t)i * C + c] = win[i];
}

__global__ void gdn_prep(float *qkv, float *a, float *b, const float *A, const float *bias,
                         unsigned Hk, unsigned Hv, unsigned D) {
    const unsigned h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const unsigned C = (2 * Hk + Hv) * D, npt = D / 32;
    float *q = qkv + (uint64_t)t * C + h * D + lane * npt, *k = q + Hk * D;
    float qs = 0, ks = 0;
    for (unsigned i = 0; i < npt; i++) { qs += q[i] * q[i]; ks += k[i] * k[i]; }
    qs = rsqrtf(sum(qs) + 1e-6f) * rsqrtf((float)D);
    ks = rsqrtf(sum(ks) + 1e-6f);
    for (unsigned i = 0; i < npt; i++) { q[i] *= qs; k[i] *= ks; }
    if (!h) for (unsigned j = lane; j < Hv; j += 32) {
        const uint64_t p = (uint64_t)t * Hv + j;
        a[p] = expf(A[j] * softplus(a[p] + bias[j]));
        b[p] = sigmoid(b[p]);
    }
}

/* One warp owns a state row across the whole chunk. In particular, MTP
 * snapshots contain the state after the requested token, not the final row. */
template<unsigned ROWS, unsigned D>
__global__ void gdn_scan(float *out, float *state, const float *qkv,
                         const float *a, const float *b, unsigned T, unsigned Hk,
                         unsigned Hv, float *snap, unsigned st,
                         float *snap2, unsigned st2) {
    const unsigned dv = (blockIdx.x * 4 + threadIdx.x / 32)*ROWS, h = blockIdx.y;
    if (dv >= D) return;
    const unsigned npt = D / 32, k0 = (threadIdx.x & 31) * npt, kh = h % Hk;
    const unsigned C = (2 * Hk + Hv) * D;
    const uint64_t idx = ((uint64_t)h * D + dv) * D + k0;
    float s[ROWS][4];
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) s[r][i] = state[idx+(uint64_t)r*D+i];
    for (unsigned t = 0; t < T; t++) {
        const float *q = qkv + (uint64_t)t * C + kh * D + k0, *k = q + Hk * D;
        const float decay = a[(uint64_t)t * Hv + h], beta = b[(uint64_t)t * Hv + h];
        #pragma unroll
        for (unsigned r = 0; r < ROWS; r++) {
            const float v = qkv[(uint64_t)t*C+2*Hk*D+h*D+dv+r];
            float u = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] *= decay; u += s[r][i]*k[i]; }
            const float delta = (v-sum(u))*beta;
            float o = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] += k[i]*delta; o += s[r][i]*q[i]; }
            o = sum(o);
            if (!(threadIdx.x&31)) out[((uint64_t)t*Hv+h)*D+dv+r] = o;
            if (snap && t == st) for (unsigned i = 0; i < npt; i++) snap[idx+(uint64_t)r*D+i] = s[r][i];
            if (snap2 && t == st2) for (unsigned i = 0; i < npt; i++) snap2[idx+(uint64_t)r*D+i] = s[r][i];
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) state[idx+(uint64_t)r*D+i] = s[r][i];
}

/* With oq it also rounds the output to int8 for ssm_out, as quantize_q8
 * does: a group of 32 is 32 / npt consecutive lanes (D <= 128). */
__global__ void gdn_out(float *o, const float *z, const float *w, unsigned H, unsigned D, float eps,
                        int8_t *oq, float *od) {
    const unsigned h = blockIdx.x, t = blockIdx.y, npt = D / 32, k0 = threadIdx.x * npt;
    const uint64_t idx = ((uint64_t)t * H + h) * D + k0;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) ss += o[idx + i] * o[idx + i];
    const float r = rsqrtf(sum(ss) / D + eps);
    float v[4];
    for (unsigned i = 0; i < npt; i++) o[idx + i] = v[i] = o[idx + i] * r * w[k0 + i] * sigmoid(z[idx + i]);
    if (!oq) return;
    float a = 0;
    for (unsigned i = 0; i < npt; i++) a = fmaxf(a, fabsf(v[i]));
    for (unsigned m = 1; m < 32 / npt; m <<= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, m));
    const float sc = __fdiv_rn(a, 127.f);
    for (unsigned i = 0; i < npt; i++) oq[idx + i] = (int8_t)__float2int_rn(sc > 0 ? __fdiv_rn(v[i], sc) : 0.f);
    if (!(k0 % 32)) od[idx / 32] = sc;
}

__global__ void ngram_gate(float *gated, float *normed, const float *R, const float *key,
                           const float *val, const float *gk, const float *gq, const float *gc,
                           unsigned E, unsigned hc, float eps) {
    const unsigned s = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const uint64_t base = ((uint64_t)t * hc + s) * E;
    __shared__ float red[32];
    float sk = 0, sq = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { sk += key[base + i] * key[base + i]; sq += R[base + i] * R[base + i]; }
    const float ik = rsqrtf(block_sum(sk, red) / E + eps);
    const float iq = rsqrtf(block_sum(sq, red) / E + eps);
    float dot = 0;
    for (unsigned i = tid; i < E; i += blockDim.x)
        dot += (key[base + i] * ik * gk[s * E + i]) * (R[base + i] * iq * gq[s * E + i]);
    const float a = block_sum(dot, red) * rsqrtf((float)E);
    const float mag = sqrtf(fmaxf(fabsf(a), 1e-6f));
    const float gate = sigmoid(a > 0 ? mag : a < 0 ? -mag : 0);
    float ss = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        const float v = gate * val[(uint64_t)t * E + i];
        gated[base + i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) normed[base + i] = gated[base + i] * inv * gc[s * E + i];
}

__global__ void ngram_conv(float *R, const float *gated, const float *normed, float *history,
                           const char *w, unsigned type, unsigned T, unsigned C, unsigned K,
                           unsigned dilation, float *snap, unsigned st, float *snap2, unsigned st2) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const unsigned H = (K - 1) * dilation;
    float hist[9], taps[4];
    for (unsigned i = 0; i < H; i++) hist[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = scalar(w, c * K + i, type);
    for (unsigned t = 0; t < T; t++) {
        const uint64_t p = (uint64_t)t * C + c;
        const float cur = normed[p];
        float v = taps[K - 1] * cur;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * hist[i * dilation];
        for (unsigned i = 0; i + 1 < H; i++) hist[i] = hist[i + 1];
        hist[H - 1] = cur;
        R[p] += gated[p] + silu(v);
        if (snap && t == st) for (unsigned i = 0; i < H; i++) snap[(uint64_t)i * C + c] = hist[i];
        if (snap2 && t == st2) for (unsigned i = 0; i < H; i++) snap2[(uint64_t)i * C + c] = hist[i];
    }
    for (unsigned i = 0; i < H; i++) history[(uint64_t)i * C + c] = hist[i];
}

} // namespace qwen4_cuda

extern "C" int ds4_gpu_qwen4_conv_stream_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *history,
        const void *map, uint64_t size, uint64_t offset, uint32_t T, uint32_t C, uint32_t K, bool activate) {
    using namespace qwen4_cuda;
    if (!T || !C || K < 2 || K > 4 || !tensor(x, (uint64_t)T * C * 4) || !tensor(history, (uint64_t)(K - 1) * C * 4)) return 0;
    const char *w = weight(map, size, offset, (uint64_t)C * K * 4);
    if (!w) return 0;
    conv<<<(C + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)x->ptr, (float *)history->ptr,
        (const float *)w, T, C, K, activate, NULL, UINT_MAX, NULL, UINT_MAX);
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_prep_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *a, ds4_gpu_tensor *b,
        const void *map, uint64_t size, uint64_t ao, uint64_t bo, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D) {
    using namespace qwen4_cuda;
    if (!T || !Hk || !Hv || D < 32 || D > 128 || D % 32 ||
        !tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !tensor(a, (uint64_t)T * Hv * 4) || !tensor(b, (uint64_t)T * Hv * 4)) return 0;
    const char *A = weight(map, size, ao, (uint64_t)Hv * 4), *bias = weight(map, size, bo, (uint64_t)Hv * 4);
    if (!A || !bias) return 0;
    gdn_prep<<<dim3(Hk, T), 32, 0, cuda_decode_stream()>>>((float *)qkv->ptr,
        (float *)a->ptr, (float *)b->ptr, (const float *)A, (const float *)bias, Hk, Hv, D);
    return launched();
}

extern "C" void ds4_gpu_qwen4_set_verify_rows_exact(bool on) { (void)on; }

extern "C" int ds4_gpu_qwen4_gdn_scan_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
        uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t bytes = (uint64_t)Hv * D * D * 4;
    if (!T || !Hk || !Hv || Hv % Hk || D < 32 || D > 128 || D % 32 ||
        !tensor(out, (uint64_t)T * Hv * D * 4) || !tensor(state, bytes) ||
        !tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !tensor(a, (uint64_t)T * Hv * 4) || !tensor(b, (uint64_t)T * Hv * 4) ||
        (snap && !tensor(snap, bytes)) || (snap2 && !tensor(snap2, bytes))) return 0;
#define QWEN_GDN(ROWS, DIM) gdn_scan<ROWS,DIM><<<dim3((DIM+4*ROWS-1)/(4*ROWS),Hv),128,0,cuda_decode_stream()>>>((float *)out->ptr, \
        (float *)state->ptr,(const float *)qkv->ptr,(const float *)a->ptr,(const float *)b->ptr, \
        T,Hk,Hv,snap ? (float *)snap->ptr : NULL,st,snap2 ? (float *)snap2->ptr : NULL,st2)
#define QWEN_GDN_DIM(DIM) case DIM: if (T > 8) { QWEN_GDN(4,DIM); } else { QWEN_GDN(1,DIM); } break
    switch (D) { QWEN_GDN_DIM(32); QWEN_GDN_DIM(64); QWEN_GDN_DIM(96); QWEN_GDN_DIM(128); }
#undef QWEN_GDN_DIM
#undef QWEN_GDN
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_out_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *z,
        const void *map, uint64_t size, uint64_t off, uint32_t T, uint32_t H, uint32_t D, float eps) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * H * D;
    if (!n || D < 32 || D > 128 || D % 32 || !tensor(out, n * 4) || !tensor(z, n * 4)) return 0;
    const char *w = weight(map, size, off, (uint64_t)D * 4);
    if (!w) return 0;
    int8_t *oq = NULL;
    float *od = NULL;
    if (!q8_out(out->ptr, T, H * D, &oq, &od)) q8_out_drop(out->ptr);
    gdn_out<<<dim3(H, T), 32, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)z->ptr, (const float *)w, H, D, eps,
                                                          oq, od);
    return launched();
}

extern "C" int ds4_gpu_qwen4_ple_gate_tensor(ds4_gpu_tensor *gated, ds4_gpu_tensor *normed,
        const ds4_gpu_tensor *R, const ds4_gpu_tensor *key, const ds4_gpu_tensor *val,
        const void *map, uint64_t size, uint64_t ko, uint64_t qo, uint64_t co,
        uint32_t T, uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_cuda;
    const uint64_t bytes = (uint64_t)T * E * hc * 4, wb = (uint64_t)E * hc * 4;
    if (!T || !E || !hc || hc > 4 || !tensor(gated, bytes) || !tensor(normed, bytes) ||
        !tensor(R, bytes) || !tensor(key, bytes) || !tensor(val, (uint64_t)T * E * 4)) return 0;
    const char *gk = weight(map, size, ko, wb), *gq = weight(map, size, qo, wb), *gc = weight(map, size, co, wb);
    if (!gk || !gq || !gc) return 0;
    ngram_gate<<<dim3(hc, T), 128, 0, cuda_decode_stream()>>>((float *)gated->ptr, (float *)normed->ptr,
        (const float *)R->ptr, (const float *)key->ptr, (const float *)val->ptr,
        (const float *)gk, (const float *)gq, (const float *)gc, E, hc, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_ple_conv_tensor(ds4_gpu_tensor *R, const ds4_gpu_tensor *gated,
        const ds4_gpu_tensor *normed, ds4_gpu_tensor *history, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t T, uint32_t C, uint32_t K, uint32_t dilation,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * C * 4, hb = (uint64_t)(K - 1) * dilation * C * 4;
    if (!T || !C || K < 2 || K > 4 || !dilation || dilation > 3 ||
        (type != 0 && type != 1) || !tensor(R, n) || !tensor(gated, n) || !tensor(normed, n) ||
        !tensor(history, hb) || (snap && !tensor(snap, hb)) || (snap2 && !tensor(snap2, hb))) return 0;
    const char *w = weight(map, size, off, row_bytes(type, (uint64_t)C * K));
    if (!w) return 0;
    ngram_conv<<<(C + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)R->ptr,
        (const float *)gated->ptr, (const float *)normed->ptr, (float *)history->ptr, w, type, T, C, K,
        dilation, snap ? (float *)snap->ptr : NULL, st, snap2 ? (float *)snap2->ptr : NULL, st2);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_norm_tensor(ds4_gpu_tensor *xn, ds4_gpu_tensor *inj,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t go, uint64_t io,
        uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * E * hc;
    if (!T || !E || !hc || hc > 8 || ni > 4 ||
        !tensor(xn, n * 4) || !tensor(R, n * 4) ||
        (ni && !tensor(inj, (uint64_t)T * hc * 8 * ni * 4))) return 0;
    const char *gamma = weight(map, size, go, (uint64_t)E * hc * 4);
    const char *wi = ni ? weight(map, size, io, row_bytes(type, (uint64_t)E * hc) * ni) : gamma;
    if (!gamma || !wi || (type != 0 && type != 1 && type != 8 && type != 30)) return 0;
    int8_t *xq = NULL;
    float *xd = NULL;
    if (T > 8 || E % 256 || !q8_out(xn->ptr, T, E * hc, &xq, &xd)) q8_out_drop(xn->ptr);
    if (T > 8) {
#define QWEN_HC_NORM(TYPE) hc_norm_prefill<TYPE><<<dim3(hc,T),256,0,cuda_decode_stream()>>>((float *)xn->ptr, \
        ni ? (float *)inj->ptr : NULL,(const float *)R->ptr,(const float *)gamma,wi,E,hc,ni,eps)
        if (type == 0) { QWEN_HC_NORM(0); }
        else if (type == 1) { QWEN_HC_NORM(1); }
        else if (type == 30) { QWEN_HC_NORM(30); }
        else { QWEN_HC_NORM(8); }
#undef QWEN_HC_NORM
        return launched();
    }
#define QWEN_HC_ROWS(TYPE) hc_norm_rows<TYPE,false><<<dim3(hc*8,T),128,0,cuda_decode_stream()>>>((float *)xn->ptr, \
        ni ? (float *)inj->ptr : NULL,NULL,(const float *)R->ptr,NULL,NULL,(const float *)gamma,wi,E,hc,ni,eps,xq,xd)
    if (type == 0) { QWEN_HC_ROWS(0); }
    else if (type == 1) { QWEN_HC_ROWS(1); }
    else if (type == 30) { QWEN_HC_ROWS(30); }
    else { QWEN_HC_ROWS(8); }
#undef QWEN_HC_ROWS
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_gate_mix_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *xn, const ds4_gpu_tensor *lo, const void *map, uint64_t size,
        uint64_t offset, uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t rank) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !rank || !tensor(out, (uint64_t)T * E * 4) ||
        !tensor(xn, (uint64_t)T * E * hc * 4) || !tensor(lo, (uint64_t)T * rank * 4)) return 0;
    const char *w = weight(map, size, offset, row_bytes(type, rank) * E * hc);
    if (!w || (type != 0 && type != 1 && type != 8 && type != 30)) return 0;
    /* hc_mix's alignment rule for the float4 reads of F32 and F16 rows */
    const uint64_t urb = row_bytes(type, rank);
    const bool rows = T <= 8 && !(rank % 4) && rank <= 512 &&
        (type == 0 ? !((uintptr_t)w & 15) && !(urb % 16) : type == 1 ? !((uintptr_t)w & 7) && !(urb % 8) : true);
#define QWEN_HC_ROWS(TYPE, R) hc_mix_rows<TYPE,R><<<(E + 3) / 4,128,0,cuda_decode_stream()>>>((float *)out->ptr, \
        (const float *)xn->ptr,(const float *)lo->ptr,w,T,E,hc,rank,row_bytes(TYPE,rank))
#define QWEN_HC(TYPE) \
    if (rows && T == 1) { QWEN_HC_ROWS(TYPE,1); } \
    else if (rows && T <= 4) { QWEN_HC_ROWS(TYPE,4); } \
    else if (rows) { QWEN_HC_ROWS(TYPE,8); } \
    else hc_mix<TYPE><<<dim3((E + 3) / 4, T),128,0,cuda_decode_stream()>>>((float *)out->ptr, \
        (const float *)xn->ptr,(const float *)lo->ptr,w,E,hc,rank,row_bytes(TYPE,rank))
    if (type == 0) { QWEN_HC(0); }
    else if (type == 1) { QWEN_HC(1); }
    else if (type == 30) { QWEN_HC(30); }
    else { QWEN_HC(8); }
#undef QWEN_HC
#undef QWEN_HC_ROWS
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_combine_tensor(ds4_gpu_tensor *R,
        const ds4_gpu_tensor *out, const ds4_gpu_tensor *inj, uint32_t T, uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !tensor(R, (uint64_t)T * E * hc * 4) ||
        !tensor(out, (uint64_t)T * E * 4) || !tensor(inj, (uint64_t)T * hc * hc * 8 * 4)) return 0;
    hc_combine<<<dim3((E + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (float *)R->ptr, (const float *)out->ptr, (const float *)inj->ptr, E, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_lo_act_tensor(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src, uint32_t T, uint32_t hc, uint32_t rank) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * rank;
    if (!n || !hc || !tensor(dst, n * 4) || !tensor(src, n * 4)) return 0;
    hc_lo<<<(n + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)dst->ptr, (const float *)src->ptr, n, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_mix_rows_tensor(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *up, const ds4_gpu_tensor *xn, uint32_t T, uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !tensor(dst, (uint64_t)T * E * 4) ||
        !tensor(up, (uint64_t)T * hc * E * 4) || !tensor(xn, (uint64_t)T * hc * E * 4)) return 0;
    hc_rows<<<dim3((E + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (float *)dst->ptr, (const float *)up->ptr, (const float *)xn->ptr, E, hc);
    return launched();
}

extern "C" void ds4_gpu_qwen4_set_rope(const float *freq, uint32_t n, float scale) {
    qwen4_cuda::rope_set = freq != NULL;
    qwen4_cuda::rope_scale = freq ? scale : 1;
    memset(qwen4_cuda::rope_freq, 0, sizeof(qwen4_cuda::rope_freq));
    if (freq) memcpy(qwen4_cuda::rope_freq, freq, std::min(n, 32u) * sizeof(float));
}

extern "C" int ds4_gpu_qwen4_attn_prep_tensor(ds4_gpu_tensor *q, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, ds4_gpu_tensor *iqout, ds4_gpu_tensor *ikc,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kp, const ds4_gpu_tensor *vp,
        const ds4_gpu_tensor *iq, const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3,
        const void *map, uint64_t size, uint64_t qo, uint64_t ko, uint64_t io,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t nrot,
        uint32_t Hi, uint32_t Di, uint32_t pos0, uint32_t cap, float base, float eps) {
    using namespace qwen4_cuda;
    const uint64_t qb = (uint64_t)T * H * D * 4, kb = (uint64_t)T * Hkv * D * 4, ib = (uint64_t)T * Hi * Di * 4;
    if (!T || !H || !Hkv || H % Hkv || !Hi || D < 32 || D > 256 || D % 32 ||
        Di < 32 || Di > 128 || Di % 32 || nrot > 64 || nrot > D || nrot > Di || nrot % 2 ||
        (uint64_t)pos0 + T > cap || !tensor(q, qb) || !tensor(gate, qb) || !tensor(qg, qb * 2) ||
        !tensor(kp, kb) || !tensor(vp, kb) || !tensor(iq, ib) || !tensor(iqout, ib) ||
        !tensor(ik, (uint64_t)T * Di * 4) || !tensor(ikc, (uint64_t)cap * Di * 4) ||
        !tensor(kc, (uint64_t)cap * kv_row_bytes((uint64_t)Hkv * D, g_kv_q8)) ||
        !tensor(vc, (uint64_t)cap * kv_row_bytes((uint64_t)Hkv * D, g_kv_q8)) ||
        !tensor(pos3, (uint64_t)cap * 16)) return 0;
    const char *gq = weight(map, size, qo, D * 4), *gk = weight(map, size, ko, D * 4), *giq = weight(map, size, io, Di * 4);
    if (!gq || !gk || !giq) return 0;
#define QWEN_PREP(Q8) attn_prep<Q8><<<dim3(H + Hkv + Hi + 1, T), 32, 0, cuda_decode_stream()>>>((float *)q->ptr, \
        (float *)gate->ptr, (char *)kc->ptr, (char *)vc->ptr, (float *)iqout->ptr, (float *)ikc->ptr, \
        (const float *)qg->ptr, (const float *)kp->ptr, (const float *)vp->ptr, (const float *)iq->ptr, \
        (const float *)ik->ptr, (const uint32_t *)pos3->ptr, (const float *)gq, (const float *)gk, (const float *)giq, \
        H, Hkv, D, Hi, Di, pos0, eps, rope(nrot, base))
    if (g_kv_q8) QWEN_PREP(true);
    else QWEN_PREP(false);
#undef QWEN_PREP
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_block_key_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3, const void *map, uint64_t size,
        uint64_t off, uint32_t b0, uint32_t N, uint32_t ratio, uint32_t D, uint32_t nrot,
        float base, float eps) {
    using namespace qwen4_cuda;
    const uint64_t rows = (uint64_t)b0 + N;
    if (!N || !ratio || D < 32 || D > 128 || D % 32 || nrot > D || nrot > 64 || nrot % 2 ||
        !tensor(out, rows * D * 2) || !tensor(ik, rows * ratio * D * 4) || !tensor(pos3, rows * ratio * 16)) return 0;
    const char *w = weight(map, size, off, D * 4);
    if (!w) return 0;
    block_key<<<N, 32, 0, cuda_decode_stream()>>>((__half *)out->ptr, (const float *)ik->ptr,
        (const uint32_t *)pos3->ptr, (const float *)w, b0, ratio, D, eps, rope(nrot, base));
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_score_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *tiles,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *key, uint32_t T, uint32_t N,
        uint32_t H, uint32_t D, uint32_t pos0, uint32_t ratio) {
    using namespace qwen4_cuda;
    const unsigned nt = (N + 7) / 8;
    if (!T || !N || !H || !D || !ratio || !tensor(out, (uint64_t)T * N * 4) ||
        !tensor(q, (uint64_t)T * H * D * 4) || !tensor(key, (uint64_t)N * D * 2) ||
        (tiles && !tensor(tiles, (uint64_t)T * nt * 4))) return 0;
    idx_score<<<dim3((N + 3) / 4, T), 128, 0, cuda_decode_stream()>>>((float *)out->ptr,
        (const float *)q->ptr, (const __half *)key->ptr, N, H, D, pos0, ratio);
    if (!launched()) return 0;
    if (tiles) tile_max<<<dim3((nt + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (unsigned *)tiles->ptr, (const float *)out->ptr, N, nt);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_select_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *score,
        const ds4_gpu_tensor *tiles, uint32_t N, uint32_t T, uint32_t K) {
    using namespace qwen4_cuda;
    (void)tiles;
    if (!T || !K || K > N || !tensor(out, (uint64_t)T * K * 4) || !tensor(score, (uint64_t)T * N * 4)) return 0;
    /* rows of 8K scores and more search the tile maxima first;
     * DS4_QWEN4_IDX_SELECT_FULL=1 searches the whole row for A/B */
    const unsigned NT = (N + 7) / 8, cap = 8 * K;
    const size_t smem = ((size_t)NT + cap) * 4;
    static int max_smem[DS4_MAX_GPUS] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int &ms = max_smem[current_tier()];   /* the attribute is the device's */
    if (ms < 0) {
        ms = 0;
        /* the opt-in limit (96 KB on Volta) holds the static histogram too */
        if (cudaFuncSetAttribute(idx_select, cudaFuncAttributeMaxDynamicSharedMemorySize, 88 * 1024) == cudaSuccess)
            ms = 88 * 1024;
        (void)cudaGetLastError();
    }
    const bool tiled = N >= 8192 && NT >= K && smem <= (size_t)ms && !getenv("DS4_QWEN4_IDX_SELECT_FULL");
    idx_select<<<T, 1024, tiled ? smem : 0, cuda_decode_stream()>>>((int *)out->ptr, (const float *)score->ptr,
                                                                    N, K, tiled ? cap : 0);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_expand_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *count,
        const ds4_gpu_tensor *blocks, uint32_t T, uint32_t K, uint32_t ratio, uint32_t pos0, uint32_t stride) {
    using namespace qwen4_cuda;
    if (!T || !K || !ratio || (uint64_t)K * ratio + ratio - 1 > stride ||
        !tensor(out, (uint64_t)T * stride * 4) || !tensor(count, (uint64_t)T * 4) || !tensor(blocks, (uint64_t)T * K * 4)) return 0;
    idx_expand<<<T, 256, 0, cuda_decode_stream()>>>((int *)out->ptr, (unsigned *)count->ptr,
        (const int *)blocks->ptr, K, ratio, pos0, stride);
    return launched();
}

extern "C" uint64_t ds4_gpu_qwen4_attn_part_floats(uint32_t T, uint32_t H, uint32_t D) {
    return (uint64_t)T * H * 64 * (D + 2);
}

/* Round x's T rows of K to int8 once for the decode and verify products
 * that read it next (dense rows and the experts' gate and up), until the
 * next call; x must not change meanwhile.  x = NULL drops it. */
extern "C" int ds4_gpu_qwen4_q8_input(const ds4_gpu_tensor *x, uint32_t T, uint32_t K) {
    using namespace qwen4_cuda;
    const int tier = current_tier();
    g_q8_input[tier].x = NULL;
    if (!x || !act_q8() || !T || T > 8 || !K || K % 32) return 1;
    if (!tensor(x, (uint64_t)T * K * 4)) return 0;
    const uint64_t n = (uint64_t)T * K, qbytes = (n + 15) / 16 * 16, bytes = qbytes + n / 32 * 4;
    if (g_q8_input[tier].cap < bytes) {
        if (g_q8_input[tier].q) (void)cudaFree(g_q8_input[tier].q);
        g_q8_input[tier].q = NULL;
        g_q8_input[tier].cap = 0;
        if (cudaMalloc(&g_q8_input[tier].q, bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
        g_q8_input[tier].cap = bytes;
    }
    g_q8_input[tier].d = (float *)(g_q8_input[tier].q + qbytes);
    quantize_q8<<<(unsigned)((n + 255) / 256),256,0,cuda_decode_stream()>>>(g_q8_input[tier].q, g_q8_input[tier].d,
                                                                               (const float *)x->ptr, n);
    if (!launched()) return 0;
    g_q8_input[tier].x = x->ptr;
    g_q8_input[tier].T = T;
    g_q8_input[tier].K = K;
    return 1;
}

extern "C" int ds4_gpu_qwen4_dense_mm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t off, uint32_t type, uint32_t T, uint32_t K, uint32_t M) {
    using namespace qwen4_cuda;
    if (!T || !K || !M || !tensor(x, (uint64_t)T*K*4) || !tensor(out, (uint64_t)T*M*4)) return 0;
    const uint64_t rb = row_bytes(type,K);
    if (!rb) return 0;
    const char *w = weight(map, size, off, rb*M);
    if (!w) return 0;
    if (T <= 8) return matvec_dispatch((float *)out->ptr, w, (const float *)x->ptr, type, T, K, M);
    if (type == 1 && T >= 32 && T <= INT_MAX && K <= INT_MAX && M <= INT_MAX &&
        g_cublas_ready && !g_quality_mode && !getenv("DS4_CUDA_NO_TF32"))
        return dense_f16_blas((float *)out->ptr,(const float *)x->ptr,(const __half *)w,T,K,M);
    if (type != 0 && T >= 32 && T <= INT_MAX && K <= INT_MAX && M <= INT_MAX &&
        g_cublas_ready && !g_quality_mode && !getenv("DS4_CUDA_NO_TF32"))
        return dense_half_blas((float *)out->ptr,(const float *)x->ptr,w,type,T,K,M);
    if (g_cublas_ready && T >= 32 && K <= INT_MAX && M <= INT_MAX && T <= INT_MAX)
        return dense_blas((float *)out->ptr,(const float *)x->ptr,w,type,T,K,M);
    return matrix_dispatch((float *)out->ptr, (const float *)x->ptr, w, NULL, NULL, NULL,
                           type, 1, T, 1, 1, K, M, 0, false);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *map,
        uint64_t size, uint64_t off, uint64_t K, uint64_t M, const ds4_gpu_tensor *x, uint64_t T) {
    if (K > UINT_MAX || M > UINT_MAX || T > UINT_MAX) return 0;
    return ds4_gpu_qwen4_dense_mm_tensor(out, x, map, size, off, 8, T, K, M);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_weights_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *w,
        uint32_t K, uint32_t M, const ds4_gpu_tensor *x) {
    using namespace qwen4_cuda;
    const uint64_t rb = row_bytes(8, K);
    if (!K || !M || !rb || !tensor(w, rb*M) || !tensor(x, (uint64_t)K*4) || !tensor(out, (uint64_t)M*4)) return 0;
    return matvec_dispatch((float *)out->ptr, (const char *)w->ptr, (const float *)x->ptr, 8, 1, K, M);
}

extern "C" int ds4_gpu_qwen4_multi_gemv_tensor(const ds4_gpu_tensor *x, uint32_t T,
        uint32_t K, uint32_t N, ds4_gpu_tensor *const *outs, const void *map, uint64_t size,
        const uint64_t *offsets, const uint32_t *types, const uint32_t *rows) {
    if (!N || N > 4 || !outs || !offsets || !types || !rows) return 0;
    for (unsigned i = 0; i < N; i++)
        if (!ds4_gpu_qwen4_dense_mm_tensor(outs[i], x, map, size, offsets[i], types[i], T, K, rows[i])) return 0;
    return 1;
}

extern "C" int ds4_gpu_qwen4_q8_pair_tensor(ds4_gpu_tensor *o0, ds4_gpu_tensor *o1,
        const void *map, uint64_t size, uint64_t w0, uint64_t w1, uint64_t K,
        uint64_t M0, uint64_t M1, const ds4_gpu_tensor *x, uint64_t T) {
    if ((M0 & 1) || (M1 & 1)) return 0;
    return ds4_gpu_qwen4_matmul_q8_0_tensor(o0, map, size, w0, K, M0, x, T) &&
           ds4_gpu_qwen4_matmul_q8_0_tensor(o1, map, size, w1, K, M1, x, T);
}

extern "C" int ds4_gpu_qwen4_router_topk_tensor(ds4_gpu_tensor *sel, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t K, ds4_gpu_tensor *sg, uint32_t T, uint32_t NE, uint32_t NS) {
    using namespace qwen4_cuda;
    if (!T || !NE || NE > 512 || !NS || NS > NE || NS > 32 ||
        !tensor(sel, (uint64_t)T*NS*4) || !tensor(weights, (uint64_t)T*NS*4) || !tensor(logits, (uint64_t)T*NE*4)) return 0;
    const char *wg = K ? weight(map, size, off, row_bytes(type, K)) : NULL;
    if (K && (!wg || !tensor(x, (uint64_t)T*K*4) || !tensor(sg, (uint64_t)T*4))) return 0;
    router<<<T,64,0,cuda_decode_stream()>>>((int *)sel->ptr, (float *)weights->ptr, (const float *)logits->ptr,
        K ? (const float *)x->ptr : NULL, wg, K ? (float *)sg->ptr : NULL, NE, NS, K, type);
    return launched();
}

extern "C" int ds4_gpu_qwen4_moe_mid2_tensor(ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t go, uint64_t uo,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t sgo, uint64_t suo, uint32_t st, uint32_t sut) {
    using namespace qwen4_cuda;
    const unsigned NO = NS + (st != UINT_MAX);
    if (!T || !NE || !NS || !K || !M || !tensor(mid, (uint64_t)T*NO*M*4) ||
        !tensor(x, (uint64_t)T*K*4) || !tensor(sel, (uint64_t)T*NS*4)) return 0;
    const uint64_t bytes = expert_row_bytes(type, K)*M*NE;
    const char *g = weight(map,size,go,bytes), *u = weight(map,size,uo,bytes);
    const char *sg = st != UINT_MAX ? weight(map,size,sgo,row_bytes(st, K)*M) : NULL;
    const char *su = st != UINT_MAX ? weight(map,size,suo,row_bytes(sut, K)*M) : NULL;
    if (!g || !u || (st != UINT_MAX && (!sg || !su))) return 0;
    return moe_mv_dispatch((float *)mid->ptr,(const float *)x->ptr,(const int *)sel->ptr,
                           g,u,sg,su,type,st,sut,NE,T,NS,K,M,false);
}

extern "C" int ds4_gpu_qwen4_moe_mid_tensor(ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t go, uint64_t uo,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t sgo, uint64_t suo, uint32_t st) {
    return ds4_gpu_qwen4_moe_mid2_tensor(mid,x,sel,map,size,go,uo,type,NE,T,NS,K,M,sgo,suo,st,st);
}

extern "C" int ds4_gpu_qwen4_moe_down_tensor(ds4_gpu_tensor *part, const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t off,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t so, uint32_t st) {
    using namespace qwen4_cuda;
    const unsigned NO = NS + (st != UINT_MAX);
    if (!T || !NE || !NS || !K || !M || !tensor(part,(uint64_t)T*NO*M*4) ||
        !tensor(mid,(uint64_t)T*NO*K*4) || !tensor(sel,(uint64_t)T*NS*4)) return 0;
    const char *w = weight(map,size,off,expert_row_bytes(type,K)*M*NE);
    const char *sw = st != UINT_MAX ? weight(map,size,so,row_bytes(st,K)*M) : NULL;
    if (!w || (st != UINT_MAX && !sw)) return 0;
    return moe_mv_dispatch((float *)part->ptr,(const float *)mid->ptr,(const int *)sel->ptr,
                           w,NULL,sw,NULL,type,st,st,NE,T,NS,K,M,true);
}

extern "C" int ds4_gpu_qwen4_moe_reduce_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *part,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *sg, const ds4_gpu_tensor *sh,
        ds4_gpu_tensor *R, const ds4_gpu_tensor *inj, uint32_t T, uint32_t NS, uint32_t stride,
        uint32_t D, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !NS || !D || stride < NS + (sg && !sh) || hc > 4 ||
        !tensor(out,(uint64_t)T*D*4) || !tensor(part,(uint64_t)T*stride*D*4) ||
        !tensor(weights,(uint64_t)T*NS*4) || (sg && !tensor(sg,(uint64_t)T*4)) ||
        (sh && !tensor(sh,(uint64_t)T*D*4)) ||
        (hc && (!tensor(R,(uint64_t)T*hc*D*4) || !tensor(inj,(uint64_t)T*hc*hc*8*4)))) return 0;
    moe_reduce<<<dim3((D+255)/256,T),256,0,cuda_decode_stream()>>>((float *)out->ptr,
        hc ? (float *)R->ptr : NULL, hc ? (const float *)inj->ptr : NULL, (const float *)part->ptr,
        (const float *)weights->ptr, sg ? (const float *)sg->ptr : NULL, sh ? (const float *)sh->ptr : NULL,
        NS,stride,D,hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_moe_build_lists_tensor(ds4_gpu_tensor *lists, ds4_gpu_tensor *counts,
        const ds4_gpu_tensor *sel, uint32_t T, uint32_t NS, uint32_t NE, uint32_t cap) {
    using namespace qwen4_cuda;
    if (!T || !NS || NS > NE || !NE || NE > 512 || cap < T || (uint64_t)T*NS > INT_MAX ||
        !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4) || !tensor(sel,(uint64_t)T*NS*4)) return 0;
    expert_lists<<<1,256,0,cuda_decode_stream()>>>((int *)lists->ptr,(int *)counts->ptr,(const int *)sel->ptr,T*NS,NE,cap);
    return launched();
}

static int qwen4_moe_mm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t o0, uint64_t o1, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap, bool down) {
    using namespace qwen4_cuda;
    if (!T || !NE || !NS || NS > NE || NO < NS || !K || !M || cap < T ||
        !tensor(out,(uint64_t)T*NO*M*4) || !tensor(x,(uint64_t)T*(down ? NO : 1)*K*4) ||
        !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4)) return 0;
    const uint64_t bytes = expert_row_bytes(type,K)*M*NE;
    const char *w0 = weight(map,size,o0,bytes), *w1 = down ? NULL : weight(map,size,o1,bytes);
    if (!w0 || (!down && !w1)) return 0;
    return matrix_dispatch((float *)out->ptr,(const float *)x->ptr,w0,w1,(const int *)lists->ptr,
        (const int *)counts->ptr,type,NE,T,NS,NO,K,M,cap,down);
}

extern "C" int ds4_gpu_qwen4_moe_mm_mid_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t go, uint64_t uo, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_moe_mm(out,x,lists,counts,map,size,go,uo,type,NE,T,NS,NO,K,M,cap,false);
}

extern "C" int ds4_gpu_qwen4_moe_mm_down_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_moe_mm(out,x,lists,counts,map,size,off,0,type,NE,T,NS,NO,K,M,cap,true);
}

extern "C" int ds4_gpu_qwen4_gdn_front_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *mixed, ds4_gpu_tensor *ga, ds4_gpu_tensor *gb,
        const void *map, uint64_t size, uint64_t co, uint64_t ao, uint64_t bo, uint64_t so, uint64_t dto,
        uint32_t type, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D, uint32_t CK, uint32_t K,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t C = ((uint64_t)2*Hk+Hv)*D, hb = (CK-1)*C*4;
    if (!T || !Hk || !Hv || !D || !K || C > UINT_MAX || CK < 2 || CK > 4 ||
        !tensor(qkv,(uint64_t)T*C*4) || !tensor(state,hb) ||
        (snap && (st >= T || !tensor(snap,hb))) || (snap2 && (st2 >= T || !tensor(snap2,hb)))) return 0;
    const char *w = weight(map,size,co,C*CK*4);
    if (!w || !ds4_gpu_qwen4_dense_mm_tensor(ga,mixed,map,size,ao,type,T,K,Hv) ||
              !ds4_gpu_qwen4_dense_mm_tensor(gb,mixed,map,size,bo,type,T,K,Hv)) return 0;
    conv<<<(C+255)/256,256,0,cuda_decode_stream()>>>((float *)qkv->ptr,(float *)state->ptr,(const float *)w,
        T,C,CK,true,snap ? (float *)snap->ptr : NULL,st,snap2 ? (float *)snap2->ptr : NULL,st2);
    return launched() && ds4_gpu_qwen4_gdn_prep_tensor(qkv,ga,gb,map,size,so,dto,T,Hk,Hv,D);
}

extern "C" int ds4_gpu_qwen4_decode_fusions_enabled(void) { return 1; }

/* the graph sets the K/V format before it allocates its caches */
extern "C" void ds4_gpu_qwen4_set_kv_q8(bool on) { qwen4_cuda::g_kv_q8 = on; }

/* Q8 K/V suits the GPUs without sm_80's tensor-core attention tiles
 * (attention_group stages f16 rows) */
extern "C" int ds4_gpu_qwen4_kv_q8_preferred(void) { return !ds4_cuda_attn_tokentile_arch_ok(); }
/* the half expert tiles: sm_80's, and Volta's for the types volta_tiles takes */
extern "C" int ds4_gpu_qwen4_tensor_tiles(void) {
    return (ds4_cuda_attn_tokentile_arch_ok() || qwen4_cuda::volta()) && !getenv("DS4_CUDA_NO_TF32");
}

extern "C" int ds4_gpu_qwen4_hc_combine_norm_tensor(ds4_gpu_tensor *next, const ds4_gpu_tensor *blk,
        const ds4_gpu_tensor *oldinj, ds4_gpu_tensor *xn, ds4_gpu_tensor *inj, const ds4_gpu_tensor *R,
        const void *map, uint64_t size, uint64_t go, uint64_t io, uint32_t type,
        uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    const uint64_t bytes = (uint64_t)T*E*hc*4;
    if (!qwen4_cuda::tensor(next,bytes) || !qwen4_cuda::tensor(R,bytes) || next->ptr == R->ptr ||
        !inj || !oldinj || inj->ptr == oldinj->ptr) return 0;
    if (T <= 8 && ni && hc <= 8 && ni <= 4 && qwen4_cuda::tensor(xn,bytes) && qwen4_cuda::tensor(blk,(uint64_t)T*E*4) &&
        qwen4_cuda::tensor(oldinj,(uint64_t)T*hc*hc*8*4) && qwen4_cuda::tensor(inj,(uint64_t)T*hc*8*ni*4) &&
        (type == 0 || type == 1 || type == 8 || type == 30)) {
        using namespace qwen4_cuda;
        const char *gamma = weight(map,size,go,(uint64_t)E*hc*4), *wi = weight(map,size,io,row_bytes(type,(uint64_t)E*hc)*ni);
        if (!gamma || !wi) return 0;
        int8_t *xq = NULL;
        float *xd = NULL;
        if (E % 256 || !q8_out(xn->ptr, T, E * hc, &xq, &xd)) q8_out_drop(xn->ptr);
#define QWEN_HC_COMBINE(TYPE) hc_norm_rows<TYPE,true><<<dim3(hc*8,T),128,0,cuda_decode_stream()>>>((float *)xn->ptr, \
        (float *)inj->ptr,(float *)next->ptr,(const float *)R->ptr,(const float *)blk->ptr,(const float *)oldinj->ptr, \
        (const float *)gamma,wi,E,hc,ni,eps,xq,xd)
        if (type == 0) { QWEN_HC_COMBINE(0); }
        else if (type == 1) { QWEN_HC_COMBINE(1); }
        else if (type == 30) { QWEN_HC_COMBINE(30); }
        else { QWEN_HC_COMBINE(8); }
#undef QWEN_HC_COMBINE
        return launched();
    }
    if (!cuda_ok(cudaMemcpyAsync(next->ptr,R->ptr,bytes,cudaMemcpyDeviceToDevice,cuda_decode_stream()),"Qwen HC copy")) return 0;
    return ds4_gpu_qwen4_hc_combine_tensor(next,blk,oldinj,T,E,hc) &&
           ds4_gpu_qwen4_hc_norm_tensor(xn,inj,next,map,size,go,io,type,T,E,hc,ni,eps);
}

extern "C" int ds4_gpu_qwen4_mtp_stage_tensor(ds4_gpu_tensor *cat, const ds4_gpu_tensor *e,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t eo, uint64_t ho,
        uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_cuda;
    if (!E || !hc || hc > 4 || !tensor(cat,(uint64_t)(hc+1)*2*E*4) ||
        !tensor(e,(uint64_t)E*4) || !tensor(R,(uint64_t)hc*E*4)) return 0;
    const char *ge = weight(map,size,eo,(uint64_t)E*4), *gh = weight(map,size,ho,(uint64_t)hc*E*4);
    if (!ge || !gh) return 0;
    mtp_stage<<<hc+1,256,0,cuda_decode_stream()>>>((float *)cat->ptr,(const float *)e->ptr,
        (const float *)R->ptr,(const float *)ge,(const float *)gh,E,hc,eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_mtp_combine_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *proj,
        uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!E || !hc || hc > 4 || !tensor(out,(uint64_t)hc*E*4) || !tensor(proj,(uint64_t)(hc+1)*E*4)) return 0;
    mtp_combine<<<((uint64_t)E*hc+255)/256,256,0,cuda_decode_stream()>>>((float *)out->ptr,(const float *)proj->ptr,E,hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_argmax_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *logits, uint32_t N) {
    using namespace qwen4_cuda;
    const uint64_t blocks = ((uint64_t)N+4095)/4096;
    if (!N || !tensor(out,8) || !tensor(logits,(uint64_t)N*4) || !tensor(scratch,blocks*sizeof(max_pair))) return 0;
    argmax<false><<<blocks,256,0,cuda_decode_stream()>>>((int *)out->ptr,(max_pair *)scratch->ptr,(const float *)logits->ptr,N);
    argmax<true><<<1,256,0,cuda_decode_stream()>>>((int *)out->ptr,(max_pair *)scratch->ptr,NULL,blocks);
    return launched();
}

extern "C" int ds4_gpu_qwen4_vision_encode(float *out, const float *patches, const float *pos_embed,
        uint32_t N, uint32_t grid_w, const void *map, uint64_t size, const ds4_qwen4_vision_weights *w) {
    using namespace qwen4_cuda;
    if (!out || !patches || !pos_embed || !w || !map || !N || N > 65535 || ds4_gpu_commands_active()) return 0;
    const unsigned E = w->n_embd, FF = w->n_ff, H = w->n_head, D = H ? E/H : 0;
    const unsigned P = w->n_patch, M = w->n_merge, O = w->n_out;
    if (M != 2 || !grid_w || grid_w%M || N%grid_w || (N/grid_w)%M ||
        (D != 64 && D != 72) || H*D != E || !FF || !O || !P || P > 32 || E%32) return 0;
    const unsigned IP = 3*P*P, ME = E*M*M, merged = N/(M*M);
    std::vector<ds4_gpu_tensor *> buffers;
    auto alloc = [&](uint64_t n) {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n*4);
        buffers.push_back(t);
        return t;
    };
    ds4_gpu_tensor *patch = alloc((uint64_t)N*IP), *a0 = alloc((uint64_t)N*E), *a1 = alloc((uint64_t)N*E);
    ds4_gpu_tensor *pos = alloc((uint64_t)N*E), *x = alloc((uint64_t)N*E), *tmp = alloc((uint64_t)N*E);
    ds4_gpu_tensor *qkv = alloc((uint64_t)N*3*E), *q = alloc((uint64_t)N*E), *k = alloc((uint64_t)N*E);
    ds4_gpu_tensor *v = alloc((uint64_t)N*E), *attn = alloc((uint64_t)N*E), *ffn = alloc((uint64_t)N*FF);
    ds4_gpu_tensor *m0 = alloc((uint64_t)merged*ME), *res = alloc((uint64_t)merged*O);
    bool ok = true, active = false;
    for (const auto b : buffers) if (!b) ok = false;
    auto ptr = [](ds4_gpu_tensor *t) { return (float *)t->ptr; };
    auto wf = [&](uint64_t off, unsigned n) { return (const float *)weight(map,size,off,(uint64_t)n*4); };
    auto mm = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t off, unsigned type,
                  unsigned rows, unsigned K, unsigned M) {
        return ds4_gpu_qwen4_dense_mm_tensor(dst,src,map,size,off,type,rows,K,M);
    };
    auto norm = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t wo, uint64_t bo) {
        const float *wgt = wf(wo,E), *bias = wf(bo,E);
        if (!wgt || !bias) return 0;
        vis_norm<<<N,256,0,cuda_decode_stream()>>>(ptr(dst),(const float *)src->ptr,wgt,bias,E,w->eps);
        return launched();
    };
    auto add = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t bo, unsigned rows,
                   unsigned width, unsigned mode) {
        const float *bias = wf(bo,width);
        if (!bias) return 0;
        vis_add<<<((uint64_t)rows*width+255)/256,256,0,cuda_decode_stream()>>>(ptr(dst),
            src ? (const float *)src->ptr : NULL,bias,rows,width,mode);
        return launched();
    };
    do {
        if (!ok) break;
        ok = ds4_gpu_tensor_write(patch,0,patches,(uint64_t)N*IP*4) &&
             ds4_gpu_tensor_write(pos,0,pos_embed,(uint64_t)N*E*4);
        if (!ok || !(active = ds4_gpu_begin_commands())) { ok = false; break; }
        ok = mm(a0,patch,w->patch_w0,w->patch_type,N,IP,E) && mm(a1,patch,w->patch_w1,w->patch_type,N,IP,E);
        const float *pb = wf(w->patch_b,E);
        if (!ok || !pb) { ok = false; break; }
        vis_patch<<<((uint64_t)N*E+255)/256,256,0,cuda_decode_stream()>>>(ptr(x),ptr(a0),ptr(a1),pb,ptr(pos),N,E);
        ok = launched();
        for (unsigned l = 0; ok && l < DS4_QWEN4_VISION_LAYERS; l++) {
            const auto &lw = w->layer[l];
            ok = norm(tmp,x,lw.ln1_w,lw.ln1_b) && mm(qkv,tmp,lw.qkv_w,lw.qkv_type,N,E,3*E);
            const float *qb = wf(lw.qkv_b,3*E);
            if (!ok || !qb) { ok = false; break; }
            vis_qkv<<<N,256,0,cuda_decode_stream()>>>(ptr(q),ptr(k),ptr(v),ptr(qkv),qb,H,D,grid_w);
            vis_attention<<<dim3(N,H),32,0,cuda_decode_stream()>>>(ptr(attn),ptr(q),ptr(k),ptr(v),N,H,D);
            ok = launched() && mm(tmp,attn,lw.out_w,lw.out_type,N,E,E) && add(x,tmp,lw.out_b,N,E,2) &&
                 norm(tmp,x,lw.ln2_w,lw.ln2_b) && mm(ffn,tmp,lw.up_w,lw.up_type,N,E,FF) &&
                 add(ffn,NULL,lw.up_b,N,FF,0) && mm(tmp,ffn,lw.down_w,lw.down_type,N,FF,E) &&
                 add(x,tmp,lw.down_b,N,E,2);
        }
        ok = ok && norm(tmp,x,w->post_ln_w,w->post_ln_b) && mm(m0,tmp,w->mm0_w,w->mm0_type,merged,ME,ME) &&
             add(m0,NULL,w->mm0_b,merged,ME,1) && mm(res,m0,w->mm2_w,w->mm2_type,merged,ME,O) &&
             add(res,NULL,w->mm2_b,merged,O,2);
    } while (false);
    if (active && !ds4_gpu_end_commands()) ok = false;
    if (ok) ok = ds4_gpu_tensor_read(res,0,out,(uint64_t)merged*O*4);
    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) ds4_gpu_tensor_free(*it);
    return ok;
}

extern "C" int ds4_gpu_qwen4_attn_decode_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc,
        const ds4_gpu_tensor *sel, const ds4_gpu_tensor *count, ds4_gpu_tensor *partial,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t pos0, bool sparse, uint32_t stride, float scale) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * H * D * 4, cb = ((uint64_t)pos0 + T) * kv_row_bytes((uint64_t)Hkv * D, g_kv_q8);
    if (!T || !H || !Hkv || H % Hkv || (D != 32 && D != 128 && D != 256) ||
        !tensor(out, n) || !tensor(q, n) || !tensor(gate, n) || !tensor(kc, cb) || !tensor(vc, cb) ||
        (sparse && (!stride || !tensor(sel, (uint64_t)T * stride * 4) || !tensor(count, (uint64_t)T * 4)))) return 0;
    const unsigned keys = sparse ? stride : pos0 + T;
    if (!partial && T >= 32 && D == 256 && H/Hkv <= 16 && !g_kv_q8 &&
        !((uintptr_t)kc->ptr&15) && !((uintptr_t)vc->ptr&15) &&
        ds4_cuda_attn_tokentile_arch_ok() && !g_quality_mode && !getenv("DS4_QWEN4_NO_ATTN_MM")) {
        attention_group<<<dim3(Hkv,T),256,0,cuda_decode_stream()>>>((float *)out->ptr,
            (const float *)q->ptr,(const float *)gate->ptr,(const __half *)kc->ptr,(const __half *)vc->ptr,
            sparse ? (const int *)sel->ptr : NULL,sparse ? (const unsigned *)count->ptr : NULL,
            H,Hkv,pos0,stride,sparse,scale);
        return launched();
    }
    const unsigned splits = partial ? std::min(64u, (keys + 31) / 32) : 1;
    if (partial && !tensor(partial, (uint64_t)T * H * splits * (D + 2) * 4)) return 0;
    const dim3 grid((H + 3) / 4, T, splits);
#define QWEN_ATTN(DIM, Q8) attention<DIM,Q8><<<grid, 128, 0, cuda_decode_stream()>>>((float *)out->ptr, \
        partial ? (float *)partial->ptr : NULL, (const float *)q->ptr, (const float *)gate->ptr, \
        (const char *)kc->ptr, (const char *)vc->ptr, sparse ? (const int *)sel->ptr : NULL, \
        sparse ? (const unsigned *)count->ptr : NULL, H, Hkv, pos0, stride, sparse, splits, (keys + splits - 1) / splits, scale)
    if (D == 32) { if (g_kv_q8) QWEN_ATTN(32, true); else QWEN_ATTN(32, false); }
    else if (D == 128) { if (g_kv_q8) QWEN_ATTN(128, true); else QWEN_ATTN(128, false); }
    else { if (g_kv_q8) QWEN_ATTN(256, true); else QWEN_ATTN(256, false); }
#undef QWEN_ATTN
    if (!launched()) return 0;
    if (splits > 1) attn_merge<<<dim3(H, T), D, 0, cuda_decode_stream()>>>((float *)out->ptr,
        (const float *)partial->ptr, (const float *)gate->ptr, H, D, splits);
    return launched();
}
