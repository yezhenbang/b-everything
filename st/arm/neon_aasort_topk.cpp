

#include "MPOpHelper.hpp"
#include "arm_neon.h"
#include "neon/neon_bitonic_sort.h"

// #include <smmintrin.h>
#include <algorithm>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h> /* for memcpy */
#include <unistd.h> /* for sysconf */

/*
 * aasort implemented following instructions in paper
 * "AA-Sort: A New Parallel Sorting Algorithm for Multi-Core SIMD Processors"
 * by H. Inoue et al.
 * See: www.trl.ibm.com/people/inouehrs/pdf/PACT2007-SIMDsort.pdf
 *
 * An earlier, non published version, which provides some more detail especially
 * regarding in-core step 2, can be found here:
 * http://www.research.ibm.com/trl/people/inouehrs/pdf/SPE-SIMDsort.pdf
 *
 * Conversely to the paper, we do not use thread-level concurrency.
 *
 * NOTE: We expect the number of data elements to be dividable by 16!
 *
 * Found quite some inspiration here:
 * https://github.com/herumi/opti/blob/master/intsort.hpp
 */

static uint32_t l2_cache = 64 * 1024;

#define __ai static inline __attribute__((__always_inline__, __nodebug__))

#ifdef __LITTLE_ENDIAN__
#define REGISTER_VSHUFFLE_NAIVE_F32(__r0, __r1, __r2, __r3)                                       \
    __ai float32x4_t vshuffle##__r0##__r1##__r2##__r3##_f32(float32x4_t __p0, float32x4_t __p1) { \
        float32x4_t __ret;                                                                        \
        __ret = __builtin_shufflevector(__p0, __p1, __r0, __r1, __r2, __r3);                      \
        return __ret;                                                                             \
    }
#else
#define REGISTER_VSHUFFLE_NAIVE_F32(__r0, __r1, __r2, __r3)                                       \
    __ai float32x4_t vshuffle##__r0##__r1##__r2##__r3##_f32(float32x4_t __p0, float32x4_t __p1) { \
        float32x4_t __rev0 = __builtin_shufflevector(__p0, __p0, 3, 2, 1, 0);                     \
        float32x4_t __rev1 = __builtin_shufflevector(__p1, __p1, 3, 2, 1, 0);                     \
        float32x4_t __ret = __builtin_shufflevector(__rev0, __rev1, __r0, __r1, __r2, __r3);      \
        __ret = __builtin_shufflevector(__ret, __ret, 3, 2, 1, 0);                                \
        return __ret;                                                                             \
    }
#endif

#define vtrn64_fp32(a, b)                                                              \
    do {                                                                               \
        float32x2_t vtrn64_tmp_a0 = vget_low_f32(a), vtrn64_tmp_a1 = vget_high_f32(a); \
        float32x2_t vtrn64_tmp_b0 = vget_low_f32(b), vtrn64_tmp_b1 = vget_high_f32(b); \
        a = vcombine_f32(vtrn64_tmp_a0, vtrn64_tmp_b0);                                \
        b = vcombine_f32(vtrn64_tmp_a1, vtrn64_tmp_b1);                                \
    } while (0)

#define vtrn32_fp32(a, b)                     \
    do {                                      \
        float32x4x2_t vtrn = vtrnq_f32(a, b); \
        a = vtrn.val[0];                      \
        b = vtrn.val[1];                      \
    } while (0)

__ai void aasort_vector_transpose(float32x4x4_t& a) {
    vtrn64_fp32(a.val[0], a.val[2]);
    vtrn64_fp32(a.val[1], a.val[3]);
    vtrn32_fp32(a.val[0], a.val[1]);
    vtrn32_fp32(a.val[2], a.val[3]);
}

__ai void aasort_vector_cmpswap(float32x4_t& a, float32x4_t& b) {
    float32x4_t tmp = vmaxq_f32(a, b);
    b = vminq_f32(a, b);
    a = tmp;
}

__ai void aasort_vector_cmpswap_skew(float32x4_t& a, float32x4_t& b) {
    /*
     * a,b should be descendant
     * [a0,a1,a2,a3]
     *   ^\ ^\ ^\
     * [b0,b1,b2,b3]
     *
     * We need:
     *  a -> [max(a0,b1),max(a1,b2),max(a2,b3),a3]
     *  b -> [b0,min(a0,b1),min(a1,b2),min(a2,b3)]
     */

    /*
     *   a   -> [a0,a1,a2,a3]
     * shl_a -> [a3,a0,a1,a2]
     * */
    float32x4_t shl_a = vextq_f32(a, a, 3);
    float32x4_t max_ab = vmaxq_f32(shl_a, b);
    float32x4_t min_ab = vminq_f32(shl_a, b);
    /*
     * shl_a -> [a3,a0,a1,a2]
     *   b   -> [b0,b1,b2,b3]
     *  min  -> [a3,min(a0,b1),min(a1,b2),min(a2,b3)]
     * Need to reverse a3 -> b0
     * */
    b = vsetq_lane_f32(vgetq_lane_f32(b, 0), min_ab, 0);
    /*
     * shl_a -> [a3,a0,a1,a2]
     *   b   -> [b0,b1,b2,b3]
     *  min  -> [b0,max(a0,b1),max(a1,b2),max(a2,b3)]
     *  ext  -> [max(a0,b1),max(a1,b2),max(a2,b3),a3]
     * */
    a = vextq_f32(max_ab, shl_a, 1);
}

static inline uint32_t aasort_is_sorted(uint32_t n, float* in) {
#if 1 // neon optimize cmp
    float32x4_t l = vld1q_f32(in);
    uint32x4_t c = {0, 0, 0, 0};

    for (uint32_t i = 4; i <= n - 4; i += 4) {
        float32x4_t r = vld1q_f32(in + i);
        uint32x4_t t = vcltq_f32(l, r);
        c = vorrq_u32(c, t);
        l = r;
    }

    return (vmaxvq_u32(c) == 0U);
#else
    uint32_t r = 1;
    for (uint32_t i = 0; r && i < (n / 4) - 1; i++) {
        const uint32_t off = 4 * i;
        r &= ((in[off] >= in[off + 4]) & (in[off + 1] >= in[off + 5]) &
              (in[off + 2] >= in[off + 6]) & (in[off + 3] >= in[off + 7]));
    }
    // r &= _mm_testc_si128(in[i + 1], _mm_max_epu32(in[i], in[i + 1]));
    return r;
#endif
}

static const float aasort_shrink_factor = 1.24733095F;
static const float aasort_shrink_factor_recp = 0.8017F;

static uint32_t aasort_in_core(uint32_t n, float32_t* in, float32_t* out) {
    /*
     * (1) Sort values within each vector in ascending order.
     *
     * NOTE: Although not explicitly stated in the paper,
     * efficient data-parallel sorting requires to rearrange
     * the data of 4 vectors, i.e., sort the first elements
     * in one vector, the second elements in the next, and so on.
     */
    for (uint32_t i = 0; i < n; i += 16) {
        float32x4x4_t batch = vld4q_f32(in + i);

        float32x4_t M0 = vmaxq_f32(batch.val[0], batch.val[1]);
        float32x4_t m0 = vminq_f32(batch.val[0], batch.val[1]);
        float32x4_t M1 = vmaxq_f32(batch.val[2], batch.val[3]);
        float32x4_t m1 = vminq_f32(batch.val[2], batch.val[3]);
        batch.val[0] = vmaxq_f32(M0, M1);
        batch.val[3] = vminq_f32(m0, m1);
        M0 = vminq_f32(M0, M1);
        m0 = vmaxq_f32(m0, m1);
        batch.val[1] = vmaxq_f32(M0, m0);
        batch.val[2] = vminq_f32(M0, m0);

        aasort_vector_transpose(batch);
        vst4q_f32(in + i, batch);
    }

    /*
     * (2) Execute combsort to sort the values into the transposed order.
     */
    uint32_t gap = (uint32_t)((n / 4) * aasort_shrink_factor_recp);
    while (gap > 1) {
        gap = (gap == 9 || gap == 10) ? 11 : gap;
        uint i = 0;
        if (gap >= 4 && (n / 4) - gap >= 4) {
            for (; i < ((n / 4) - gap) - 4; i += 4) {
                float32x4_t va = vld1q_f32(in + 4 * i);
                float32x4_t vb = vld1q_f32(in + 4 * (i + gap));
                float32x4_t va1 = vld1q_f32(in + 4 * (i + 1));
                float32x4_t vb1 = vld1q_f32(in + 4 * (i + 1 + gap));
                float32x4_t va2 = vld1q_f32(in + 4 * (i + 2));
                float32x4_t vb2 = vld1q_f32(in + 4 * (i + 2 + gap));
                float32x4_t va3 = vld1q_f32(in + 4 * (i + 3));
                float32x4_t vb3 = vld1q_f32(in + 4 * (i + 3 + gap));
                aasort_vector_cmpswap(va, vb);
                aasort_vector_cmpswap(va1, vb1);
                aasort_vector_cmpswap(va2, vb2);
                aasort_vector_cmpswap(va3, vb3);
                vst1q_f32(in + 4 * i, va);
                vst1q_f32(in + 4 * (i + gap), vb);
                vst1q_f32(in + 4 * (i + 1), va1);
                vst1q_f32(in + 4 * (i + 1 + gap), vb1);
                vst1q_f32(in + 4 * (i + 2), va2);
                vst1q_f32(in + 4 * (i + 2 + gap), vb2);
                vst1q_f32(in + 4 * (i + 3), va3);
                vst1q_f32(in + 4 * (i + 3 + gap), vb3);
            }
        }
        for (; i < ((n / 4) - gap); i++) {
            float32x4_t va = vld1q_f32(in + 4 * i);
            float32x4_t vb = vld1q_f32(in + 4 * (i + gap));
            aasort_vector_cmpswap(va, vb);
            vst1q_f32(in + 4 * i, va);
            vst1q_f32(in + 4 * (i + gap), vb);
        }

        for (i = ((n / 4) - gap); i < n / 4; i++) {
            float32x4_t va = vld1q_f32(in + 4 * i);
            float32x4_t vb = vld1q_f32(in + 4 * (i + gap - (n / 4)));
            aasort_vector_cmpswap_skew(va, vb);
            vst1q_f32(in + 4 * i, va);
            vst1q_f32(in + 4 * (i + gap - (n / 4)), vb);
        }

        gap *= aasort_shrink_factor_recp;
    }

    /*
     * As with combsort, bubblesort is executed at the end to make sure the array
     * is sorted. However, in the pre-version of the paper the authors state that
     * they have limited the number of bubblesort iterations to 10 and would fallback
     * to a vectorized merge sort if that limit would ever be reached. Here, we simply
     * output a warning and stop the execution.
     */
    do {
        do {
            float32x4_t va = vld1q_f32(in);
            for (uint i = 0; i < ((n / 4) - 1); i++) {
                float32x4_t vb = vld1q_f32(in + 4 * (i + 1));
                aasort_vector_cmpswap(va, vb);
                vst1q_f32(in + 4 * i, va);
                va = vb;
            }
            vst1q_f32(in + n - 4, va);
            float32x4_t vb = vld1q_f32(in);
            aasort_vector_cmpswap_skew(va, vb);
            vst1q_f32(in + n - 4, va);
            vst1q_f32(in, vb);
        } while (false);
    } while (!aasort_is_sorted(n, in));

/*
* (3) Reorder the values from the transposed order into the original order.
*
* For us, this also means copying the data into the output array.
*/

#if 1 // optimize memcpy
    const uint32_t gap_i = n / 4;
    for (uint32_t i = 0; i < n / 16; i++) {
        float32x4x4_t batch = vld4q_f32(in + i * 16);
        vst1q_f32(out + i * 4, batch.val[0]);
        vst1q_f32(out + i * 4 + gap_i, batch.val[1]);
        vst1q_f32(out + i * 4 + gap_i * 2, batch.val[2]);
        vst1q_f32(out + i * 4 + gap_i * 3, batch.val[3]);
    }
#else
    for (uint32_t i = 0; i < n; i += 16) {
        float32x4x4_t batch = vld4q_f32(in + i);
        aasort_vector_transpose(batch);
        vst4q_f32(in + i, batch);
    }

    for (uint32_t j = 0; j < n / 16; j++) {
        for (uint32_t i = 0; i < 4; i++) {
            memcpy(out + 4 * (i * (n / 16) + j), in + 4 * (i + j * 4), 4 * sizeof(float32_t));
        }
    }
#endif

    return 1;
}

REGISTER_VSHUFFLE_NAIVE_F32(1, 4, 5, 6);

static inline void aasort_vector_merge(float32x4_t& va, float32x4_t& vb) {
    /* COMPARE stage 1 */
    float32x4_t m = vminq_f32(va, vb);   // {m0,m1,m2,m3}
    float32x4_t M = vmaxq_f32(va, vb);   // {M0,M1,M2,M3}
    float32x4_t s0 = vextq_f32(M, M, 2); // {M2,M3,-,-}

    /* COMPARE stage 2 */
    float32x4_t mm = vminq_f32(m, s0);                            // {mm0,mm1,-,-}
    float32x4_t MM = vmaxq_f32(m, s0);                            // {MM0,MM1,-,-}
    float32x4_t s1 = vsetq_lane_f32(vgetq_lane_f32(m, 2), MM, 2); // {MM0,MM1,m2 ,-}
    float32x4_t s2 = vshuffle1456_f32(M, mm);                     // {M1,mm0,mm1,-}

    /* COMPARE stage 3 */
    float32x4_t mmm = vminq_f32(s1, s2);               // {mmm0,mmm1,mmm2,mmm3}
    float32x4_t MMM = vmaxq_f32(s1, s2);               // {MMM0,MMM1,MMM2,MMM3}
    float32x4_t s3 = vextq_f32(mmm, mmm, 3);           // {-,mmm0,mmm1,mmm2}
    vb = vsetq_lane_f32(vgetq_lane_f32(m, 3), MMM, 3); // {MMM0,MMM1,MMM2,m3}
    va = vsetq_lane_f32(vgetq_lane_f32(M, 0), s3, 0);  // {M0,mmm0,mmm1,mmm2}

    /* Transpose */
    vtrn32_fp32(va, vb);
    vtrn64_fp32(va, vb); // a:{M0,MMM0,mmm0,MMM1} b:{mmm1,MMM2,mmm2,m3}
    // M0 >= MMM0 >= mmm0 >= MMM1 >= mmm1 >= MMM2 >= mmm2 >= m3
}

void aasort_out_of_core_k(uint32_t an,
                          float32_t* a,
                          uint32_t bn,
                          float32_t* b,
                          float32_t* out,
                          const uint32_t k) {
    float32x4_t vmin = vld1q_f32(a);
    float32x4_t vmax = vld1q_f32(b);
    uint32_t ap = 4, bp = 4, op = 0;

    while (ap < an && bp < bn && op < k) {
        aasort_vector_merge(vmax, vmin);
        vst1q_f32(out + op, vmax);
        op += 4;
        if (a[ap] >= b[bp]) {
            vmax = vld1q_f32(a + ap);
            ap += 4;
        } else {
            vmax = vld1q_f32(b + bp);
            bp += 4;
        }
    }

    if (op >= k) {
        return;
    }

    if (ap < an) {
        aasort_vector_merge(vmax, vmin);
        vst1q_f32(out + op, vmax);
        op += 4;
        while (ap < an && op < k) {
            vmax = vld1q_f32(a + ap);
            ap += 4;
            aasort_vector_merge(vmax, vmin);
            vst1q_f32(out + op, vmax);
            op += 4;
        }
    } else if (bp < bn) {
        aasort_vector_merge(vmax, vmin);
        vst1q_f32(out + op, vmax);
        op += 4;
        while (bp < bn && op < k) {
            vmax = vld1q_f32(b + bp);
            bp += 4;
            aasort_vector_merge(vmax, vmin);
            vst1q_f32(out + op, vmax);
            op += 4;
        }
    } else {
        aasort_vector_merge(vmax, vmin);
        vst1q_f32(out + op, vmax);
        op += 4;
    }

    vst1q_f32(out + op, vmin);
}

#define AlignN(x, k) (((x - 1) / k + 1) * k)

void aasort_k(uint32_t n, float32_t* in, float32_t* out, uint32_t k) {
    /*
     * (1) Divide all of the data to be sorted into blocks that
     * fit in the cache or the local memory of the processor.
     */
    // As stated in the paper, we use half the L2 cache as block size.
    if (!l2_cache)
        l2_cache = ( uint32_t )sysconf(_SC_LEVEL2_CACHE_SIZE);
    uint32_t block_size = l2_cache / 2;
    uint32_t block_elements = block_size / 4;
    if (k) {
        block_elements = std::min(AlignN(k, 16), block_elements);
    }

    std::vector< float32_t > tmp_out(n + AlignN(k, 16));

    for (uint32_t i = 0; i < n; i += block_elements) {
        uint32_t m = std::min(n - i, block_elements);

        /*
         * (2) Sort each block with the in-core sorting algorithm.
         */
        if (!aasort_in_core(m, in + i, tmp_out.data() + i))
            return;
    }

    /*
     * (3) Merge the sorted blocks with the out-of-core sorting algorithm.
     */
    int currently_in_in = 0;
    float32_t* tin = tmp_out.data();
    float32_t* tout = in;
    const uint32_t align_4k = AlignN(k, 4);
    uint32_t last_n = n;

    while (block_elements < last_n) {
        uint32_t op = 0;
        for (uint32_t i = 0; i < last_n; i += block_elements * 2) {
            if (last_n - i <= block_elements) {
                // Last block? Merge into last block
                aasort_out_of_core_k(
                    align_4k, tout + op - align_4k, last_n - i, tin + i, &tmp_out[n], align_4k);
                memcpy(tout + op - align_4k, &tmp_out[n], align_4k * 4);
            } else {
                // Merge two blocks.
                uint32_t an = block_elements;
                uint32_t bn = std::min(last_n - (i + block_elements), block_elements);
                aasort_out_of_core_k(
                    an, tin + i, bn, tin + i + block_elements, tout + op, align_4k);
                op += align_4k;
            }
        }
        last_n = op;

        // block_elements *= 2;

        if (currently_in_in) {
            tin = tmp_out.data();
            tout = in;
        } else {
            tin = in;
            tout = tmp_out.data();
        }
        currently_in_in = !currently_in_in;
    }

    if (currently_in_in) {
        memcpy(out, in, k * 4);
    } else {
        memcpy(out, tmp_out.data(), k * 4);
    }
}

int main(int argc, char const* argv[]) {
    static const std::vector< int > K = {16, 32, 64, 128};
    static const std::vector< uint > N = {
        512, 1024, 65536, 1 << 10, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 40000};
    static const std::vector< uint > CACHE = {64};

    for (auto cache : CACHE) {
        l2_cache = cache * 1024;
        std::cout << "L2cache: " << l2_cache << std::endl;
        for (auto n : N) {
            std::vector< float32_t > random_data, sorted_data;
            std::vector< uint > sorted_index;
            srand(time(NULL));
            for (int i = 0; i < n; i++) {
                random_data.push_back(static_cast< float32_t >(rand() % 10000) / 1000.0F);
            }
            sorted_index.resize(n);
            std::iota(sorted_index.begin(), sorted_index.end(), 0U);
            std::sort(sorted_index.begin(),
                      sorted_index.end(),
                      [&random_data](const uint& a, const uint& b) {
                          return random_data[a] > random_data[b];
                      });
            sorted_data.resize(n);
            for (int i = 0; i < n; i++) {
                sorted_data[i] = random_data[sorted_index[i]];
            }

            for (auto k : K) {
                auto input = random_data;
                std::vector< float32_t > output(k);

                int time_aa = measure_us([&]() { aasort_k(n, input.data(), output.data(), k); }, 1);
                input = random_data;
                int time_stl = measure_us([&]() {
                    std::partial_sort(
                        input.begin(), input.begin() + k, input.end(), std::greater< float32_t >());
                });
                std::cout << "partial_sort: N-" << n << " :" << time_stl << "us" << std::endl;
                std::cout << "aa_sort: N-" << n << " K-" << k << " :" << time_aa << "us"
                          << std::endl;
                for (int i = 0; i < output.size(); i++) {
                    if (fabsf(output[i] - sorted_data[i]) > 1e-7) {
                        std::cerr << "Sort ERROR" << std::endl;
                        break;
                    }
                }
            }
        }
    }
    return 0;
}
