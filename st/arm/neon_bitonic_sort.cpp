#include "neon/neon_bitonic_sort.h"
#include "MPOpHelper.hpp"

#include "arm_neon.h"
#include "time.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <math.h>
#include <memory.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <vector>

#define vminmax_fp32x4(a, b)   \
    do {                       \
        float32x4_t tmp = a;   \
        a = vminq_f32(tmp, b); \
        b = vmaxq_f32(tmp, b); \
    } while (0)

#define vmaxmin_fp32x4(a, b)   \
    do {                       \
        float32x4_t tmp = a;   \
        a = vmaxq_f32(tmp, b); \
        b = vminq_f32(tmp, b); \
    } while (0)

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

// /*
// 读写密集问题
// 寄存器利用  比如函数内实现两次
// 分块数据预取 提高cache
// tiling
// 尽量减少分支
// 排序函数的调研
// 主要函数的汇编实现

// - OP组织框架
// - 单元测试
// */

static inline void bitonic_fp32_resort_decrese_4(float32x4_t& a, float32x4_t& b) {
    /* Start with bitonic sequence (a & b):
     * +    input a    +   +    input b    +
     * +---+---+---+---+   +---+---+---+---+
     * | d | c | b | a |   | e | f | g | h |
     * +---+---+---+---+   +---+---+---+---+
     */

    vmaxmin_fp32x4(a, b);
    /* First vmaxmin(a,b): Got two bitonic sequences a,b; Each ai >= bj
     * +---+---+---+---+
     * | e | f | g | h |  as a
     * +---+---+---+---+
     * +---+---+---+---+
     * | d | c | b | a |  as b
     * +---+---+---+---+
     */
    vtrn64_fp32(a, b);
    vmaxmin_fp32x4(a, b);
    /* Continue split bitonic (a) & (b). We now have:
     * +---+---+---+---+
     * | g | h | d | c |  as a
     * +---+---+---+---+
     * +---+---+---+---+
     * | e | f | b | a |  as b
     * +---+---+---+---+
     *
     * Elements: a_low > b_low > a_high > b_high
     */
    vtrn32_fp32(a, b);
    vmaxmin_fp32x4(a, b);
    /* Finally we compare each pair in a_low,b_low,a_high,b_high from last step.
     * Then we get sorted sequence in transpose order.
     * +---+---+---+---+
     * | h | f | d | b |  as a
     * +---+---+---+---+
     * +---+---+---+---+
     * | g | e | c | a |  as b
     * +---+---+---+---+
    */
    float32x4x2_t vzip_tmp = vzipq_f32(a, b);
    a = vzip_tmp.val[0];
    b = vzip_tmp.val[1];
    /* Since we now have separate vectors of odd and even lanes, we can vzip to original
     * positioning.
     * Now a0>=a1>=a2>=a3>=b0>=b1>=b2>=b3
     */
    return;
}

static inline void bitonic_fp32_resort_increse_4(float32x4_t& a, float32x4_t& b) {
    /* Start with bitonic sequence (a & b):
     * +    input a    +   +    input b    +
     * +---+---+---+---+   +---+---+---+---+
     * | d | c | b | a |   | e | f | g | h |
     * +---+---+---+---+   +---+---+---+---+
     *
     * +     idx a     +   +    input b    +
     * +---+---+---+---+   +---+---+---+---+
     * | d | c | b | a |   | e | f | g | h |
     * +---+---+---+---+   +---+---+---+---+
     */

    vminmax_fp32x4(a, b);
    /* First vminmax(a,b): Got two bitonic sequences a,b; Each ai >= bj
     * +---+---+---+---+
     * | d | c | b | a |  as a
     * +---+---+---+---+
     * +---+---+---+---+
     * | e | f | g | h |  as b
     * +---+---+---+---+
     */
    vtrn64_fp32(a, b);
    vminmax_fp32x4(a, b);
    /* Continue split bitonic (a) & (b). We now have:
     * +---+---+---+---+
     * | b | a | e | f |  as a
     * +---+---+---+---+
     * +---+---+---+---+
     * | d | c | g | h |  as b
     * +---+---+---+---+
     *
     * Elements: a_low < b_low < a_high < b_high
     */
    vtrn32_fp32(a, b);
    vminmax_fp32x4(a, b);
    /* Finally we compare each pair in a_low,b_low,a_high,b_high from last step.
     * Then we get sorted sequence in transpose order.
     * +---+---+---+---+
     * | a | c | e | g |  as a
     * +---+---+---+---+
     * +---+---+---+---+
     * | b | d | f | h |  as b
     * +---+---+---+---+
    */
    float32x4x2_t vzip_tmp = vzipq_f32(a, b);
    a = vzip_tmp.val[0];
    b = vzip_tmp.val[1];
    /* Since we now have separate vectors of odd and even lanes, we can vzip to original
     * positioning.
     * Now a0<=a1<=a2<=a3<=b0<=b1<=b2<=b3
     */
    return;
}

static inline void bitonic_fp32_resort_4(float32_t* a, float32_t* b, bool decrease = true) {
    float32x4_t va = vld1q_f32(a);
    float32x4_t vb = vld1q_f32(b);
    if (decrease) {
        bitonic_fp32_resort_decrese_4(va, vb);
    } else {
        bitonic_fp32_resort_increse_4(va, vb);
    }
    vst1q_f32(a, va);
    vst1q_f32(b, vb);
    return;
}

static inline void bitonic_fp32_merge_4(float32_t* a, float32_t* b, bool decrease = true) {
    // ASSERT(k > 4);
    float32x4_t va = vld1q_f32(a);
    float32x4_t vb = vld1q_f32(b);
    if (decrease) {
        vmaxmin_fp32x4(va, vb);
    } else {
        vminmax_fp32x4(va, vb);
    }
    vst1q_f32(a, va);
    vst1q_f32(b, vb);

    /* TODO: test
    val[]
    return val[increase];
    */
    return;
}

static inline void bitonic_fp32_merge_4x2(float32_t* a, float32_t* b, bool decrease = true) {
    // ASSERT(k > 4);
    float32x4_t va1 = vld1q_f32(a);
    float32x4_t va2 = vld1q_f32(a + 4);
    float32x4_t vb1 = vld1q_f32(b);
    float32x4_t vb2 = vld1q_f32(b + 4);

#if 1
    do {
        float32x4x2_t tmp1, tmp2;
        tmp1.val[0] = vmaxq_f32(va1, vb1);
        tmp1.val[1] = vminq_f32(va1, vb1);
        tmp2.val[0] = vmaxq_f32(va2, vb2);
        tmp2.val[1] = vminq_f32(va2, vb2);

        va1 = tmp1.val[decrease ^ 1];
        vb1 = tmp1.val[decrease & 1];
        va2 = tmp2.val[decrease ^ 1];
        vb2 = tmp2.val[decrease & 1];
    } while (0);
#else
    if (decrease) {
        vmaxmin_fp32x4(va1, vb1);
        vmaxmin_fp32x4(va2, vb2);
    } else {
        vminmax_fp32x4(va1, vb1);
        vminmax_fp32x4(va2, vb2);
    }
#endif

    vst1q_f32(a, va1);
    vst1q_f32(a + 4, va2);
    vst1q_f32(b, vb1);
    vst1q_f32(b + 4, vb2);
    return;
}

/*
static inline void bitonic_fp32_create_4_inplace(float32_t* data, uint data_size) {
#pragma unroll
    for (uint data_batch = 0; data_batch < data_size / 32; data_batch++) {
        float32_t* data_offset = data + data_batch * 32;
        float32x4_t vdec1 = vld1q_f32(data_offset);
        float32x4_t vdec2 = vld1q_f32(data_offset + 8);
        float32x4_t vdec3 = vld1q_f32(data_offset + 16);
        float32x4_t vdec4 = vld1q_f32(data_offset + 24);
        float32x4_t vinc1 = vld1q_f32(data_offset + 4);
        float32x4_t vinc2 = vld1q_f32(data_offset + 12);
        float32x4_t vinc3 = vld1q_f32(data_offset + 20);
        float32x4_t vinc4 = vld1q_f32(data_offset + 28);
        vtrn64_fp32(vdec1, vdec3);
        vtrn64_fp32(vdec2, vdec4);
        vtrn64_fp32(vinc1, vinc3);
        vtrn64_fp32(vinc2, vinc4);
        vtrn32_fp32(vdec1, vdec2);
        vtrn32_fp32(vdec3, vdec4);
        vtrn32_fp32(vinc1, vinc2);
        vtrn32_fp32(vinc3, vinc4);
        vst1q_f32(data_offset, vdec1);
        vst1q_f32(data_offset + 8, vdec2);
        vst1q_f32(data_offset + 16, vdec3);
        vst1q_f32(data_offset + 24, vdec4);
        vst1q_f32(data_offset + 4, vinc1);
        vst1q_f32(data_offset + 12, vinc2);
        vst1q_f32(data_offset + 20, vinc3);
        vst1q_f32(data_offset + 28, vinc4);

        vmaxmin_fp32x4(vdec1, vdec2);
        vmaxmin_fp32x4(vdec3, vdec4);
        vminmax_fp32x4(vinc1, vinc2);
        vminmax_fp32x4(vinc3, vinc4);

        vmaxmin_fp32x4(vdec2, vdec3);
        vmaxmin_fp32x4(vdec1, vdec4);
        vminmax_fp32x4(vinc2, vinc3);
        vminmax_fp32x4(vinc1, vinc4);

        vmaxmin_fp32x4(vdec1, vdec2);
        vmaxmin_fp32x4(vdec3, vdec4);
        vminmax_fp32x4(vinc1, vinc2);
        vminmax_fp32x4(vinc3, vinc4);
        vst1q_f32(data_offset, vdec1);
        vst1q_f32(data_offset + 8, vdec2);
        vst1q_f32(data_offset + 16, vdec3);
        vst1q_f32(data_offset + 24, vdec4);
        vst1q_f32(data_offset + 4, vinc1);
        vst1q_f32(data_offset + 12, vinc2);
        vst1q_f32(data_offset + 20, vinc3);
        vst1q_f32(data_offset + 28, vinc4);

        vtrn32_fp32(vdec1, vdec2);
        vtrn32_fp32(vdec3, vdec4);
        vtrn32_fp32(vinc1, vinc2);
        vtrn32_fp32(vinc3, vinc4);
        vtrn64_fp32(vdec1, vdec3);
        vtrn64_fp32(vdec2, vdec4);
        vtrn64_fp32(vinc1, vinc3);
        vtrn64_fp32(vinc2, vinc4);
        vst1q_f32(data_offset, vdec1);
        vst1q_f32(data_offset + 8, vdec2);
        vst1q_f32(data_offset + 16, vdec3);
        vst1q_f32(data_offset + 24, vdec4);
        vst1q_f32(data_offset + 4, vinc1);
        vst1q_f32(data_offset + 12, vinc2);
        vst1q_f32(data_offset + 20, vinc3);
        vst1q_f32(data_offset + 28, vinc4);
    }
}
*/

static inline void bitonic_fp32_create_8(const float32_t* data, uint data_size, float32_t* out) {
    uint data_batch = 0;
/* Tell compiler this loop can be expanded */
#pragma unroll
    /* Make each fp32x32 batch into 8-bitonic_seq x4 */
    for (; data_batch < data_size / 32; data_batch++) {
        const uint offset = data_batch << 5;
        float32x4_t vdec1 = vld1q_f32(data + offset);
        float32x4_t vdec2 = vld1q_f32(data + offset + 8);
        float32x4_t vdec3 = vld1q_f32(data + offset + 16);
        float32x4_t vdec4 = vld1q_f32(data + offset + 24);
        float32x4_t vinc1 = vld1q_f32(data + offset + 4);
        float32x4_t vinc2 = vld1q_f32(data + offset + 12);
        float32x4_t vinc3 = vld1q_f32(data + offset + 20);
        float32x4_t vinc4 = vld1q_f32(data + offset + 28);
        /* Transpose dec and inc
         * Dec:                 Inc:
         * +---+---+---+---+     +---+---+---+---+
         * | 1 | 2 | 3 | 4 |     | 5 | 6 | 7 | 8 |
         * +---+---+---+---+     +---+---+---+---+
         * | 9 | 10| 11| 12|     | 13| 14| 15| 16|
         * +---+---+---+---+     +---+---+---+---+
         * | 17| 18| 19| 20|     | 21| 22| 23| 24|
         * +---+---+---+---+     +---+---+---+---+
         * | 25| 26| 27| 28|     | 29| 30| 31| 32|
         * +---+---+---+---+     +---+---+---+---+
         * To:
         * Dec:                 Inc:
         * +---+---+---+---+     +---+---+---+---+
         * | 1 | 9 | 17| 25|     | 5 | 13| 21| 29|
         * +---+---+---+---+     +---+---+---+---+
         * | 2 | 10| 18| 26|     | 6 | 14| 22| 30|
         * +---+---+---+---+     +---+---+---+---+
         * | 3 | 11| 19| 27|     | 7 | 15| 23| 31|
         * +---+---+---+---+     +---+---+---+---+
         * | 4 | 12| 20| 28|     | 8 | 16| 24| 32|
         * +---+---+---+---+     +---+---+---+---+
         * so that we can sort 4 4_sequence at a time
         */
        vtrn64_fp32(vdec1, vdec3);
        vtrn64_fp32(vdec2, vdec4);
        vtrn64_fp32(vinc1, vinc3);
        vtrn64_fp32(vinc2, vinc4);
        vtrn32_fp32(vdec1, vdec2);
        vtrn32_fp32(vdec3, vdec4);
        vtrn32_fp32(vinc1, vinc2);
        vtrn32_fp32(vinc3, vinc4);
        /* Comp and swap:
         * Dec:              Comparison:
         * +---+---+---+---+
         * | 1 | 9 | 17| 25| -  -    -
         * +---+---+---+---+ |  |    |
         * | 2 | 10| 18| 26| -  | -  -
         * +---+---+---+---+    | |
         * | 3 | 11| 19| 27| -  | -  -
         * +---+---+---+---+ |  |    |
         * | 4 | 12| 20| 28| -  -    -
         * +---+---+---+---+
         * For dec use maxmin, for inc use minmax.
         * After comparison, we get Dec and Inc like:
         * Dec:                 Inc:
         * +---+---+---+---+     +---+---+---+---+
         * | 4 | 12| 20| 28|     | 5 | 13| 21| 29|
         * +---+---+---+---+     +---+---+---+---+
         * | 3 | 11| 19| 27|     | 6 | 14| 22| 30|
         * +---+---+---+---+     +---+---+---+---+
         * | 2 | 10| 18| 26|     | 7 | 15| 23| 31|
         * +---+---+---+---+     +---+---+---+---+
         * | 1 | 9 | 17| 25|     | 8 | 16| 24| 32|
         * +---+---+---+---+     +---+---+---+---+
         */
        vmaxmin_fp32x4(vdec1, vdec2);
        vmaxmin_fp32x4(vdec3, vdec4);
        vminmax_fp32x4(vinc1, vinc2);
        vminmax_fp32x4(vinc3, vinc4);

        vmaxmin_fp32x4(vdec2, vdec3);
        vmaxmin_fp32x4(vdec1, vdec4);
        vminmax_fp32x4(vinc2, vinc3);
        vminmax_fp32x4(vinc1, vinc4);

        vmaxmin_fp32x4(vdec1, vdec2);
        vmaxmin_fp32x4(vdec3, vdec4);
        vminmax_fp32x4(vinc1, vinc2);
        vminmax_fp32x4(vinc3, vinc4);
        /* re-Transpose dec and inc, and restore data into its original position
         * then we get data like:
         * From Dec:         From Inc:
         * +---+---+---+---+ +---+---+---+---+
         * | 4 | 3 | 2 | 1 | | 5 | 6 | 7 | 8 |
         * +---+---+---+---+ +---+---+---+---+
         * | 12| 11| 10| 9 | | 13| 14| 15| 16|
         * +---+---+---+---+ +---+---+---+---+
         * | 20| 19| 18| 17| | 21| 22| 23| 24|
         * +---+---+---+---+ +---+---+---+---+
         * | 28| 27| 26| 25| | 29| 30| 31| 32|
         * +---+---+---+---+ +---+---+---+---+
         */
        vtrn32_fp32(vdec1, vdec2);
        vtrn32_fp32(vdec3, vdec4);
        vtrn32_fp32(vinc1, vinc2);
        vtrn32_fp32(vinc3, vinc4);
        vtrn64_fp32(vdec1, vdec3);
        vtrn64_fp32(vdec2, vdec4);
        vtrn64_fp32(vinc1, vinc3);
        vtrn64_fp32(vinc2, vinc4);
        vst1q_f32(out + offset, vdec1);
        vst1q_f32(out + offset + 4, vinc1);
        vst1q_f32(out + offset + 8, vdec2);
        vst1q_f32(out + offset + 12, vinc2);
        vst1q_f32(out + offset + 16, vdec3);
        vst1q_f32(out + offset + 20, vinc3);
        vst1q_f32(out + offset + 24, vdec4);
        vst1q_f32(out + offset + 28, vinc4);
    }

    for (uint offset = data_batch << 5; offset < data_size; offset += 8) {
        uint decrease_size = std::min(data_size - offset, 4u);
        std::partial_sort_copy(data + offset,
                               data + offset + decrease_size,
                               out + offset,
                               out + offset + decrease_size,
                               std::greater< float >());
        uint increase_size = std::min(data_size - offset, 8u);
        std::partial_sort_copy(data + offset + decrease_size,
                               data + offset + increase_size,
                               out + offset + decrease_size,
                               out + offset + increase_size);
    }
    return;
}

static inline void
bitonic_fp32_sort(const float32_t* data, const uint data_size, const uint k, float32_t* out) {
    const uint align_data_size = RoundUp2N(data_size);
    const uint align_k = RoundUp2N(k);
    std::vector< float32_t > data_align;
    data_align.reserve(align_data_size);
    data_align.resize(data_size);
    data_align.resize(align_data_size, std::numeric_limits< float32_t >::lowest());
    // Print(data_tmp, data_size, "create_8");
    /* Initial data to 8-bitonic-sequences */
    bitonic_fp32_create_8(data, data_size, &data_align[0]);
    float32_t* data_tmp = &data_align[0];
    {
        // Specify func for inc=4 && inc=2
        uint inc = 4;
        for (uint low = 0; low < align_data_size; low += 8) {
            bool reverse = (8 & low) == 0;
            bitonic_fp32_resort_4(data_tmp + low, data_tmp + low + inc, reverse);
        }
        /*
         * We got 16-bitonic-sequences like
         * \\\\\\\\////////  \\\\\\\\////////
         */
    }

    for (uint half_bitonic_seq_size = 8; half_bitonic_seq_size < align_k;
         half_bitonic_seq_size <<= 1) {
        // Generate 2*i-bitonic-sequence from i-bitonic-sequence.
        uint bitonic_seq_size = half_bitonic_seq_size << 1;
        for (uint inc = half_bitonic_seq_size; inc > 4; inc >>= 1) {
            for (uint tid = 0; tid < align_data_size / 2; tid += 8) {
                uint low = (tid << 1) - (tid & inc - 1);
                bool reverse = (bitonic_seq_size & low) == 0;
                bitonic_fp32_merge_4x2(data_tmp + low, data_tmp + low + inc, reverse);
            }
        }
        {
            // Specify func for inc=4 && inc=2
            uint inc = 4;
            for (uint tid = 0; tid < align_data_size / 2; tid += 4) {
                uint low = (tid << 1) - (tid & inc - 1);
                bool reverse = (bitonic_seq_size & low) == 0;
                bitonic_fp32_resort_4(data_tmp + low, data_tmp + low + inc, reverse);
            }
        }

        // char buf[64];
        // sprintf(buf, "Merge to %u", bitonic_seq_size);
        // Print(data_tmp, data_size, buf);
    }

#if 0
    //======================================== otm2
    {
        uint remain_datasize = align_data_size;
        const uint half_bitonic_seq_size = align_k; // constraint k
        const uint bitonic_seq_size = align_k << 1;

        while (remain_datasize > align_k) {
            {
                uint idx = 0;
                const uint inc = align_k;
                for (unsigned int tid = 0; tid < (remain_datasize >> 1); tid += 8) {
                    unsigned int low = (tid << 1) - (tid & (inc - 1));
                    float32x4_t va1 = vld1q_f32(data_tmp + low);
                    float32x4_t vb1 = vld1q_f32(data_tmp + low + inc);
                    float32x4_t va2 = vld1q_f32(data_tmp + low + 4);
                    float32x4_t vb2 = vld1q_f32(data_tmp + low + inc + 4);

                    float32x4_t vmax1, vmax2;
                    vmax1 = vmaxq_f32(va1, vb1);
                    vmax2 = vmaxq_f32(va2, vb2);

                    vst1q_f32(data_tmp + idx, vmax1);
                    vst1q_f32(data_tmp + idx + 4, vmax2);
                    idx += 8;
                }
                remain_datasize >>= 1;
            }

            for (unsigned int bitonic_seq_size = align_k; bitonic_seq_size <= align_k * 2;
                 bitonic_seq_size <<= 1) {
                for (unsigned int inc = bitonic_seq_size >> 1; inc > 4; inc >>= 1) {
                    for (unsigned int tid = 0; tid < (remain_datasize >> 1); tid += 8) {
                        uint low = (tid << 1) - (tid & (inc - 1));
                        bool reverse = (bitonic_seq_size & low) == 0;
                        bitonic_fp32_merge_4x2(data_tmp + low, data_tmp + low + inc, reverse);
                    }
                }

                {
                    // Specify func for inc=4 && inc=2
                    uint inc = 4;
                    for (uint tid = 0; tid < remain_datasize / 2; tid += 4) {
                        uint low = (tid << 1) - (tid & inc - 1);
                        bool reverse = (bitonic_seq_size & low) == 0;
                        bitonic_fp32_resort_4(data_tmp + low, data_tmp + low + inc, reverse);
                    }
                }
            }

            remain_datasize >>= 1;
        }

        {
            const unsigned int bitonic_seq_size = align_k;
            for (unsigned int inc = bitonic_seq_size >> 1; inc > 4; inc >>= 1) {
                for (unsigned int tid = 0; tid < (remain_datasize >> 1); tid += 8) {
                    uint low = (tid << 1) - (tid & (inc - 1));
                    bitonic_fp32_merge_4x2(data_tmp + low, data_tmp + low + inc);
                }
            }
            {
                // Specify func for inc=4 && inc=2
                uint inc = 4;
                for (uint tid = 0; tid< remain_datasize >> 2; tid += 4) {
                    uint low = (tid << 1) - (tid & inc - 1);
                    bitonic_fp32_resort_4(data_tmp + low, data_tmp + low + inc);
                }
            }
        }

        // now data[0~k] should be decreased
        memcpy(out, data_tmp, k * sizeof(float));
        return;
    }
#else
    //=========================== origin

    // Got (n/2k) bitonic sequences of length 2k
    uint idx = 0;
    uint bitonic_offset = 2 * align_k;
    uint remain_datasize = align_data_size;
    uint half_bitonic_seq_size = align_k; // constraint k
    uint bitonic_seq_size = half_bitonic_seq_size << 1;
    while (remain_datasize >= 2 * align_k) {
        for (uint bitonic_seq_low = 0; bitonic_seq_low < align_data_size;
             bitonic_seq_low += bitonic_offset) {
            for (uint tid = 0; tid < align_k; tid += 8) {
                uint low = bitonic_seq_low + tid;
                bitonic_fp32_merge_4x2(data_tmp + low, data_tmp + low + bitonic_offset / 2, true);
            }
        }

        for (uint bitonic_seq_low = 0; bitonic_seq_low < align_data_size;
             bitonic_seq_low += bitonic_offset) {
            for (uint inc = align_k >> 1; inc > 4; inc >>= 1) {
                for (uint j = 0; j < half_bitonic_seq_size / inc / 2; j++) {
                    for (uint tid = 0; tid < inc; tid += 8) {
                        uint low = bitonic_seq_low + j * inc * 2 + tid;
                        bool decrease = ((bitonic_seq_low / bitonic_offset) & 1) == 0;
                        bitonic_fp32_merge_4x2(data_tmp + low, data_tmp + low + inc, decrease);
                    }
                }
            }
            {
                // Specify func for inc=4 && inc=2
                for (uint tid = 0; tid < align_k; tid += 8) {
                    uint low = bitonic_seq_low + tid;
                    bool decrease = ((bitonic_seq_low / bitonic_offset) & 1) == 0;
                    bitonic_fp32_resort_4(data_tmp + low, data_tmp + low + 4, decrease);
                }
            }
        }
        bitonic_offset <<= 1;
        remain_datasize >>= 1;
    }
    // now data[0~k] should be decreased
    memcpy(out, data_tmp, k * sizeof(float));
    return;
#endif
}

static inline void
bitonic_fp32_sort_k16(const float32_t* data, const uint data_size, const uint k, float32_t* out) {
    const uint align_data_size = RoundUp2N(data_size);
    const uint align_k = 16;
    std::vector< float32_t > data_align;
    data_align.reserve(align_data_size);
    data_align.resize(data_size);
    data_align.resize(align_data_size, std::numeric_limits< float32_t >::lowest());
    /* Initial data to 8-bitonic-sequences */
    bitonic_fp32_create_8(data, data_size, &data_align[0]);
    float32_t* data_tmp = &data_align[0];
    // Print(data_tmp, data_size, "create_8");
    /*
     * We got 8-bitonic-sequences like
     * \\\\//// \\\\//// \\\\//// \\\\////
     */
    {
        // Specify func for inc=4 && inc=2
        const uint inc = 4;
        for (uint low = 0; low < align_data_size; low += 8) {
            const bool reverse = (8 & low) == 0;
            bitonic_fp32_resort_4(data_tmp + low, data_tmp + low + inc, reverse);
        }
    }
    /*
     * We got 16-bitonic-sequences like
     * \\\\\\\\////////  \\\\\\\\////////
     */
    {
        const uint half_bitonic_seq_size = 8;
        // Generate 2*i-bitonic-sequence from i-bitonic-sequence.
        const uint bitonic_seq_size = 16;
        const uint inc = 8;
        for (uint tid = 0; tid < align_data_size / 2; tid += 8) {
            uint low = (tid << 1) - (tid & inc - 1);
            bool reverse = (bitonic_seq_size & low) == 0;
            bitonic_fp32_merge_4x2(data_tmp + low, data_tmp + low + inc, reverse);
        }
    }
    {
        // Specify func for inc=4 && inc=2
        const uint inc = 4;
        for (uint low = 0; low < align_data_size; low += 8) {
            const bool reverse = (16 & low) == 0;
            bitonic_fp32_resort_4(data_tmp + low, data_tmp + low + inc, reverse);
        }
    }

    // char buf[64];
    // sprintf(buf, "Merge to %u", bitonic_seq_size);
    // Print(data_tmp, data_size, buf);

    // Got (n/2k) bitonic sequences of length 2k
    uint idx = 0;
    uint bitonic_offset = 2 * align_k;
    uint remain_datasize = align_data_size;
    uint half_bitonic_seq_size = align_k;
    uint bitonic_seq_size = half_bitonic_seq_size << 1;
    while (remain_datasize >= 2 * align_k) {
        for (uint bitonic_seq_low = 0; bitonic_seq_low < align_data_size;
             bitonic_seq_low += bitonic_offset) {
            bitonic_fp32_merge_4x2(
                data_tmp + bitonic_seq_low, data_tmp + bitonic_seq_low + bitonic_offset / 2, true);
            bitonic_fp32_merge_4x2(data_tmp + bitonic_seq_low + 8,
                                   data_tmp + bitonic_seq_low + 8 + bitonic_offset / 2,
                                   true);

            {
                uint low = bitonic_seq_low;
                bool decrease = ((bitonic_seq_low / bitonic_offset) & 1) == 0;
                bitonic_fp32_merge_4x2(data_tmp + low, data_tmp + low + 8, decrease);
            }
            {
                // Specify func for inc=4 && inc=2
                {
                    uint low = bitonic_seq_low;
                    bool decrease = ((bitonic_seq_low / bitonic_offset) & 1) == 0;
                    bitonic_fp32_resort_4(data_tmp + low, data_tmp + low + 4, decrease);
                    bitonic_fp32_resort_4(data_tmp + low + 8, data_tmp + low + 12, decrease);
                }
            }
        }

        bitonic_offset <<= 1;
        remain_datasize >>= 1;
    }

    // now data[0~k] should be decreased
    memcpy(out, data_tmp, k * sizeof(float));
    return;
}

bool MPOpNeonBitonicSort::Run(const Blobs& inputs, Blobs& output) {
    std::vector< float32_t > sort_data(k_);
    const auto& input = inputs.find("input")->second;
    bitonic_fp32_sort_k16(&input[0], input.size(), k_, &sort_data[0]);
    output["TopK"];
    output["TopK"].swap(sort_data);
    return true;
}

static inline void
cpu_bitonic_fp32_sort(const float32_t* data, const uint data_size, const uint k, float32_t* out) {
    const uint align_data_size = RoundUp2N(data_size);
    const uint align_k = RoundUp2N(k);
    std::vector< float32_t > data_align;
    data_align.reserve(align_data_size);
    data_align.resize(data_size);
    data_align.resize(align_data_size, std::numeric_limits< float32_t >::lowest());
    memcpy(&data_align[0], data, data_size * sizeof(float32_t));

    for (unsigned int bitonic_seq_size = 2; bitonic_seq_size <= align_k * 2;
         bitonic_seq_size <<= 1) {
        for (unsigned int inc = bitonic_seq_size >> 1; inc > 0; inc >>= 1) {
            for (unsigned int tid = 0; tid < (align_data_size >> 1); tid++) {
                unsigned int low = (tid << 1) - (tid & (inc - 1));
                bool reverse = (bitonic_seq_size & low) == 0;
                bool to_swap = data[low] < data[low + inc];
                if (to_swap ^ reverse) {
                    std::swap(data_align[low], data_align[low + inc]);
                }
            }
        }
    }

    uint remain_datasize = align_data_size;
    const uint half_bitonic_seq_size = align_k; // constraint k
    const uint bitonic_seq_size = align_k << 1;

    while (remain_datasize > align_k) {
        {
            const uint inc = align_k;
            uint idx = 0;
            for (unsigned int tid = 0; tid < (remain_datasize >> 1); tid++) {
                unsigned int low = (tid << 1) - (tid & (inc - 1));
                data_align[idx++] = std::max(data_align[low], data_align[low + inc]);
            }
            remain_datasize >>= 1;
        }

        for (unsigned int bitonic_seq_size = align_k; bitonic_seq_size <= align_k * 2;
             bitonic_seq_size <<= 1) {
            for (unsigned int inc = bitonic_seq_size >> 1; inc > 0; inc >>= 1) {
                for (unsigned int tid = 0; tid < (remain_datasize >> 1); tid++) {
                    unsigned int low = (tid << 1) - (tid & (inc - 1));
                    bool reverse = (bitonic_seq_size & low) == 0;
                    bool to_swap = data[low] < data[low + inc];
                    if (to_swap ^ reverse) {
                        std::swap(data_align[low], data_align[low + inc]);
                    }
                }
            }
        }

        remain_datasize >>= 1;
    }

    {
        const unsigned int bitonic_seq_size = align_k;
        for (unsigned int inc = bitonic_seq_size >> 1; inc > 0; inc >>= 1) {
            for (unsigned int tid = 0; tid < (remain_datasize >> 1); tid++) {
                unsigned int low = (tid << 1) - (tid & (inc - 1));
                bool reverse = (bitonic_seq_size & low) == 0;
                bool to_swap = data[low] < data[low + inc];
                if (to_swap ^ reverse) {
                    std::swap(data_align[low], data_align[low + inc]);
                }
            }
        }
    }

    // now data[0~k] should be decreased
    memcpy(out, &data_align[0], k * sizeof(float));
    return;
}

void tmp_sort_index() {
    float32_t tmp[4] = {1.2345678, 1.2345678, 1.2345678, 1.2345678};
    /*
     * +---+---+---+---+
     * | a | b | c | d |     v_value
     * +---+---+---+---+
     *
     * +----+----+----+----+
     * | ia | ib | ic | id |    v_idx
     * +----+----+----+----+
     *   0     1    2    3
     *
     * set 0 -> a.low(2)
     * set 1 -> b.low(2)
     * set 2 -> c.low(2)
     * set 3 -> d.low(2)
     * +---+---+---+---+
     * |a_0|b_1|c_2|d_3|     v_value_idx
     * +---+---+---+---+
     *
     * After sort v_value_idx, dump idx out, and tbl get correspond idx.
     *
     * +---+---+---+---+
     * |d_3|c_2|b_1|a_0|     sorted_v_value_idx
     * +---+---+---+---+
     * dump idx ->
     * +---+---+---+---+
     * | 3 | 2 | 1 | 0 |     sorted_v_idx
     * +---+---+---+---+
     * tbl->
     * +----+----+----+----+
     * | id | ic | ib | ia |    sorted_v_idx
     * +----+----+----+----+
     *
     * */

    // float32x4_t v_tmp = vld1q_f32(tmp);
    // float32x4_t v_bit;
    // v_bit = vsetq_lane_u32(0U, v_bit, 0);
    // v_bit = vsetq_lane_u32(1U, v_bit, 1);
    // v_bit = vsetq_lane_u32(2U, v_bit, 2);
    // v_bit = vsetq_lane_u32(3U, v_bit, 3);
    // uint32x4_t v_mask = vdupq_n_u32(0b111);
    // v_tmp = vbslq_f32(v_mask, v_bit, v_tmp);
    // uint32x4_t v_idx = vbslq_u32(v_mask, vreinterpretq_u32_f32(v_tmp), v_bit);
    // vst1q_f32(tmp, v_tmp);
    // // std::cout << tmp[0] * 1000.F << " " << tmp[1] * 1000.F << " " << tmp[2] * 1000.F << " "
    // //           << tmp[3] * 1000.F << std::endl;
    // uint idx[4];
    // vst1q_u32(idx, v_idx);
    // // std::cout << idx[0] << " " << idx[1] << " " << idx[2] << " " << idx[3] << std::endl;

    // return 0;
}

/*
void test_bitonic_sort(float32_t* data_in, float32_t* data_out, uint32_t n) {
    using namespace std;
    static vector< float32_t > merged_0_1(n);
    static vector< float32_t > merged_2_3(n);

    constexpr int vector_width_in_bytes = 16; // quadword=128bit
    constexpr int vector_width = vector_width_in_bytes / sizeof(data_in[0]);
    int n_div_vec_width = n / vector_width;
    int log2n = static_cast< int >(log2(n_div_vec_width));
    if (exp2(log2n) != n_div_vec_width) {
        std::cerr << "error log2\n";
        return;
    }

    for (auto i = 0; i < log2n; ++i) {
        for (auto j = i; j >= 0; --j) {
            int arrow_len = 1 << j;
            for (auto k = 0; k < n_div_vec_width / 2; k++) {
                bool is_up = k & (1 << i);
                uint32_t mask = (1 << j) - 1;
                size_t upper_idx = ((k & ~mask) << 1) | (k & mask);
                size_t lower_idx = upper_idx + arrow_len;
                float32x4_t v_upper = vld1q_f32(&data_in[upper_idx * vector_width]);
                float32x4_t v_lower = vld1q_f32(&data_in[(upper_idx + arrow_len) * vector_width]);
                float32x4_t v_min = vminq_s32(v_upper, v_lower);
                float32x4_t v_max = vmaxq_s32(v_upper, v_lower);
                auto max_idx = (upper_idx + static_cast< int >(is_up) * arrow_len) * vector_width;
                auto min_idx = (upper_idx + static_cast< int >(!is_up) * arrow_len) * vector_width;
                vst1q_f32(&data_in[max_idx], v_max);
                vst1q_f32(&data_in[min_idx], v_min);
            }
        }
    }

    float32_t* to_merge[4] = {data_in, data_in + 1, data_in + 2, data_in + 3};

    merge_two_cols(to_merge[0], to_merge[1], 4, merged_0_1.data(), n >> 2);
    merge_two_cols(to_merge[2], to_merge[3], 4, merged_2_3.data(), n >> 2);

    to_merge[0] = merged_0_1.data();
    to_merge[1] = merged_2_3.data();
    merge_two_cols(to_merge[0], to_merge[1], 1, data_out, n >> 1);
}
*/

static std::vector< int > K = {16, 8, 32, 64, 128 /*, 100, 47, 30, 18, 256, 10*/};

int main(int argc, char const* argv[]) {
    const uint N = 65536;
    std::vector< float32_t > random_data, sorted_data;
    std::vector< uint > sorted_index;
    for (int i = 0; i < N; i++) {
        random_data.push_back(static_cast< float32_t >(rand() % 10000) / 1000.0F);
        // random_data.push_back(i + 1);
    }
    sorted_index.resize(N);
    std::iota(sorted_index.begin(), sorted_index.end(), 0U);
    std::sort(
        sorted_index.begin(), sorted_index.end(), [&random_data](const uint& a, const uint& b) {
            return random_data[a] > random_data[b];
        });
    sorted_data.resize(N);
    for (int i = 0; i < N; i++) {
        sorted_data[i] = random_data[sorted_index[i]];
    }

    for (const auto& k : K) {
        std::vector< uint > index(N);
        std::iota(index.begin(), index.end(), 0U);
        auto start = GetRealTimeUs();
        std::partial_sort(index.begin(),
                          index.begin() + k,
                          index.end(),
                          [&random_data](const uint& a, const uint& b) {
                              return random_data[a] > random_data[b];
                          });
        auto end = GetRealTimeUs();
        std::cout << "Partial_sort_index: K-" << k << " :" << (end - start) << "us" << std::endl;
    }
    std::cout << std::endl;

    for (const auto& k : K) {
        auto input = random_data;
        std::vector< float32_t > output(N);

        int time_bitonic =
            measure_us([&]() { bitonic_fp32_sort(input.data(), N, k, output.data()); });
        int time_stl = measure_us(
            [&]() { std::sort(input.begin(), input.end(), std::greater< float32_t >()); });
        std::cout << "std_sort_index: K-" << k << " :" << time_stl << "us" << std::endl;
        std::cout << "bitonic_sort_index: K-" << k << " :" << time_bitonic << "us" << std::endl;
        if (!is_sorted(output, std::greater< float32_t >())) {
            std::cerr << "Sort ERROR" << std::endl;
        }
    }
    // for (const auto& k : K) {
    //     auto input = random_data;
    //     std::vector< float32_t > output(k);
    //     auto start = GetRealTimeUs();
    //     bitonic_fp32_sort(&input[0], input.size(), k, &output[0]);
    //     auto end = GetRealTimeUs();
    //     std::cout << "neon_bitonic_sort: K-" << k << " :" << (end - start) << "us" << std::endl;
    //     for (size_t i = 0; i < k; i++) {
    //         if (fabsf(output[i] - sorted_data[i]) > 1e-6) {
    //             Print(&output[0], k, 16, "Error data");
    //             Print(&sorted_data[0], k, 16, "Expect data");
    //             break;
    //         }
    //     }
    // }
    /* code */
    return 0;
}
