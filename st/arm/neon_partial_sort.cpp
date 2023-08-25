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

namespace CPU_HEAP {
void cpu_adjust_heap(float32_t* __first,
                     uint __holeIndex,
                     uint __len,
                     uint __valueIndex,
                     bool __increase = true) {
    const uint __topIndex = __holeIndex;
    uint __secondChild = __holeIndex;
    float32_t __value = *(__first + __valueIndex);
    while (__secondChild < (__len - 1) / 2) {    // LOOP log(k/2)
        __secondChild = 2 * (__secondChild + 1); // lea     esi, [rbx+2+rbx]
        if ((__first[__secondChild] < __first[__secondChild - 1]) ^ __increase)
            __secondChild--;
        *(__first + __holeIndex) = *(__first + __secondChild);
        __holeIndex = __secondChild;
    }
    if ((__len & 1) == 0 && __secondChild == (__len - 2) / 2) {
        __secondChild = 2 * (__secondChild + 1);
        *(__first + __holeIndex) = *(__first + (__secondChild - 1));
        __holeIndex = __secondChild - 1;
    }
    uint __parent = (__holeIndex - 1) / 2;
    while (__holeIndex > __topIndex && ((*(__first + __parent) < __value) ^ __increase)) {
        *(__first + __holeIndex) = *(__first + __parent);
        __holeIndex = __parent;
        __parent = (__holeIndex - 1) / 2;
    }
    *(__first + __holeIndex) = __value;
}

void cpu_make_heap(float32_t* __first, float32_t* __last, bool __increase = true) {
    if (__last - __first < 2)
        return;

    const uint __len = __last - __first;
    uint __parent = (__len - 2) / 2;
    while (true) {
        cpu_adjust_heap(__first, __parent, __len, __parent, __increase);
        if (__parent == 0)
            return;
        __parent--;
    }
}

inline void
cpu_pop_heap(float32_t* __first, float32_t* __last, float32_t* __result, bool __increase = true) {
    float32_t __value = *__result;
    *__result = *__first;
    *__first = __value;
    cpu_adjust_heap(__first, 0U, __last - __first, 0, __increase);
}

void cpu_heap_select(float32_t* __first,
                     float32_t* __middle,
                     float32_t* __last,
                     bool __increase = true) {
    cpu_make_heap(__first, __middle, __increase);
    for (float32_t* __i = __middle; __i < __last; ++__i)
        if ((*__i < *__first) ^ __increase)
            cpu_pop_heap(__first, __middle, __i, __increase);
}

void cpu_sort_heap(float32_t* __first, float32_t* __last, bool __increase = true) {
    while (__last - __first > 1) {
        --__last;
        cpu_pop_heap(__first, __last, __last, __increase);
    }
}

void cpu_partial_sort(float32_t* __first,
                      float32_t* __middle,
                      float32_t* __last,
                      bool __increase = true) {
    cpu_heap_select(__first, __middle, __last, __increase);
    cpu_sort_heap(__first, __middle, __increase);
}
} // namespace CPU_HEAP

namespace CPU_HEAP_WITH_ADDITIONAL {
template < typename ADDITION_T >
void cpu_adjust_heap(float32_t* __first,
                     ADDITION_T* __firstAdditional,
                     uint __holeIndex,
                     uint __len,
                     uint __valueIndex,
                     bool __increase = true) {
    const uint __topIndex = __holeIndex;
    uint __secondChild = __holeIndex;
    float32_t __value = *(__first + __valueIndex);
    ADDITION_T __valueAdditional = *(__firstAdditional + __valueIndex);
    while (__secondChild < (__len - 1) / 2) {
        __secondChild = 2 * (__secondChild + 1);
        if ((__first[__secondChild] < __first[__secondChild - 1]) ^ __increase)
            __secondChild--;
        *(__first + __holeIndex) = *(__first + __secondChild);
        *(__firstAdditional + __holeIndex) = *(__firstAdditional + __secondChild);
        __holeIndex = __secondChild;
    }
    if ((__len & 1) == 0 && __secondChild == (__len - 2) / 2) {
        __secondChild = 2 * (__secondChild + 1);
        *(__first + __holeIndex) = *(__first + (__secondChild - 1));
        *(__firstAdditional + __holeIndex) = *(__firstAdditional + (__secondChild - 1));
        __holeIndex = __secondChild - 1;
    }
    uint __parent = (__holeIndex - 1) / 2;
    while (__holeIndex > __topIndex && ((*(__first + __parent) < __value) ^ __increase)) {
        *(__first + __holeIndex) = *(__first + __parent);
        *(__firstAdditional + __holeIndex) = *(__firstAdditional + __parent);
        __holeIndex = __parent;
        __parent = (__holeIndex - 1) / 2;
    }
    *(__first + __holeIndex) = __value;
    *(__firstAdditional + __holeIndex) = __valueAdditional;
}

template < typename ADDITION_T >
void cpu_make_heap(float32_t* __first,
                   float32_t* __last,
                   ADDITION_T* __firstAdditional,
                   bool __increase = true) {
    if (__last - __first < 2)
        return;

    const uint __len = __last - __first;
    uint __parent = (__len - 2) / 2;
    while (true) {
        cpu_adjust_heap(__first, __firstAdditional, __parent, __len, __parent, __increase);
        if (__parent == 0)
            return;
        __parent--;
    }
}

template < typename ADDITION_T >
inline void cpu_pop_heap(float32_t* __first,
                         float32_t* __last,
                         float32_t* __result,
                         ADDITION_T* __firstAdditional,
                         bool __increase = true) {
    float32_t __value = *__result;
    *__result = *__first;
    *__first = __value;
    ADDITION_T __valueAdditional = *(__firstAdditional + (__result - __first));
    *(__firstAdditional + (__result - __first)) = *__firstAdditional;
    *__firstAdditional = __valueAdditional;
    cpu_adjust_heap(__first, __firstAdditional, 0U, __last - __first, 0, __increase);
}

template < typename ADDITION_T >
void cpu_heap_select(float32_t* __first,
                     float32_t* __middle,
                     float32_t* __last,
                     ADDITION_T* __firstAdditional,
                     bool __increase = true) {
    cpu_make_heap(__first, __middle, __firstAdditional, __increase);
    for (float32_t* __i = __middle; __i < __last; ++__i)
        if ((*__i < *__first) ^ __increase)
            cpu_pop_heap(__first, __middle, __i, __firstAdditional, __increase);
}

template < typename ADDITION_T >
void cpu_sort_heap(float32_t* __first,
                   float32_t* __last,
                   ADDITION_T* __firstAdditional,
                   bool __increase = true) {
    while (__last - __first > 1) {
        --__last;
        cpu_pop_heap(__first, __last, __last, __firstAdditional, __increase);
    }
}

template < typename ADDITION_T >
void cpu_partial_sort(float32_t* __first,
                      float32_t* __middle,
                      float32_t* __last,
                      ADDITION_T* __firstAdditional,
                      bool __increase = true) {
    cpu_heap_select(__first, __middle, __last, __firstAdditional, __increase);
    cpu_sort_heap(__first, __middle, __firstAdditional, __increase);
}
}

namespace NEON_HEAP {
void neon_adjust_heap(float32_t* __first, uint __holeIndex, uint __len, uint __valueIndex) {
    const uint __topIndex = __holeIndex;
    uint __secondChild = __holeIndex;
    float32x4_t v_value = vld1q_f32(__first + __valueIndex);
    while (__secondChild / 4 < (__len - 1) / 4 / 2) {
        // std::cout << "has rchild" << std::endl;
        __secondChild = 2 * (__secondChild / 4 + 1) * 4;

        // store min value in rchild, then swap rchild and hole
        float32x4_t v_lchild = vld1q_f32(__first + __secondChild - 4);
        float32x4_t v_rchild = vld1q_f32(__first + __secondChild);
        float32x4_t v_min = vminq_f32(v_lchild, v_rchild);
        v_lchild = vmaxq_f32(v_lchild, v_rchild);
        v_rchild = v_min;
        vst1q_f32(__first + __secondChild - 4, v_lchild);
        vst1q_f32(__first + __holeIndex, v_rchild);

        __holeIndex = __secondChild;
        neon_adjust_heap(__first, __secondChild - 4, __len, __secondChild - 4);
    }
    uint __batch4_len = ((__len + 3) / 4);
    if ((__batch4_len & 1) == 0 && (__secondChild / 4) == (__batch4_len - 2) / 2) {
        // std::cout << "copy lchild" << std::endl;
        __secondChild = 2 * (__secondChild / 4 + 1) * 4 - 4;
        memcpy(__first + __holeIndex, __first + __secondChild, 4 * sizeof(float32_t));
        __holeIndex = __secondChild;
    }
    uint __parent = (__holeIndex / 4 - 1) / 2 * 4;
    // std::cout << "hole top parent" << __holeIndex << " " << __topIndex << " " << __parent
    //   << std::endl;
    while (__holeIndex > __topIndex) {
        float32x4_t v_parent = vld1q_f32(__first + __parent);
        float32x4_t v_min = vminq_f32(v_parent, v_value);
        v_parent = vmaxq_f32(v_parent, v_value);
        v_value = v_min;
        vst1q_f32(__first + __holeIndex, v_parent);
        __holeIndex = __parent;
        __parent = (__holeIndex / 4 - 1) / 2 * 4;
        // std::cout << "hole top parent" << __holeIndex << " " << __topIndex << " " << __parent
        //   << std::endl;
    }
    vst1q_f32(__first + __holeIndex, v_value);
}

void neon_make_heap(float32_t* __first, float32_t* __last) {
    if (__last - __first < 5)
        return;

    const uint __len = __last - __first;
    // (((len -1)/4 -1)/2)*4 = ((len -1)/4 -1)*2 = (len-1)/2-2
    uint __parent = (((__len - 1) / 4 - 1) / 2) * 4;
    while (true) {
        neon_adjust_heap(__first, __parent, __len, __parent);
        std::string buf = "after adjust " + std::to_string(__parent);
        // PrintBatch4(__first, __last - __first, buf.c_str());
        if (__parent == 0)
            return;
        __parent -= 4;
    }
}

inline void neon_pop_heap(float32_t* __first, float32_t* __last, float32_t* __result) {
    static const uint batch_size = 4 * sizeof(float32_t);
    float32_t tmp[4];
    memcpy(tmp, __result, batch_size);
    memcpy(__result, __first, batch_size);
    memcpy(__first, tmp, batch_size);
    neon_adjust_heap(__first, 0U, __last - __first, 0);
}

void neon_heap_select(float32_t* __first, float32_t* __middle, float32_t* __last) {
    // ASSERT middle - first == 4k
    // ASSERT last - middle >= 4 || last == middle
    neon_make_heap(__first, __middle);
    // PrintBatch4(__first, __last - __first, "after make heap");
    float32_t* __i = __middle;
    for (; __i <= __last - 4; __i += 4)
        if ((*__i > *__first) || (*(__i + 1) > *(__first + 1)) || (*(__i + 2) > *(__first + 2)) ||
            (*(__i + 3) > *(__first + 3))) {
            neon_pop_heap(__first, __middle, __i);
            // std::string buf = "after pop " + std::to_string(__i - __first);
            // PrintBatch4(__first, __last - __first, buf.c_str());
        }
    if (__i != __last) {
        static const float32_t minimal_fp32 = std::numeric_limits< float32_t >::lowest();
        for (__i--; __i >= __last - 4; __i--) {
            *(__i) = minimal_fp32;
        }
        if ((*__i > *__first) || (*(__i + 1) > *(__first + 1)) || (*(__i + 2) > *(__first + 2)) ||
            (*(__i + 3) > *(__first + 3))) {
            neon_pop_heap(__first, __middle, __i);
            // PrintBatch4(__first, __last - __first, "after pop last");
        }
    }
}

void neon_partial_sort(float32_t* __first, const uint k, const uint n, bool __increase = true) {
    if (n < 4 * k + 4) {
        return ::CPU_HEAP::cpu_partial_sort(__first, __first + k, __first + n, __increase);
    }
    // PrintBatch4(__first, n, "origin");
    neon_heap_select(__first, __first + 4 * k, __first + n);
    // PrintBatch4(__first, n, "after 4k heap");
    return ::CPU_HEAP::cpu_partial_sort(__first, __first + k, __first + 4 * k, __increase);
}
}

static std::vector< int > K = {16, 8, 32, 64, 128, 100, 47, 30, 18, 256, 10};
// static std::vector< int > K = {16};

int main(int argc, char const* argv[]) {
    const uint N = 65536;
    // const uint N = 200;
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

    // for (const auto& k : K) {
    //     auto input = random_data;
    //     auto start = GetRealTimeUs();
    //     std::partial_sort(
    //         input.begin(), input.begin() + k, input.end(), std::greater< float32_t >());
    //     auto end = GetRealTimeUs();
    //     std::cout << "Partial_sort: K-" << k << " :" << (end - start) << "us" << std::endl;
    //     for (size_t i = 0; i < k; i++) {
    //         if (fabsf(input[i] - sorted_data[i]) > 1e-6) {
    //             Print(&input[0], k, 16, "Error sort");
    //             break;
    //         }
    //     }
    // }
    // std::cout << std::endl;

    // for (const auto& k : K) {
    //     auto input = random_data;
    //     auto start = GetRealTimeUs();
    //     CPU_HEAP::cpu_partial_sort(&input[0], &input[0] + k, &input[0] + input.size());
    //     auto end = GetRealTimeUs();
    //     std::cout << "cpu_sort: K-" << k << " :" << (end - start) << "us" << std::endl;
    //     for (size_t i = 0; i < k; i++) {
    //         if (fabsf(input[i] - sorted_data[i]) > 1e-6) {
    //             Print(&input[0], k, 16, "Error sort");
    //             break;
    //         }
    //     }
    // }
    // std::cout << std::endl;

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
    // std::cout << std::endl;

    // for (const auto& k : K) {
    //     auto input = random_data;
    //     std::vector< uint > index(N);
    //     std::iota(index.begin(), index.end(), 0U);
    //     auto start = GetRealTimeUs();
    //     CPU_HEAP_WITH_ADDITIONAL::cpu_partial_sort(&input[0], &input[0] + k, &input[0] +
    //     input.size(), &index[0]);
    //     auto end = GetRealTimeUs();
    //     std::cout << "cpu_sort_index: K-" << k << " :" << (end - start) << "us" << std::endl;
    //     for (size_t i = 0; i < k; i++) {
    //         if (fabsf(input[i] - sorted_data[i]) > 1e-7) {
    //             Print(&input[0], k, 16, "Error data");
    //             Print(&sorted_data[0], k, 16, "Expect data");
    //             break;
    //         }
    //         if (fabsf(random_data[index[i]] - random_data[sorted_index[i]]) > 1e-7) {
    //             Print(&index[0], k, 16, "Error index");
    //             Print(&sorted_index[0], k, 16, "Expect index");
    //             break;
    //         }
    //     }
    // }
    // std::cout << std::endl;

    for (const auto& k : K) {
        auto input = random_data;
        auto start = GetRealTimeUs();
        NEON_HEAP::neon_partial_sort(&input[0], k, input.size());
        auto end = GetRealTimeUs();
        std::cout << "neon_sort: K-" << k << " :" << (end - start) << "us" << std::endl;
        for (size_t i = 0; i < k; i++) {
            if (fabsf(input[i] - sorted_data[i]) > 1e-6) {
                Print(&input[0], k, 16, "Error data");
                Print(&sorted_data[0], k, 16, "Expect data");
                break;
            }
        }
    }

    /* code */
    return 0;
}
