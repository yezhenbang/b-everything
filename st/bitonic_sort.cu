/*
 * Author: Rachel
 * <zhangruiqing01@baidu.com>
 *
 * File: bitonic_sort.cu
 * Create Date: 2015-08-05 17:10:44
 *
 */

#include "gputimer.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NThreads 8
#define NBlocks 4

#define Num NThreads* NBlocks


/* 
Align2(n)
Align2(k)

block: n/k
threads: k/2

total threads: n / 2
one thread complexity: 
    part1 generate k-sequences: log(k)*log(k)  约10算术/位运算操作+1比较分支+1交换
    part2 loop reduce N to k: (log(N)-log(k))*log(k)  约10算术/位运算操作+1比较分支+1交换
*/
__device__ void bitonic_sort_k(float* data, int data_num, int k) {
    int bid = BlockID(), tid = ThreadID();

    ShareMemory float share_data[];
    share_data[tid] = data[tid];
    share_data[tid + k*2] = data[tid + k*2];

    // generate (n/k) bitonic sequences of length 2k
    for (int half_bitonic_seq_size = 1; half_bitonic_seq_size <= k; half_bitonic_seq_size<<1) {
        int bitonic_seq_size = half_bitonic_seq_size << 1;
        for (int inc = half_bitonic_seq_size; inc > 0; inc >>= 1) {
            int low = (tid << 1) - (tid & inc - 1);
            bool reverse = bitonic_seq_size & low == 0;
            bool to_swap = share_data[low] < share_data[low + inc];
            // if (to_swap ^ reverse) {
            //     swap(share_data[low], share_data[low + inc]);
            // }
            // Using max min might be faster than swap???
            float tmp = share_data[low];
            share_data[low] = max(tmp, share_data[low + inc]);
            share_data[low + inc] = min(tmp, share_data[low+inc]);
            __syncthreads();
        }
    }

    // done , got (n/k) bitonic sequences of length 2k
    // [\\\k\\\///k/// | \\\k\\\///k/// \\\k\\\///k/// \\\k\\\///k///]
    int low = (tid << 1) - (tid & k - 1);
    share_data[low] = max(share_data[low], share_data[low + k]);
    // drop lower part, reduce data size to n/2 
    // [\\\k///       \\\k///       \\\k///       \\\k///       ] keep higher part

    // [       \\\k///       \\\k///       \\\k///       \\\k///] drop lower part

    __syncthreads();

    // ===============================

    int drop_tid = 1 << 1; // half threads done
    int remain_datasize = data_num >> 1;

    while (remain_datasize > k && drop_tid & tid == 0) {
        // loop merge 2k and reduce k, until last k
        int half_bitonic_seq_size = k; // constraint k
        int drop_offset = (drop_tid << 1 - 1) * k;
        for (int inc = half_bitonic_seq_size; inc > 0; inc >>= 1) {
            int low = (tid << 1) - (tid & inc - 1);
            bool reverse = bitonic_seq_size & low == 0;
            bool to_swap = share_data[low] < share_data[low + inc + drop_offset];
            if (to_swap ^ reverse) {
                swap(share_data[low], share_data[low + inc + drop_offset]);
            }
            __syncthreads();
        }

        remain_datasize >> 1;
        drop_tid << 1;

    } while (remain_datasize > k && drop_tid & tid == 0);

    // remain tid 0~k-1
    data[tid] = share_data[tid];
}



using namespace Gadgetron;

__device__ void swap(int& a, int& b) {
    int t = a;
    a = b;
    b = t;
}

__global__ void bitonic_sort(int* arr) {
    extern __shared__ int shared_arr[];
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // const unsigned int tid = threadIdx.x;
    shared_arr[tid] = arr[tid];
    __syncthreads();

    // for(int i=2; i<=blociDim.x; i<<=1){
    for (unsigned int i = 2; i <= Num; i <<= 1) {
        for (unsigned int j = i >> 1; j > 0; j >>= 1) {
            unsigned int tid_comp = tid ^ j;
            if (tid_comp > tid) {
                if ((tid & i) == 0) { // ascending
                    if (shared_arr[tid] > shared_arr[tid_comp]) {
                        swap(shared_arr[tid], shared_arr[tid_comp]);
                    }
                } else { // desending
                    if (shared_arr[tid] < shared_arr[tid_comp]) {
                        swap(shared_arr[tid], shared_arr[tid_comp]);
                    }
                }
            }
            __syncthreads();
        }
    }
    arr[tid] = shared_arr[tid];
}

int main(int argc, char* argv[]) {
    GPUTimer timer;
    int* arr = ( int* )malloc(Num * sizeof(int));

    // init array value
    time_t t;
    srand(( unsigned )time(&t));
    for (int i = 0; i < Num; i++) {
        arr[i] = rand() % 1000;
    }

    // init device variable
    int* ptr;
    cudaMalloc(( void** )&ptr, Num * sizeof(int));
    cudaMemcpy(ptr, arr, Num * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < Num; i++) {
        printf("%d\t", arr[i]);
    }
    printf("\n");

    dim3 blocks(NBlocks, 1);
    dim3 threads(NThreads, 1);

    timer.start();
    bitonic_sort< < < blocks, threads, Num * sizeof(int) > > >(ptr);
    // bitonic_sort<<<1,Num,Num*sizeof(int)>>>(ptr);
    timer.stop();

    cudaMemcpy(arr, ptr, Num * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < Num; i++) {
        printf("%d\t", arr[i]);
    }
    printf("\n");

    cudaFree(ptr);
    return 0;
}