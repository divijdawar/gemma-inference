#include "kernels.cuh"
#include <cuda/pipeline> 
#include <cooperative_groups.h>
namespace cg = cooperative_groups; 

//The following file contains the cuda kernel for token embeddings. 

__global__ void embedding(
    const __nv_bfloat16* __restrict__ table , // [V, H]
    const int* __restrict__ ids, 
    __nv_bfloat16* __restrict__ out, // [N,H]
    const int hidden_size, 
    const int total_tokens
){ 
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread points to a single token 
    if(idx >= total_tokens) return; 

    const int tid_y = threadIdx.y; // each thread points to a single element in the hidden size 
    const int num_y = blockDim.y; 

    const __nv_bfloat16* src = table + ids[idx] * hidden_size; 
    __nv_bfloat16* dst = out + idx * hidden_size;

    auto block = cg::this_thread_block(); 
    const int bytes_per_thread = 16; 
    const int elements_per_thread = bytes_per_thread/ sizeof(__nv_bfloat16);

    for (int i = tid_y * elements_per_thread; i < hidden_size; i += num_y * elements_per_thread) { 
        cuda_memcpy_async(
            dst + i, 
            src + i, 
            min(bytes_per_thread , (hidden_size - i) * (int)sizeof(__nv_bfloat16)),
            block
        );
    }
    cg::sync(block); 
}