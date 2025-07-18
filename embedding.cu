#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

    // Async pipeline approach for better efficiency
    auto block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline();
    
    constexpr int bytes_per_transfer = 16; // 16 bytes = 8 bfloat16 elements
    constexpr int elements_per_transfer = bytes_per_transfer / sizeof(__nv_bfloat16);
    
    // Pipeline depth - allows overlapping transfers
    constexpr int pipeline_depth = 2;
    
    for (int i = tid_y * elements_per_transfer; i < hidden_size; i += num_y * elements_per_transfer) {
        // Calculate actual bytes to transfer (handle remainder)
        int remaining_elements = min(elements_per_transfer, hidden_size - i);
        int bytes_to_transfer = remaining_elements * sizeof(__nv_bfloat16);
        
        // Async memory copy with pipeline
        pipe.producer_acquire();
        cuda::memcpy_async(block, dst + i, src + i, bytes_to_transfer, pipe);
        pipe.producer_commit();
        
        // Consumer wait and commit for pipeline management
        if (i >= pipeline_depth * num_y * elements_per_transfer) {
            pipe.consumer_wait();
            pipe.consumer_release();
        }
    }
    
    // Wait for all remaining pipeline stages
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    block.sync();
}