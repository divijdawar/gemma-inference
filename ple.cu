#include <cuda_runtime.h>
#include <cuda_bf16.h>

template<int vec = 16> 
__global__ void ple(
    const __nv_bfloat16* __restrict__ ple, // [sequence_length, ple_dim]
    __nv_bfloat16*  __restrict__ embeddings, // [batch_size, sequence_length, hidden_size]
    const int hidden_size,
    const int sequence_length,
    const int batch_size,
    const int ple_dim
) { 
    constexpr int tx = vec; 
    constexpr int ty = 1; 

    extern __shared__ __nv_bfloat16 tile[]; 

    int idx = threadIdx.x; 
    int seq = blockIdx.y; 
    int tid = blockIdx.z; 

    for (int p = idx; p < ple_dim; p += tx){
        tile[p] = __ldg(ple + seq*ple_dim + p);
    }
}