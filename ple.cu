#include <cuda_runtime.h>
#include <cuda_bf16.h>

template<int BlockSize>
__global__ void add_ple_broadcast(
    const __nv_bfloat16* __restrict__ ple,
    __nv_bfloat16*       __restrict__ embeddings,
    const int hidden_size,
    const int sequence_length,
    const int batch_size,
    const int ple_dim
) {
    extern __shared__ __nv_bfloat16 ple_tile[];

    const int seq_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int thread_idx = threadIdx.x;

    const __nv_bfloat16* ple_src = ple + seq_idx * ple_dim;
    for (int i = thread_idx; i < ple_dim; i += BlockSize) {
        ple_tile[i] = __ldg(ple_src + i);
    }
    __syncthreads();

    __nv_bfloat16* embedding = embeddings + (batch_idx * sequence_length + seq_idx) * hidden_size;
    for (int h = thread_idx; h < hidden_size; h += BlockSize) {
        int ple_idx = h % ple_dim;
        embedding[h] = __hadd(embedding[h], ple_tile[ple_idx]);
    }
}