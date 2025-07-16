//Header file for all CUDA kernels
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>


__global__ void embedding(
    const __nv_bfloat16* __restrict__ table,
    const int* __restrict__ ids,
    __nv_bfloat16* __restrict__ out, 
    const int hidden_size,
    const int total_tokens
);
