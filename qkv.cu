#include <cuda_runtime.h> 
#include <cuda_bf16.h> 
#include <cuda/pipeline> 

struct __align(8)__ regs{ uint32_t r0, r1, r2, r3; };

__global__ void qkv_projection(
    __nv_bfloat16* X, 
    __nv_bfloat16* W_qkv, 
    __nv_bfloat16* bias,
    __nv_bfloat16* output, 
    int b, int s, int h
) {
}

// loading matrix X from shared memory to the registers 
__device__ void ld_matrix_shared_to_reg(
    regs* r,
    const __nv_bfloat16* smem
) { 
    asm volatile( 
        "ldmatrix.sync.aligned.m8n8.x4.shared.bf16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r->r0), "=r"(r->r1), "=r"(r->r2), "=r"(r->r3)
        : "r"(__cvta_generic_to_shared(smem)) 
    );
}

// loading from global memory to shared memory 
template <int TILE_M, int TILE_N>
__device__ __forceinline__ void ld_matrix_global_to_shared(
    const __nv_bfloat16* __restrict__ A, 
    __nv_bfloat16* __restrict__ B,
    int row_offset, 
    int col_offset,
    int lda
) { 
    int idx = threadIdx.x; 
    int idy = threadIdx.y; 

    static_assert(TILE_M % 8 == 0 &&  TILE_N % 8 == 0,
                  "tile must be a multiple of 8x8")
    
    constexpr int elems = 8; // 4 regs * 32 bits per regs results in 8 elements 
    constexpr int bytes = elems * sizeof(__nv_bfloat16); 
    constexpr int copies_per_tile = (TILE_M * TILE_N) / elems; 
    
    int lane = threadIdx.x & 31; 

    for (int copy = lane; copy < copies_per_tile; copy += 32) { 
        const int elem_idx = copy * copies_per_tile; 
        const int row = elem_idx / TILE_N; 
        const int col = elem_idx % TILE_N; 

        const void *ptr = reinterpret_cast<const void*>(
            A + (row_offset + row) * lda + (col_offset + col); 
        )

        void *smem = reinterpret_cast<void*>(
            B + (row * TILE_N + col); 
        )

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, 16;"
            :: "r"(smem), "l"(ptr); 
        )
    }
    asm volatile("cp.async.commit_group;\n":::"memory"); 
    asm volatile("cp.aync.wait_group 0;\n":::"memory");
    __syncthreads(); 
}