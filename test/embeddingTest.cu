#include <iostream> 
#include <vector> 
#include <fstream> 
#include <string> 
#include <random>

#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in file '%s' at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

typedef long long ll; 

bool load_weights(
    const std::string& file_path, 
    std::vector<__nv_bfloat16>& h_table, 
    int expected_vocab_size, 
    int expected_hidden_size
) { 
    std::ifstream file(file_path, std::ios::binary); 
    if (!file.is_open()) {
        std::cerr << "Cannot open embedding weights file: " << file_path << std::endl;
        return false;
    }

    size_t expected_elements = static_cast<size_t>(expected_vocab_size) * expected_hidden_size; 
    std::vector<uint16_t> raw_data(expected_elements); 

    file.read(reinterpret_cast<char*>(raw_data.data()), expected_elements * sizeof(uint16_t)); 

    if (!file.good()){
        std::cerr << "Error reading embedding weights file" << std::endl; 
        return false;
    }
    for (size_t i = 0; i < expected_elements; ++i) { 
        h_table[i] = *reinterpret_cast<__nv_bfloat16*>(&raw_data[i]); 
    }
    file.close(); 
    std::cout<<"Successfully loaded file"<<std::endl;
    return true;
}

int main() {
    constexpr int vocab_size = 262144;
    constexpr int hidden_size = 2048;
    constexpr int sequence_len = 2048; 
    constexpr int batch_size = 4; 
    constexpr int total_tokens = sequence_len * batch_size; 

    constexpr int warmup_runs = 10; 
    constexpr int benchmark_runs = 100; 

    std::cout << "---Benchmarking embedding kernel---" << std::endl; 

    //Initializing host buffers 
    const size_t table_elements = static_cast<size_t>(vocab_size) * hidden_size;
    const size_t ids_elements = total_tokens;
    const size_t out_elements = static_cast<size_t>(total_tokens) * hidden_size;

    const size_t table_bytes = table_elements * sizeof(__nv_bfloat16);
    const size_t ids_bytes = ids_elements * sizeof(int);
    const size_t out_bytes = out_elements * sizeof(__nv_bfloat16);

    std::vector<__nv_bfloat16> h_table(table_elements);
    std::vector<int> h_ids(ids_elements);
    std::vector<__nv_bfloat16> h_out(out_elements);

    // loading pre-trained weights 
    if (!load_weights("embedding.bin", h_table, vocab_size, hidden_size)) {
        return EXIT_FAILURE; 
    }

    // generating random tokens 
    std::mt19937 rng(42); 
    std::uniform_int_distribution<int> dist(0, vocab_size - 1); 
    for (auto &id: h_ids) id = dist(rng); 

    // allocating device buffers 
    __nv_bfloat16 *d_table = nullptr; 
    __nv_bfloat16 *d_out = nullptr; 
    int *d_ids = nullptr; 

    CHECK_CUDA(cudaMalloc((void**)&d_table, table_bytes)); 
    CHECK_CUDA(cudaMalloc((void**)&d_out, out_bytes)); 
    CHECK_CUDA(cudaMalloc((void**)&d_ids, ids_bytes)); 

    // copying data to device 
    CHECK_CUDA(cudaMemcpy(d_table, h_table.data(), table_bytes, cudaMemcpyHostToDevice)); 
    CHECK_CUDA(cudaMemcpy(d_ids, h_ids.data(), ids_bytes, cudaMemcpyHostToDevice)); 

    constexpr int threads_x = 128; 
    constexpr int threads_y = 4; 
    dim3 block(threads_x, threads_y,1); 
    dim3 grid((total_tokens + threads_x - 1) / threads_x, 1, 1); 

    // warmup 
    for (int i = 0; i < warmup_runs; ++i) { 
        embedding<<<grid, block>>>(d_table, d_ids, d_out, hidden_size, total_tokens); 
        CHECK_CUDA(cudaGetLastError()); 
    }
    CHECK_CUDA(cudaDeviceSynchronize()); 

    // main loop 
    cudaEvent_t start, stop; 
    CHECK_CUDA(cudaEventCreate(&start)); 
    CHECK_CUDA(cudaEventCreate(&stop)); 

    CHECK_CUDA(cudaEventRecord(start)); 
    for (int i = 0; i < benchmark_runs; ++i) { 
        embedding<<<grid, block>>>(d_table, d_ids, d_out, hidden_size, total_tokens); 
    }
    CHECK_CUDA(cudaEventRecord(stop)); 
    CHECK_CUDA(cudaEventSynchronize(stop));
    float total_time = 0.0f; 
    CHECK_CUDA(cudaEventElapsedTime(&total_time, start, stop)); 
    const double average_time_ms = total_time / benchmark_runs; 

    const double bytes_rw = 2.0 * out_bytes;
    const double bandwidth_gb_s = bytes_rw / (average_time_ms / 1000.0) / 1e9;

    // A single bfloat16 read and write is counted as 2 floating point operations
    const double gflops = (2.0 * out_elements) / (average_time_ms * 1e6);

    std::cout<<"Avg time / launch: " << average_time_ms << " ms" << std::endl; 
    std::cout<<"Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl; 
    std::cout<<"GFLOPS: " << gflops << std::endl; 

    cudaEventDestroy(start); 
    cudaEventDestroy(stop); 
    cudaFree(d_table); 
    cudaFree(d_out); 
    cudaFree(d_ids); 
    
    return 0;
}  
