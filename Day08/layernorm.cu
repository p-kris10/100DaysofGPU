#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

__global__ void layer_norm_gpu(float* d_in, float* d_out, int M, int N) {
    //one thread for each sample

    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    

    if (idx >= M) return; 

    float sum_x = 0.0;
    float sum_diff = 0.0;

    // Compute (mu) 
    for (int j = 0; j < N; j++) {
        sum_x += d_in[idx * N + j];  
    }

    float mu = sum_x / N;  // Mean

    // Compute (var) 
    for (int j = 0; j < N; j++) {
        sum_diff += (d_in[idx * N + j] - mu) * (d_in[idx * N + j] - mu);  
    }

    float var = sum_diff / N;  
    float stddev = sqrt(var);  

    
    for (int j = 0; j < N; j++) {
        d_out[idx * N + j] = (d_in[idx * N + j] - mu) / stddev;  
    }

}

void reduce_cpu(float* h_in, float* h_out, int M, int N) {
    // For each sample in M
    for (int i = 0; i < M; i++) {
        float sum_x = 0.0;
        float sum_diff = 0.0;

        // Sum
        for (int j = 0; j < N; j++) {
            sum_x += h_in[i * N + j];
        }

        // Mean (mu)
        float mu = sum_x / N;

        // Variance
        for (int j = 0; j < N; j++) {
            sum_diff += (h_in[i * N + j] - mu) * (h_in[i * N + j] - mu);
        }

        float var = sum_diff / N;
        float stddev = sqrt(var);

        // Normalize
        for (int j = 0; j < N; j++) {
            h_out[i * N + j] = (h_in[i * N + j] - mu) / stddev;
        }
    }
}

int main() {
    const int M = 1024;  
    const int N = 256;   
    size_t size = M * N * sizeof(float);

   
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    //init
    for (int i = 0; i < M * N; i++) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX;
    }

 
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

  
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int num_threads = 256;
    int num_blocks = (M + num_threads - 1) / num_threads;
    layer_norm_gpu<<<num_blocks, num_threads>>>(d_in, d_out, M, N);

    
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

   
    auto cpu_start = std::chrono::high_resolution_clock::now();
    reduce_cpu(h_in, h_out, M, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    
    printf("GPU Time: %.3f ms\n", gpu_time);
    printf("CPU Time: %.3f ms\n", cpu_time.count());

    
    printf("Sample 0 normalized values: ");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", h_out[i]);
    }
    printf("\n");

  
    cudaFree(d_in);
    cudaFree(d_out);
   
}
