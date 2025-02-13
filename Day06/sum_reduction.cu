%%writefile sum_reduction.cu

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

__global__ void reduce_global(float *d_in, float *d_out, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (idx >= N) return;


    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < N) {
            d_in[idx] += d_in[idx + s];
        }
        __syncthreads();
    }


    if (tid == 0) {
        d_out[blockIdx.x] = d_in[idx];
    }
}

void reduce_cpu(float *h_in, float *h_out, int N) {
    h_out[0] = 0;
    for (int i = 0; i < N; i++) {
        h_out[0] += h_in[i];
    }
}

int main() {
    const int N = 1<<24;
    size_t size = N * sizeof(float);

    float *h_in = (float *)malloc(size);
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;  // Initialize for known sum
    }

    size_t num_threads = 1024;
    int num_blocks = (N + num_threads - 1) / num_threads;
    float *h_out = (float *)malloc((num_blocks + 1) * sizeof(float));

    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, (num_blocks + 1) * sizeof(float));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);



    reduce_global<<<num_blocks, num_threads>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out,num_blocks*sizeof(float), cudaMemcpyDeviceToHost);

    float gpu_final_sum = 0;
    for (int i = 0; i < num_blocks; i++) {
        gpu_final_sum += h_out[i];
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);



    auto cpu_start = std::chrono::high_resolution_clock::now();
    reduce_cpu(h_in, h_out, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    float cpu_sum = h_out[0];

    if (gpu_final_sum == cpu_sum) {
        printf("Success! GPU and CPU sums match: %.2f\n", gpu_final_sum);
    } else {
        printf("SUM mismatch! GPU: %.2f, CPU: %.2f\n", gpu_final_sum, cpu_sum);
    }

    printf("GPU Time: %.3f ms\n", gpu_time);
    printf("CPU Time: %.3f ms\n", cpu_time.count());


    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
