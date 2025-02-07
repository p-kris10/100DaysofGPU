
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void initWith(float num, float *a, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        a[i] = num;
    }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        result[i] = a[i] + b[i];
    }
}

void test(float target, float *array, int N) {
    for (int i = 0; i < N; i++) {
        if (array[i] != target) {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main() {
    const int N = 1 << 26; // 2^26 elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    size_t num_threads = 1024;
    size_t num_blocks = (N + num_threads - 1) / num_threads;

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    
    clock_t start, end;
    start = clock();

   
    initWith<<<num_blocks, num_threads>>>(3, d_a, N);
    initWith<<<num_blocks, num_threads>>>(4, d_b, N);
    initWith<<<num_blocks, num_threads>>>(0, d_c, N);

    
    addVectorsInto<<<num_blocks, num_threads>>>(d_c, d_a, d_b, N);

    
    cudaDeviceSynchronize();
    end = clock();

    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify correctness
    test(7, h_c, N);

    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    
    float time2 = ((float)(end - start)) / CLOCKS_PER_SEC;
    printf("CUDA: %f seconds\n", time2);

    
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
