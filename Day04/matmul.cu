
#include<cuda_runtime.h>
#include<iostream>
#include<vector>
#include<chrono>
#include<bits/stdc++.h>

#define TILE_DIM 32

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int width)
{
    __shared__ float Mds[TILE_DIM][TILE_DIM];
    __shared__ float Nds[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float Pval = 0;

    for(int ph = 0; ph < width / TILE_DIM; ph++)
    {
        Mds[ty][tx] = d_M[row * width + ph * TILE_DIM + tx];
        Nds[ty][tx] = d_N[(ph * TILE_DIM + ty) * width + col];
        __syncthreads();

        for(int k = 0; k < TILE_DIM; k++)
        {
            Pval += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    d_P[row * width + col] = Pval;
}

void InitializeMatrix(std::vector<float>& matrix, int width) {
    for (int i = 0; i < width * width; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}


void MatrixMulCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0;
            for (int k = 0; k < width; k++) {
                sum += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = sum;
        }
    }
}

int main() {
    int width = 1024; 

    size_t size = width * width * sizeof(float);

    std::vector<float> h_M(width * width);
    std::vector<float> h_N(width * width);
    std::vector<float> h_P(width * width, 0);
    
    InitializeMatrix(h_M, width);
    InitializeMatrix(h_N, width);

    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    cudaMemcpy(d_M, h_M.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N.data(), size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid(width / TILE_DIM, width / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
    cudaEventRecord(stop);

    cudaMemcpy(h_P.data(), d_P, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "CUDA Time taken: " << milliseconds << " ms" << std::endl;

    std::vector<float> h_P_CPU(width * width, 0);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    MatrixMulCPU(h_M, h_N, h_P_CPU, width);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU Time taken: " << cpu_duration.count() << " ms" << std::endl;

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

