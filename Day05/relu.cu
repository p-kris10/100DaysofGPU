#include<cuda_runtime.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include<bits/stdc++.h>


__global__ void relu(float* A,float* B,int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;

    for(int i=idx;i<N;i = i + step)
    {
        B[i] = max(0.0f,A[i]);
    }

}


void relu_launcher(float* A,float* B,int N)
{
    float* d_A;
    float* d_B;
    size_t arr_size = N* sizeof(float);

    cudaMalloc((void**)&d_A,arr_size);
    cudaMalloc((void**)&d_B,arr_size);

    cudaMemcpy(d_A,A,arr_size,cudaMemcpyHostToDevice);

    size_t num_threads = 1024;
    size_t num_blocks = (N + num_threads - 1) / num_threads;

    relu<<<num_blocks,num_threads>>>(d_A,d_B,N);

    cudaMemcpy(B,d_B,arr_size,cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    


}