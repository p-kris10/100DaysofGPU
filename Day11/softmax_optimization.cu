%%writefile softmax.cu

#include <iostream>
#include <cmath>
#include <limits>
#include <chrono>
#include <cuda_runtime.h>
#include <cfloat>


#define ROWS 100 
#define COLS 200  
#define BLOCK_DIM_X 256


void softmax_cpu(float* input,float* output,int M,int N)
{
    // input is MxN
    float denominator;
    float maxVal;
    //for each m vector
    for(int i=0;i<M;i++)
    {
        denominator = 0.0f;
        maxVal = input[i*N];

        for (int j = 0; j < N; j++) {
            maxVal = fmaxf(maxVal, input[i * N + j]);
        }


        for(int j=0;j<N;j++)
        {
            denominator += expf(input[i*N + j] - maxVal);
        }


        for(int j=0;j<N;j++)
        {
            output[i*N + j] = expf(input[i*N + j] - maxVal)/denominator;
        }

    }

}

__global__ void softmax_sh(float* input, float* output, int M, int N)
{
    
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float maxVal;
    __shared__ float reduction[BLOCK_DIM_X]; 

    if (row < M && col < N) 
    {
        // Step 1: Compute max value in the row
        if (threadIdx.x == 0)
        {
            maxVal = -FLT_MAX;
            for (int j = 0; j < N; j++) 
            {
                maxVal = fmaxf(maxVal, input[row * N + j]);
            }
        }

        __syncthreads(); 

       
        float exp_val = expf(input[row * N + col] - maxVal);
        reduction[threadIdx.x] = exp_val;

        __syncthreads();

    
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (threadIdx.x < stride)
            {
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            }
            __syncthreads(); 
        }

  
        if (threadIdx.x == 0)
        {
            float denominator = 0.0f;
            denominator = reduction[0];

      
            for (int j = 0; j < N; j++)
            {
                output[row * N + j] = expf(input[row * N + j] - maxVal) / denominator;
            }
        }
    }
}




int main()
{
    float input[ROWS * COLS], output_cpu[ROWS * COLS], output_gpu[ROWS * COLS];

    for (int i = 0; i < ROWS * COLS; i++)
    {
        input[i] = static_cast<float>(rand() % 10) / 10.0f;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_cpu(input, output_cpu, ROWS, COLS);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " ms" << std::endl;

    float *d_input, *d_output;
    cudaMalloc(&d_input,ROWS * COLS* sizeof(float));
    cudaMalloc(&d_output, ROWS * COLS * sizeof(float));

    cudaMemcpy(d_input, input, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_DIM_X);
    dim3 gridDim(1, ROWS);


    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    softmax_sh<<<gridDim, blockDim>>>(d_input, d_output, ROWS, COLS);
    cudaEventRecord(stop_gpu);

    cudaDeviceSynchronize();
    cudaMemcpy(output_gpu, d_output, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_gpu);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    bool match = true;
    // for (int i = 0; i < ROWS ; i++)
    // {
    //   for(int j=0;j<COLS;j++)
    //   {
    //     std::cout<<output_cpu[i*COLS + j]<<" : "<<output_gpu[i*COLS + j]<<"\n"<<i<<" "<<j<<"\n";
    //   }
        
    // }
    
    for (int i = 0; i < ROWS * COLS; i++)
    {
        if (fabs(output_cpu[i] - output_gpu[i]) > 1e-5) 
        {
            match = false;
            break;
        }
    }

    std::cout << (match ? "Results match!" : "Mismatch in results!") << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
