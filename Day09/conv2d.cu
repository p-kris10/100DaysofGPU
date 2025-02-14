#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#define MASK_DIM 3


__constant__ float mask_c[MASK_DIM][MASK_DIM];


__global__ void convolution_kernel(float* input,float* output,int width, int height)
{
    //one thread for one output pixel

    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    int MASK_RADIUS = MASK_DIM/2;
    
    if(outRow < height && outCol < width)
    {
        float sum = 0.0f;
        for(int maskRow = 0;maskRow<MASK_DIM;maskRow++)
        {
            for(int maskCol = 0;maskCol<MASK_DIM;maskCol++)
            {
                int inRow = outRow - MASK_RADIUS + maskRow;
                int inCol = outCol - MASK_RADIUS + maskCol;
                if (inRow < height && inRow >=0 && inCol < width && inCol >=0)
                {
                    sum += mask_c[maskRow][maskCol]*input[inRow*width + inCol];
                }
               
            }
        }

        output[outRow*width + outCol] = sum;

    }
    

}

void convolution_cpu(const float* input, float* output, int width, int height, const float* mask) {
    int MASK_RADIUS = MASK_DIM/2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            float sum = 0.0f;

            for (int maskRow = 0; maskRow < MASK_DIM; maskRow++) {
                for (int maskCol = 0; maskCol < MASK_DIM; maskCol++) {

                    int inRow = i -  MASK_RADIUS + maskRow;
                    int inCol = j -  MASK_RADIUS + maskCol;

                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        sum += mask[maskRow * MASK_DIM + maskCol] * input[inRow * width + inCol];
                    }
                }
            }
            output[i * width + j] = sum;
        }
    }
}


bool compare_results(const float* cpu_result, const float* gpu_result, int size, float epsilon = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (std::fabs(cpu_result[i] - gpu_result[i]) > epsilon) {
            return false;
        }
    }
    return true;
}


int main() {
    int width = 1024, height = 1024;
    float* input = new float[width * height];
    float* output_cpu = new float[width * height];
    float* output_gpu = new float[width * height];
    float mask[MASK_DIM * MASK_DIM] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

    std::fill(input, input + width * height, 1.0f);
    std::fill(output_cpu, output_cpu + width * height, 0.0f);
    std::fill(output_gpu, output_gpu + width * height, 0.0f);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    convolution_cpu(input, output_cpu, width, height, mask);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " ms" << std::endl;

    float *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));
    cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    convolution_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(output_gpu, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_time.count() << " ms" << std::endl;

    bool correct = compare_results(output_cpu, output_gpu, width * height);
    std::cout << "Results match: " << (correct ? "Yes" : "No") << std::endl;

    return 0;
}