#include <cmath>
#include <limits>
#define M 4  
#define N 5  


__global__ void softmax(float* input, float* output, int M, int N)
{
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < M && col < N) 
    {
        // Step 1: Find maxVal per row : (repreats for every thread need to optimize)
        float maxVal = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; j++) 
        {
            maxVal = fmaxf(maxVal, input[row * N + j]);
        }

        // Step 2: Compute denominator
        float denominator = 0.0f;
        for (int j = 0; j < N; j++)
        {
            denominator += expf(input[row * N + j] - maxVal);
        }

        // Step 3: Compute softmax output
        output[row * N + col] = expf(input[row * N + col] - maxVal) / denominator;
    }
}





void softmax_cpu(float* input,float* output,int M,int N)
{
    // input is MxN
    float denominator;
    float maxVal = -std::numeric_limits<float>::infinity();

    //for each m vector
    for(int i=0;i<M;i++)
    {
        denominator = 0.0f;
        float maxVal = -std::numeric_limits<float>::infinity();

        for (int j = 0; j < N; j++) {
            maxVal = std::max(maxVal, input[i * N + j]);
        }


        for(int j=0;j<N;j++)
        {
            denominator += std::exp(input[i*N + j] - maxVal);
        }


        for(int j=0;j<N;j++)
        {
            output[i*N + j] = std::exp(input[i*N + j] - maxVal)/denominator;
        }

    }

}


int main()
{
    
    float input[M * N], output_cpu[M * N], output_gpu[M * N];

    // Initialize input with random values
    for (int i = 0; i < M * N; i++)
    {
        input[i] = static_cast<float>(rand() % 10) / 10.0f; // Random float [0,1]
    }

    // Compute softmax on CPU
    softmax_cpu(input, output_cpu, M, N);

    // Allocate memory on GPU
    float *d_input, *d_output;
    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(d_input, input, M * N * sizeof(float), cudaMemcpyHostToDevice);

    s
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);


    softmax<<<gridDim, blockDim>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    
    cudaMemcpy(output_gpu, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    bool match = true;
    for (int i = 0; i < M * N; i++)
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

    return 0;
}
