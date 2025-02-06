## CUDA Programming Basics

### Thread and Block Indices
- Each thread in a CUDA program has a unique index within its thread block, starting at `0`.
- Each block also has a unique index within the grid, starting at `0`.
- Threads are grouped into blocks, and blocks are grouped into a **grid**, which is the highest entity in the CUDA thread hierarchy.
- CUDA kernels are executed in a grid of one or more blocks, with each block containing one or more threads.

#### Special Variables
- `threadIdx.x`: Index of the thread within its block.
- `blockIdx.x`: Index of the block within the grid.
- `gridDim.x`: Number of blocks in the grid.

### Declaring and Launching a Kernel
#### 1. **Declaring a Kernel**
- Use the `__global__` keyword to declare a kernel.
- Example:
  ```cpp
  __global__ void vectorAdd(float *A, float *B, float *C, int N) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < N) {
          C[idx] = A[idx] + B[idx]; // Element-wise addition
      }
  }
  ```

#### 2. **Launching a Kernel**
- Example kernel launch from the host:
  ```cpp
  // Host code
  float *d_input, *d_output;
  cudaMalloc(&d_input, N * sizeof(float));
  cudaMalloc(&d_output, N * sizeof(float));

  // Launch kernel with `number_of_blocks` and `threads_per_block`
  kernel<<<number_of_blocks, threads_per_block>>>(d_input, d_output, N);
  ```

### CUDA Memory Transfer Functions
Data needs to be transferred between the host (CPU) and the device (GPU) before and after kernel execution. The key functions for memory transfer are:

- `cudaMemcpy(dst, src, size, cudaMemcpyKind)` - Copies memory between host and device.
  ```cpp
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); // Host to Device
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost); // Device to Host
  ```
- `cudaMalloc(&devPtr, size)` - Allocates memory on the device.
- `cudaFree(devPtr)` - Frees allocated memory on the device.


### Determining Number of Blocks
When the total number of elements `N` and the number of threads per block are known, the number of blocks can be calculated to ensure there are at least `N` threads in the grid:
```cpp
int N = 100000; // Total number of elements
size_t threads_per_block = 256; // Desired number of threads per block
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
```

### Grid Stride Loop
A grid stride loop is a technique used to iterate over data in parallel, allowing each thread to process multiple elements if necessary. This is particularly useful when the number of threads is less than the number of data elements.:
```cpp
__global__ void kernel(float *input, float *output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x; // Total threads in grid
  for (int i = idx; i < N; i += stride) {
    output[i] = input[i] * 2.0f; // Example operation
  }
}
```

### CPU vs GPU Comparison
| **CPU**                                  | **GPU**                                  |
|------------------------------------------|------------------------------------------|
| Fewer cores, optimized for sequential tasks | Thousands of cores, optimized for parallel tasks |
| Low latency, complex control flow        | High throughput, SIMD parallelism        |
| Better for task parallelism               | Better for data parallelism               |

### CUDA Execution Model
- **Host (CPU)**: Manages memory and kernel launches.
- **Device (GPU)**: Executes parallel code via kernels.

Note: My personal notes tidied up a bit using LLMs
