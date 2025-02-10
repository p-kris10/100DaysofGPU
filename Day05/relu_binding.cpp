#include <torch/extension.h>

// Declare the blur_call function

void relu_launcher(float* A,float* B,int N);

// Write the C++ function that we will call from Python
void relu_binding(float* A,float* B,int N) {
    relu_launcher(A,B,N);
}

PYBIND11_MODULE(gpu_kernels, m) {
  m.def(
    "relu_custom", // Name of the Python function to create
    [](torch::Tensor input, torch::Tensor output) {
            TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
            TORCH_CHECK(output.dtype() == torch::kFloat32, "Output tensor must be float32");
            TORCH_CHECK(input.sizes() == output.sizes(), "Input and output tensors must have the same shape");

            int N = input.numel(); // Total number of elements in the tensor

            float* input_ptr = input.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();

            relu_binding(input_ptr, output_ptr, N);
        }, // Corresponding C++ function to call
    "Launches the ReLU kernel", // Docstring
    py::arg("input"),
    py::arg("output")
  );
}