#include <torch/extension.h>

// Declare the blur_call function

void blur_launcher(unsigned char* Pout_h, unsigned char* Pin_h, int width, int height);

// Write the C++ function that we will call from Python
void blur_binding(unsigned char* Pout_h, unsigned char* Pin_h, int width, int height) {
    blur_launcher(Pout_h, Pin_h, width, height);
}

PYBIND11_MODULE(gpu_kernels, m) {
  m.def(
    "blur_gpu", // Name of the Python function to create
    [](torch::Tensor input, torch::Tensor output) {
            TORCH_CHECK(input.dtype() == torch::kUInt8, "Input tensor must be unsigned char");
            TORCH_CHECK(output.dtype() == torch::kUInt8, "Output tensor must be unsigned char");
            TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (height x width x channels)");
            TORCH_CHECK(input.size(2) == 3, "Input tensor must have 3 channels");
            
            int height = input.size(0);
            int width = input.size(1);

            std::cout << "Input image dimensions: Height = " << height << ", Width = " << width << std::endl;
            
            unsigned char* input_ptr = input.data_ptr<unsigned char>();
            unsigned char* output_ptr = output.data_ptr<unsigned char>();
            
            blur_binding(output_ptr, input_ptr, width, height);
        }, // Corresponding C++ function to call
    "Launches the blur kernel", // Docstring,
    py::arg("input"),
    py::arg("output") // These name the arguments, allowing Python to use keyword arguments like: gpu_kernels.rgb_to_grayscale(input=my_rgb_tensor, output=my_gray_tensor)
  );
}
