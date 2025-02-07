#include <opencv2/opencv.hpp>
#include<bits/stdc++.h>
#include <time.h>

__global__ void rgb2gray_kernel(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if(col < width && row<height)
    {
      // get index given row major format
      int index = row*width + col;
      //intuition : Row 0:   R0  G0  B0   R1  G1  B1   R2  G2  B2   R3  G3  B3 
      int rgb_index = index * 3;
      unsigned char r = Pin[rgb_index];
      unsigned char g = Pin[rgb_index + 1];
      unsigned char b = Pin[rgb_index + 2];

      unsigned char gray = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);

      Pout[index] =  gray;

    }

    
}

int main() {
    cv::Mat img = cv::imread("image.jpg");
    
    clock_t start, end;
    start = clock();

    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels(); 


    unsigned char *d_input, *d_output;
    size_t imgSize = width * height * 3 * sizeof(unsigned char);
    size_t graySize = width * height * sizeof(unsigned char);

    cudaMalloc((void**)&d_input, imgSize);
    cudaMalloc((void**)&d_output, graySize);


    cudaMemcpy(d_input,img.data,imgSize,cudaMemcpyHostToDevice);

    cv::Mat gray_img(img.rows, img.cols, CV_8UC1);
    

    dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 dimBlock(16, 16, 1);

    rgb2gray_kernel<<<dimGrid,dimBlock>>>(d_output,d_input,width,height);
    cudaDeviceSynchronize();
    end = clock();
    float time2 = ((float)(end - start)) / CLOCKS_PER_SEC;
    cudaMemcpy(gray_img.data, d_output, graySize, cudaMemcpyDeviceToHost);
    
   

    printf("GPU: %f seconds\n", time2);

    cv::imwrite("gray_image_gpu.jpg", gray_img);

    cudaFree(d_input);
    cudaFree(d_output);

    printf("%d, %d, %d\n",width,height,channels);
    return 0;
}