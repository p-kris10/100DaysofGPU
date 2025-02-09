#include<bits/stdc++.h>
#include <time.h>


__global__ void blur_kernel(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

  
    if(col < width && row<height)
    {
      // get index given row major format
      int pixVal_r = 0;
      int pixVal_g = 0;
      int pixVal_b = 0;
      int pix_cnt = 0;

      for(int currRow = row - 1;currRow<row+2;currRow++)
      {
        for(int currCol = col - 1;currCol<col+2;currCol++)
        {
            if(currRow < height && currRow > -1 && currCol < width && currCol > -1)
            {
                int index = currRow*width + currCol;

                //intuition : Row 0:   R0  G0  B0   R1  G1  B1   R2  G2  B2   R3  G3  B3 

                int rgb_index = index * 3;
                unsigned char r = Pin[rgb_index];
                unsigned char g = Pin[rgb_index + 1];
                unsigned char b = Pin[rgb_index + 2];


                pixVal_r += r;
                pixVal_g += g;
                pixVal_b += b;

                pix_cnt += 1;

            }
        }
      }
        int index = row*width + col;
        int rgb_index = index * 3;
        unsigned char new_r = static_cast<unsigned char> (pixVal_r  / pix_cnt);
        unsigned char new_g = static_cast<unsigned char> (pixVal_g  / pix_cnt);
        unsigned char new_b = static_cast<unsigned char> (pixVal_b  / pix_cnt);
     

        Pout[rgb_index] =  new_r;
        Pout[rgb_index+1] =  new_g;
        Pout[rgb_index+2] =  new_b;

    }

    
}


void blur_launcher(unsigned char* Pout_h, unsigned char* Pin_h, int width, int height)
{

    unsigned char *Pout_d, *Pin_d;
    const int imgSize = width * height * 3 * sizeof(unsigned char) ;
    cudaError_t err = cudaMalloc((void**)&Pout_d, imgSize);
    
    err = cudaMalloc((void**)&Pin_d, imgSize);
    err = cudaMemcpy(Pin_d, Pin_h, imgSize, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    blur_kernel<<<dimGrid,dimBlock>>>(Pout_d,Pin_d,width,height);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();
    err = cudaMemcpy(Pout_h, Pout_d, imgSize, cudaMemcpyDeviceToHost);
    std::cout << "HERE : THE END"<<cudaGetErrorString(err)<< std::endl;
    cudaFree(Pin_d);
    cudaFree(Pout_d);


}

// int main() {
//     cv::Mat img = cv::imread("image.jpg");
    
//     clock_t start, end;
//     start = clock();

//     if (img.empty()) {
//         std::cerr << "Error: Could not load image!" << std::endl;
//         return -1;
//     }
    

//     int width = img.cols;
//     int height = img.rows;
//     int channels = img.channels(); 


//     unsigned char *d_input, *d_output;
//     size_t imgSize = width * height * 3 * sizeof(unsigned char);
//     size_t blurSize = width * height * 3 * sizeof(unsigned char);

//     cudaMalloc((void**)&d_input, imgSize);
//     cudaMalloc((void**)&d_output, blurSize);


//     cudaMemcpy(d_input,img.data,imgSize,cudaMemcpyHostToDevice);

//     cv::Mat blur_img(img.rows, img.cols, CV_8UC3);
    

//     dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
//     dim3 dimBlock(16, 16, 1);

//     blur_kernel<<<dimGrid,dimBlock>>>(d_output,d_input,width,height);
//     cudaDeviceSynchronize();
//     end = clock();
//     float time2 = ((float)(end - start)) / CLOCKS_PER_SEC;
//     cudaMemcpy(blur_img.data, d_output, blurSize, cudaMemcpyDeviceToHost);
    
   

//     printf("GPU: %f seconds\n", time2);

//     cv::imwrite("blur_image_gpu.jpg", blur_img);

//     cudaFree(d_input);
//     cudaFree(d_output);

//     printf("%d, %d, %d\n",width,height,channels);
//     return 0;
// }