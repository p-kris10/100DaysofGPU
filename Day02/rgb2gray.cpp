#include <opencv2/opencv.hpp>
#include<bits/stdc++.h>
#include <time.h>

void grayscaleCPU(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            
            int index = row * width + col;
            
            int rgb_index = index * 3;
         
            unsigned char r = Pin[rgb_index];
            unsigned char g = Pin[rgb_index + 1];
            unsigned char b = Pin[rgb_index + 2];

           
            unsigned char gray = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);

            
            Pout[index] = gray;
        }
    }
}


int main() {

    cv::Mat img = cv::imread("image.jpg");
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    clock_t start, end;
    start = clock();

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

   
    cv::Mat gray_img(height, width, CV_8UC1);

   
    grayscaleCPU(gray_img.data, img.data, width, height);

    end = clock();
    float time2 = ((float)(end - start)) / CLOCKS_PER_SEC;

    printf("CPU: %f seconds\n", time2);

    cv::imwrite("gray_image.jpg", gray_img);

    printf("%d, %d, %d\n", width, height, channels);
    return 0;
}
