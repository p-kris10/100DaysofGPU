#include <opencv2/opencv.hpp>
#include<bits/stdc++.h>
#include <time.h>
void blurCPU(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int pixVal_r = 0, pixVal_g = 0, pixVal_b = 0, pix_cnt = 0;

            for (int blurRow = -1; blurRow <= 1; blurRow++) {
                for (int blurCol = -1; blurCol <= 1; blurCol++) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        int rgb_index = (curRow * width + curCol) * 3;

                        unsigned char r = Pin[rgb_index];
                        unsigned char g = Pin[rgb_index + 1];
                        unsigned char b = Pin[rgb_index + 2];

                        pixVal_r += r;
                        pixVal_g += g;
                        pixVal_b += b;
                        pix_cnt++;
                    }
                }
            }

            
            int rgb_index = (row * width + col) * 3;
            Pout[rgb_index] = static_cast<unsigned char>(pixVal_r / pix_cnt);
            Pout[rgb_index + 1] = static_cast<unsigned char>(pixVal_g / pix_cnt);
            Pout[rgb_index + 2] = static_cast<unsigned char>(pixVal_b / pix_cnt);
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

   
    cv::Mat blur_img(height, width, CV_8UC3);

   
    grayscaleCPU(blur_img.data, img.data, width, height);

    end = clock();
    float time2 = ((float)(end - start)) / CLOCKS_PER_SEC;

    printf("CPU: %f seconds\n", time2);

    cv::imwrite("blur_image_cpu.jpg", gray_img);

    printf("%d, %d, %d\n", width, height, channels);
    return 0;
}
