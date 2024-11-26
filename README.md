# Convert Image to Grayscale image using CUDA 

## AIM:

The aim of this project is to demonstrate how to convert an image to grayscale using CUDA programming without relying on the OpenCV library. It serves as an example of GPU-accelerated image processing using CUDA.

## Procedure:
1. Load the input image using the stb_image library.
2. Allocate memory on the GPU for the input and output image buffers.
3. Copy the input image data from the CPU to the GPU.
4. Define a CUDA kernel function that performs the grayscale conversion on each pixel of the image.
5. Launch the CUDA kernel with appropriate grid and block dimensions.
6. Copy the resulting grayscale image data from the GPU back to the CPU.
7. Save the grayscale image using the stb_image_write library.
8. Clean up allocated memory.

## Program:
```
NAME    : YOGESH RAO S D
REG NO  : 212222110055
```
```c++
#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


__global__
void colorConvertToGrey(unsigned char *rgb, unsigned char *grey, int rows, int cols)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col < cols && row < rows)
	{
		int grey_offset = row * cols + col;
		int rgb_offset = grey_offset * CHANNELS;
	
    	unsigned char r = rgb[rgb_offset + 0];
	    unsigned char g = rgb[rgb_offset + 1];
	    unsigned char b = rgb[rgb_offset + 2];
	
	    grey[grey_offset] = r * 0.299f + g * 0.587f + b * 0.114f;
    }
}

size_t loadImageFile(unsigned char *grey_image, const std::string &input_file, int *rows, int *cols );

void outputImage(const std::string &output_file, unsigned char *grey_image, int rows, int cols);

unsigned char *h_rgb_image; 

int main(int argc, char **argv) 
{
	std::string input_file;
	std::string output_file;

	switch(argc) {
		case 3:
			input_file = std::string(argv[1]);
			output_file = std::string(argv[2]);
            break;
		default:
			std::cerr << "Usage: <executable> input_file output_file";
			exit(1);
	}
	
	unsigned char *d_rgb_image; 
	unsigned char *h_grey_image, *d_grey_image; 
	int rows; 
	int cols; 
	
	const size_t total_pixels = loadImageFile(h_grey_image, input_file, &rows, &cols);

	h_grey_image = (unsigned char *)malloc(sizeof(unsigned char*)* total_pixels);

	cudaMalloc(&d_rgb_image, sizeof(unsigned char) * total_pixels * CHANNELS);
	cudaMalloc(&d_grey_image, sizeof(unsigned char) * total_pixels);
	cudaMemset(d_grey_image, 0, sizeof(unsigned char) * total_pixels);
	
	cudaMemcpy(d_rgb_image, h_rgb_image, sizeof(unsigned char) * total_pixels * CHANNELS, cudaMemcpyHostToDevice);

	const dim3 dimGrid((int)ceil((cols)/16), (int)ceil((rows)/16));
	const dim3 dimBlock(16, 16);
	
	colorConvertToGrey<<<dimGrid, dimBlock>>>(d_rgb_image, d_grey_image, rows, cols);

	cudaMemcpy(h_grey_image, d_grey_image, sizeof(unsigned char) * total_pixels, cudaMemcpyDeviceToHost);

	outputImage(output_file, h_grey_image, rows, cols);
	cudaFree(d_rgb_image);
	cudaFree(d_grey_image);
	return 0;
}

size_t loadImageFile(unsigned char *grey_image, const std::string &input_file, int *rows, int *cols) 
{
	cv::Mat img_data; 

	img_data = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
	if (img_data.empty()) 
	{
		std::cerr << "Unable to laod image file: " << input_file << std::endl;
	}
		
	*rows = img_data.rows;
	*cols = img_data.cols;

	h_rgb_image = (unsigned char*) malloc(*rows * *cols * sizeof(unsigned char) * 3);
	unsigned char* rgb_image = (unsigned char*)img_data.data;

	int x = 0;
	for (x = 0; x < *rows * *cols * 3; x++)
	{
		h_rgb_image[x] = rgb_image[x];
	}
	
	size_t num_of_pixels = img_data.rows * img_data.cols;
	
	return num_of_pixels;
}

void outputImage(const std::string& output_file, unsigned char* grey_image, int rows, int cols)
{

	cv::Mat greyData(rows, cols, CV_8UC1,(void *) grey_image);
	cv::imwrite(output_file.c_str(), greyData);
}
```

## OUTPUT:

### Input Image
<img src="https://github.com/user-attachments/assets/258fe645-3af6-438f-b459-f819d2c50b0b">


### Grayscale Image
<img src="https://github.com/user-attachments/assets/6baa62a4-e302-4f4a-b56b-e2cac956e3d1">


## Result:
The CUDA program successfully converts the input image to grayscale using the GPU. The resulting grayscale image is saved as an output file. This example demonstrates the power of GPU parallelism in accelerating image processing tasks.
