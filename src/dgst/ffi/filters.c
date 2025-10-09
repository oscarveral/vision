#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "filters.h"

/// ---------------------------------------------------------------
/// Implementation of sum_array function
/// ---------------------------------------------------------------

uint32_t sum_array(const uint32_t* array, size_t length) {
	uint32_t sum = 0;
	for (size_t i = 0; i < length; i++) {
		sum += array[i];
	}
	return sum;
}

/// ---------------------------------------------------------------
/// Implementation of box_filter function
/// ---------------------------------------------------------------

uint32_t box_filter(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t filter_size) {
    // Validate filter_size.
    if (filter_size % 2 == 0 || filter_size < 1) {
        return -1;
    }
    // Limit image size to prevent overflows and excessive memory use.
    if (width*height > 4000000UL) {
        return -2;
    }

    // Precomute cumulative sum for efficiency.
    size_t half_size = filter_size / 2;
    uint32_t* cum_sum = (uint32_t*) calloc((width + 1) * (height + 1), sizeof(uint32_t));
    if (!cum_sum) {
        return -3;
    }

    for (size_t y = 1; y <= height; y++) {
        for (size_t x = 1; x <= width; x++) {
            cum_sum[y * (width + 1) + x] = input[(y - 1) * width + (x - 1)]
                                            + cum_sum[(y - 1) * (width + 1) + x]
                                            + cum_sum[y * (width + 1) + (x - 1)]
                                            - cum_sum[(y - 1) * (width + 1) + (x - 1)];
        }
    }

    // Apply box filter using the cumulative sum and mirror padding.
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            size_t x1 = (x < half_size) ? 0 : x - half_size;
            size_t y1 = (y < half_size) ? 0 : y - half_size;
            size_t x2 = (x + half_size >= width) ? width - 1 : x + half_size;
            size_t y2 = (y + half_size >= height) ? height - 1 : y + half_size;

            uint32_t area = (x2 - x1 + 1) * (y2 - y1 + 1);
            uint32_t sum = cum_sum[(y2 + 1) * (width + 1) + (x2 + 1)]
                           - cum_sum[(y1) * (width + 1) + (x2 + 1)]
                           - cum_sum[(y2 + 1) * (width + 1) + (x1)]
                           + cum_sum[(y1) * (width + 1) + (x1)];

            output[y * width + x] = (uint8_t)(sum / area);
        }
    }

    free(cum_sum);
    return 0;
}

/// ---------------------------------------------------------------
/// Helper function to generate 1D Gaussian kernel
/// ---------------------------------------------------------------

static void generate_gaussian_kernel(float* kernel, size_t size, float sigma) {
    int half_size = (int)(size / 2);
    float sum = 0.0f;
    
    // Generate kernel values
    for (int i = 0; i < (int)size; i++) {
        float x = (float)(i - half_size);
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (size_t i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

/// ---------------------------------------------------------------
/// Implementation of gaussian_filter function
/// ---------------------------------------------------------------

int32_t gaussian_filter(const uint8_t* input, uint8_t* output, size_t width, size_t height, float sigma) {
    // Validate input parameters.
    if (!input || !output || width == 0 || height == 0 || sigma <= 0.0f) {
        return -1;
    }
    
    // Calculate kernel size (3*sigma rule, ensure odd size).
    size_t kernel_size = (size_t)(6.0f * sigma + 1.0f);
    if (kernel_size % 2 == 0) kernel_size++;
    if (kernel_size < 3) kernel_size = 3;
    
    // Limit kernel size to prevent excessive memory use.
    if (kernel_size > 101) {
        return -2;
    }
    
    // Limit image size.
    if (width * height > 16000000UL) {
        return -3;
    }
    
    // Allocate memory for kernel and temporary image.
    float* kernel = (float*)malloc(kernel_size * sizeof(float));
    uint8_t* temp_image = (uint8_t*)malloc(width * height * sizeof(uint8_t));
    
    if (!kernel || !temp_image) {
        free(kernel);
        free(temp_image);
        return -4;
    }
    
    // Generate Gaussian kernel.
    generate_gaussian_kernel(kernel, kernel_size, sigma);
    
    int half_size = (int)(kernel_size / 2);
    
    // First pass: horizontal convolution (input -> temp_image).
    #pragma omp parallel for if(height > 64)
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            float sum = 0.0f;
            
            for (int k = 0; k < (int)kernel_size; k++) {
                int src_x = (int)x + k - half_size;
                
                // Mirror boundary conditions.
                if (src_x < 0) src_x = -src_x;
                else if (src_x >= (int)width) src_x = 2 * (int)width - src_x - 1;
                
                sum += (float)input[y * width + src_x] * kernel[k];
            }
            
            temp_image[y * width + x] = (uint8_t)(sum + 0.5f);
        }
    }

    // Second pass: vertical convolution (temp_image -> output).
    #pragma omp parallel for if(width > 64)
    for (size_t x = 0; x < width; x++) {
        for (size_t y = 0; y < height; y++) {
            float sum = 0.0f;
            
            for (int k = 0; k < (int)kernel_size; k++) {
                int src_y = (int)y + k - half_size;
                
                // Mirror boundary conditions.
                if (src_y < 0) src_y = -src_y;
                else if (src_y >= (int)height) src_y = 2 * (int)height - src_y - 1;
                
                sum += (float)temp_image[src_y * width + x] * kernel[k];
            }
            
            output[y * width + x] = (uint8_t)(sum + 0.5f); // Round to nearest
        }
    }
    
    // Clean up
    free(kernel);
    free(temp_image);
    
    return 0;
}
