#include <stdint.h>
#include <stdlib.h>
#include <math.h>

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

int32_t box_filter(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t filter_size) {
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
    int32_t half_size = (int32_t)(size / 2);
    float sum = 0.0f;
    
    // Generate kernel values
    for (int32_t i = 0; i < (int32_t)size; i++) {
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
    
    // Allocate memory for kernel, normalized input, and temporary image.
    float* kernel = (float*)malloc(kernel_size * sizeof(float));
    float* normalized_input = (float*)malloc(width * height * sizeof(float));
    float* temp_image = (float*)malloc(width * height * sizeof(float));
    
    if (!kernel || !normalized_input || !temp_image) {
        free(kernel);
        free(normalized_input);
        free(temp_image);
        return -4;
    }
    
    // Preprocess: Convert uint8_t input to normalized float [0.0, 1.0].
    #pragma omp parallel for if (width > 64)
    for (size_t i = 0; i < width * height; i++) {
        normalized_input[i] = (float)input[i] / 255.0f;
    }
    
    // Generate Gaussian kernel.
    generate_gaussian_kernel(kernel, kernel_size, sigma);
    
    int32_t half_size = (int32_t)(kernel_size / 2);
    
    // First pass: horizontal convolution (normalized_input -> temp_image).
    #pragma omp parallel for if(height > 64)
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            float sum = 0.0f;
            
            for (size_t k = 0; k < kernel_size; k++) {
                int32_t src_x = (int32_t)x + (int32_t)k - half_size;
                
                // Mirror boundary conditions.
                if (src_x < 0) src_x = -src_x;
                else if (src_x >= (int32_t)width) src_x = 2 * (int32_t)width - src_x - 1;
                
                sum += normalized_input[y * width + (size_t)src_x] * kernel[k];
            }
            
            temp_image[y * width + x] = sum;
        }
    }

    // Second pass: vertical convolution (temp_image -> output).
    #pragma omp parallel for if(width > 64)
    for (size_t x = 0; x < width; x++) {
        for (size_t y = 0; y < height; y++) {
            float sum = 0.0f;
            
            for (size_t k = 0; k < kernel_size; k++) {
                int32_t src_y = (int32_t)y + (int32_t)k - half_size;
                
                // Mirror boundary conditions.
                if (src_y < 0) src_y = -src_y;
                else if (src_y >= (int32_t)height) src_y = 2 * (int32_t)height - src_y - 1;
                
                sum += temp_image[(size_t)src_y * width + x] * kernel[k];
            }
            
            // Convert back to uint8_t [0, 255] and clamp to valid range.
            float normalized_result = sum * 255.0f;
            if (normalized_result < 0.0f) normalized_result = 0.0f;
            if (normalized_result > 255.0f) normalized_result = 255.0f;
            output[y * width + x] = (uint8_t)(normalized_result + 0.5f);
        }
    }
    
    // Clean up.
    free(kernel);
    free(normalized_input);
    free(temp_image);
    
    return 0;
}
