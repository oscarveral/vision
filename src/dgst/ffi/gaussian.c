#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "filters.h"

/**
 * @brief Generates a 1D Gaussian kernel.
 * @param kernel Pointer to the kernel array. Must be pre-allocated with size elements.
 * @param size Size of the kernel (must be odd).
 * @param sigma Standard deviation of the Gaussian distribution.
 */
static void generate_gaussian_kernel(float* kernel, size_t size, float sigma) {
	int32_t half_size = (int32_t)(size / 2);
	float sum		  = 0.0f;

	// Generate kernel values
	for (int32_t i = 0; i < (int32_t)size; i++) {
		float x	  = (float)(i - half_size);
		kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
		sum += kernel[i];
	}

	// Normalize kernel
	for (size_t i = 0; i < size; i++) {
		kernel[i] /= sum;
	}
}

int32_t gaussian_filter(const uint8_t* input, uint8_t* output, size_t width, size_t height, float sigma) {
	// Validate input parameters.
	if (!input || !output || width == 0 || height == 0 || sigma <= 0.0f) {
		return -1;
	}

	// Calculate kernel size (3*sigma rule, ensure odd size).
	size_t kernel_size = (size_t)(6.0f * sigma + 1.0f);
	if (kernel_size % 2 == 0)
		kernel_size++;
	if (kernel_size < 3)
		kernel_size = 3;

	// Limit kernel size to prevent excessive memory use.
	if (kernel_size > 101) {
		return -2;
	}

	// Limit image size.
	if (width * height > 16000000UL) {
		return -3;
	}

	// Allocate memory for kernel, normalized input, and temporary image.
	float* kernel			= (float*)malloc(kernel_size * sizeof(float));
	float* normalized_input = (float*)malloc(width * height * sizeof(float));
	float* temp_image		= (float*)malloc(width * height * sizeof(float));

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
#pragma omp parallel for if (height > 64)
	for (size_t y = 0; y < height; y++) {
		for (size_t x = 0; x < width; x++) {
			float sum = 0.0f;

			for (size_t k = 0; k < kernel_size; k++) {
				int32_t src_x = (int32_t)x + (int32_t)k - half_size;

				// Mirror boundary conditions.
				if (src_x < 0)
					src_x = -src_x;
				else if (src_x >= (int32_t)width)
					src_x = 2 * (int32_t)width - src_x - 1;

				sum += normalized_input[y * width + (size_t)src_x] * kernel[k];
			}

			temp_image[y * width + x] = sum;
		}
	}

// Second pass: vertical convolution (temp_image -> output).
#pragma omp parallel for if (width > 64)
	for (size_t x = 0; x < width; x++) {
		for (size_t y = 0; y < height; y++) {
			float sum = 0.0f;

			for (size_t k = 0; k < kernel_size; k++) {
				int32_t src_y = (int32_t)y + (int32_t)k - half_size;

				// Mirror boundary conditions.
				if (src_y < 0)
					src_y = -src_y;
				else if (src_y >= (int32_t)height)
					src_y = 2 * (int32_t)height - src_y - 1;

				sum += temp_image[(size_t)src_y * width + x] * kernel[k];
			}

			// Convert back to uint8_t [0, 255] and clamp to valid range.
			float normalized_result = sum * 255.0f;
			if (normalized_result < 0.0f)
				normalized_result = 0.0f;
			if (normalized_result > 255.0f)
				normalized_result = 255.0f;
			output[y * width + x] = (uint8_t)(normalized_result + 0.5f);
		}
	}

	// Clean up.
	free(kernel);
	free(normalized_input);
	free(temp_image);

	return 0;
}
