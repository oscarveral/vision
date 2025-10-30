#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "filters.h"

int32_t canny_edge_detection(const uint8_t* input, uint8_t* output, size_t width, size_t height, float low_threshold, float high_threshold) {
	// Validate input parameters.
	if (!input || !output || width == 0 || height == 0) {
		return -1;
	}

	if (high_threshold < low_threshold) {
		return -1;
	}

	// Limit image size.
	if (width * height > 16000000UL) {
		return -2;
	}

	 // Allocate memory for intermediate buffers.
	 float* gradient_magnitude = (float*)malloc(width * height * sizeof(float));
	 float* gradient_direction = (float*)malloc(width * height * sizeof(float));
	 float* nms = (float*)malloc(width * height * sizeof(float));

	if (!gradient_magnitude || !gradient_direction || !nms) {
		free(gradient_magnitude);
		free(gradient_direction);
		free(nms);
		return -3;
	}

	// Initialize buffers.
	memset(gradient_magnitude, 0, width * height * sizeof(float));
	memset(gradient_direction, 0, width * height * sizeof(float));
	memset(nms, 0, width * height * sizeof(float));

// Step 1: Calculate gradients using Sobel operators.
#pragma omp parallel for if (width * height > 64)
	for (size_t y = 1; y < height - 1; y++) {
		for (size_t x = 1; x < width - 1; x++) {
			// Sobel X (horizontal gradient).
			float gx = -(float)input[(y - 1) * width + (x - 1)] + (float)input[(y - 1) * width + (x + 1)] - 2.0f * (float)input[y * width + (x - 1)] +
			  2.0f * (float)input[y * width + (x + 1)] - (float)input[(y + 1) * width + (x - 1)] + (float)input[(y + 1) * width + (x + 1)];

			// Sobel Y (vertical gradient).
			float gy = -(float)input[(y - 1) * width + (x - 1)] - 2.0f * (float)input[(y - 1) * width + x] - (float)input[(y - 1) * width + (x + 1)] +
			  (float)input[(y + 1) * width + (x - 1)] + 2.0f * (float)input[(y + 1) * width + x] + (float)input[(y + 1) * width + (x + 1)];

			// Gradient magnitude and direction.
			gradient_magnitude[y * width + x] = sqrtf(gx * gx + gy * gy);
			gradient_direction[y * width + x] = atan2f(gy, gx);
		}
	}

// Step 2: Non-maximum suppression with linear interpolation.
// Use the same approach as OpenCV for better compatibility.
#pragma omp parallel for if (width * height > 64)
	for (size_t y = 1; y < height - 1; y++) {
		for (size_t x = 1; x < width - 1; x++) {
			float mag = gradient_magnitude[y * width + x];

			if (mag <= 1.0f)
				continue;

			float angle = gradient_direction[y * width + x];

			// Get the direction along gradient (perpendicular to edge).
			float gx = cosf(angle);
			float gy = sinf(angle);

			// Take absolute values for symmetry.
			float abs_gx = fabsf(gx);
			float abs_gy = fabsf(gy);

			float mag1, mag2;

			// Linear interpolation along gradient direction.
			// Use strict greater-than to break ties in favor of the vertical
			// branch when abs_gx == abs_gy (diagonal 45deg). This matches
			// OpenCV's tie-breaking and reduces angle-dependent artifacts.
			if (abs_gx > abs_gy) {
				// More horizontal gradient direction.
				float weight = (abs_gy > 0.0f) ? abs_gy / abs_gx : 0.0f;

				int32_t dx = (gx >= 0) ? 1 : -1;
				int32_t dy = (gy >= 0) ? 1 : -1;

				// Interpolate in positive direction.
				mag1 = (1.0f - weight) * gradient_magnitude[y * width + (x + dx)] + weight * gradient_magnitude[(y + dy) * width + (x + dx)];

				// Interpolate in negative direction.
				mag2 = (1.0f - weight) * gradient_magnitude[y * width + (x - dx)] + weight * gradient_magnitude[(y - dy) * width + (x - dx)];
			} else {
				// More vertical gradient direction.
				float weight = (abs_gx > 0.0f) ? abs_gx / abs_gy : 0.0f;

				int32_t dx = (gx >= 0) ? 1 : -1;
				int32_t dy = (gy >= 0) ? 1 : -1;

				// Interpolate in positive direction.
				mag1 = (1.0f - weight) * gradient_magnitude[(y + dy) * width + x] + weight * gradient_magnitude[(y + dy) * width + (x + dx)];

				// Interpolate in negative direction.
				mag2 = (1.0f - weight) * gradient_magnitude[(y - dy) * width + x] + weight * gradient_magnitude[(y - dy) * width + (x - dx)];
			}

			// Keep only if this pixel is a strict local maximum. Use strict
			// greater-than to avoid keeping tied neighbors which can produce
			// thicker edges at diagonal directions.
			if (mag > mag1 && mag > mag2) {
				nms[y * width + x] = mag;
			}
		}
	}

	// Step 3: Double thresholding.
	// Initialize output with zeros.
	memset(output, 0, width * height);

	// Mark strong edges (above high threshold).
	for (size_t y = 0; y < height; y++) {
		for (size_t x = 0; x < width; x++) {
			if (nms[y * width + x] >= high_threshold) {
				output[y * width + x] = 255;
			}
		}
	}

	// Step 4: Edge tracking by hysteresis using stack-based approach.
	// This is more efficient than iterative relaxation.
	typedef struct {
		size_t x, y;
	} Point;

	Point* stack = (Point*)malloc(width * height * sizeof(Point));
	if (!stack) {
		free(gradient_magnitude);
		free(gradient_direction);
		free(nms);
		return -3;
	}

	size_t stack_size = 0;

	// Initialize stack with all strong edges.
	for (size_t y = 1; y < height - 1; y++) {
		for (size_t x = 1; x < width - 1; x++) {
			if (output[y * width + x] == 255) {
				stack[stack_size].x = x;
				stack[stack_size].y = y;
				stack_size++;
			}
		}
	}

	// Process stack: trace connected weak edges.
	while (stack_size > 0) {
		stack_size--;
		size_t cx = stack[stack_size].x;
		size_t cy = stack[stack_size].y;

		// Check 8-connected neighbors.
		for (int32_t dy = -1; dy <= 1; dy++) {
			for (int32_t dx = -1; dx <= 1; dx++) {
				if (dx == 0 && dy == 0)
					continue;

				size_t nx = cx + dx;
				size_t ny = cy + dy;

				// Check if neighbor is weak edge and not yet marked.
				if (nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1) {
					if (nms[ny * width + nx] >= low_threshold && output[ny * width + nx] == 0) {
						output[ny * width + nx] = 255;
						stack[stack_size].x		= nx;
						stack[stack_size].y		= ny;
						stack_size++;
					}
				}
			}
		}
	}

	free(stack);

	// Clean up.
	free(gradient_magnitude);
	free(gradient_direction);
	free(nms);

	return 0;
}
