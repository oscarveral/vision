#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "filters.h"

/**
 * @brief Bilinear interpolation for image remapping.
 * @param input Source image data.
 * @param width Width of the source image.
 * @param height Height of the source image.
 * @param channels Number of color channels.
 * @param x X coordinate (can be fractional).
 * @param y Y coordinate (can be fractional).
 * @param output Output pixel value(s).
 */
static void bilinear_interpolate(const uint8_t* input, size_t width, size_t height, size_t channels, float x, float y, uint8_t* output) {
	// Boundary check.
	if (x < 0.0f || x >= (float)(width - 1) || y < 0.0f || y >= (float)(height - 1)) {
		// Out of bounds - set to black.
		for (size_t c = 0; c < channels; c++) {
			output[c] = 0;
		}
		return;
	}

	// Get the four surrounding pixels.
	int32_t x0 = (int32_t)floorf(x);
	int32_t y0 = (int32_t)floorf(y);
	int32_t x1 = x0 + 1;
	int32_t y1 = y0 + 1;

	// Compute interpolation weights.
	float dx = x - (float)x0;
	float dy = y - (float)y0;

	float w00 = (1.0f - dx) * (1.0f - dy);
	float w01 = (1.0f - dx) * dy;
	float w10 = dx * (1.0f - dy);
	float w11 = dx * dy;

	// Interpolate for each channel.
	for (size_t c = 0; c < channels; c++) {
		float val = w00 * (float)input[(y0 * width + x0) * channels + c] + w01 * (float)input[(y1 * width + x0) * channels + c] +
		  w10 * (float)input[(y0 * width + x1) * channels + c] + w11 * (float)input[(y1 * width + x1) * channels + c];

		output[c] = (uint8_t)(val + 0.5f); // Round to nearest integer.
	}
}

/**
 * @brief Apply Kannala-Brandt distortion to find distorted coordinates from undistorted ones.
 * This maps from ideal pinhole coordinates to the actual distorted sensor coordinates.
 *
 * @param x_undist Undistorted x coordinate (normalized, in camera plane).
 * @param y_undist Undistorted y coordinate (normalized, in camera plane).
 * @param k Distortion coefficients [k1, k2, k3, k4].
 * @param x_dist Output distorted x coordinate.
 * @param y_dist Output distorted y coordinate.
 */
static void kannala_brandt_project(float x_undist, float y_undist, const float* k, float* x_dist, float* y_dist) {
	// Compute radius in normalized plane
	float r = sqrtf(x_undist * x_undist + y_undist * y_undist);

	if (r < 1e-8f) {
		*x_dist = x_undist;
		*y_dist = y_undist;
		return;
	}

	// Compute angle theta (angle from optical axis).
	float theta	 = atanf(r);
	float theta2 = theta * theta;
	float theta4 = theta2 * theta2;
	float theta6 = theta4 * theta2;
	float theta8 = theta4 * theta4;

	// Kannala-Brandt distortion model: theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8).
	float theta_d = theta * (1.0f + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8);

	// Project to distorted coordinates.
	float scale = theta_d / r;

	*x_dist = x_undist * scale;
	*y_dist = y_undist * scale;
}

/**
 * Compute theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
 * where theta = atan(r).
 */
static float compute_theta_d_from_r(float r, const float* k) {
	float theta = atanf(r);
	float theta2 = theta * theta;
	float theta4 = theta2 * theta2;
	float theta6 = theta4 * theta2;
	float theta8 = theta4 * theta4;
	return theta * (1.0f + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8);
}

int32_t kannala_brandt_undistort(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t channels, const float* intrinsics_3x3, const float* distortion_4) {
	// Validate input parameters.
	if (!input || !output || !intrinsics_3x3 || !distortion_4) {
		return -1;
	}

	if (width == 0 || height == 0 || (channels != 1 && channels != 3)) {
		return -1;
	}

	// Limit image size.
	if (width * height > 16000000UL) {
		return -2;
	}

	// Extract intrinsic parameters (3x3 matrix in row-major order).
	// [fx  0  cx]
	// [ 0 fy  cy]
	// [ 0  0   1]
	float fx = intrinsics_3x3[0];
	float cx = intrinsics_3x3[2];
	float fy = intrinsics_3x3[4];
	float cy = intrinsics_3x3[5];

	// Allocate mapping tables.
	float* map_x = (float*)malloc(width * height * sizeof(float));
	float* map_y = (float*)malloc(width * height * sizeof(float));

	if (!map_x || !map_y) {
		free(map_x);
		free(map_y);
		return -3;
	}

// Build undistortion mapping.
// For each output (undistorted) pixel, find which input (distorted) pixel to sample from.
#pragma omp parallel for if (height > 64)
	for (size_t v = 0; v < height; v++) {
		for (size_t u = 0; u < width; u++) {
			// Output pixel coordinates -> normalized camera plane (undistorted).
			float x_undist = ((float)u - cx) / fx;
			float y_undist = ((float)v - cy) / fy;

			// Apply distortion model to find where this point came from in the distorted image.
			float x_dist, y_dist;
			kannala_brandt_project(x_undist, y_undist, distortion_4, &x_dist, &y_dist);

			// Convert back to pixel coordinates in the distorted (input) image.
			float u_src = x_dist * fx + cx;
			float v_src = y_dist * fy + cy;

			map_x[v * width + u] = u_src;
			map_y[v * width + u] = v_src;
		}
	}

// Apply remapping with bilinear interpolation.
#pragma omp parallel for if (height > 64)
	for (size_t v = 0; v < height; v++) {
		for (size_t u = 0; u < width; u++) {
			float src_x = map_x[v * width + u];
			float src_y = map_y[v * width + u];

			uint8_t pixel[3];
			bilinear_interpolate(input, width, height, channels, src_x, src_y, pixel);

			// Copy to output.
			for (size_t c = 0; c < channels; c++) {
				output[(v * width + u) * channels + c] = pixel[c];
			}
		}
	}

	// Clean up.
	free(map_x);
	free(map_y);

	return 0;
}


/**
 * Map an array of points from distorted (input) pixel coordinates into
 * undistorted (output) pixel coordinates using the inverse Kannala-Brandt mapping.
 *
 * points_in/out are arrays of floats with layout [u0, v0, u1, v1, ...].
 */
int32_t kannala_brandt_map_points_to_undistorted(const float* points_in, float* points_out, size_t n_points, const float* intrinsics_3x3, const float* distortion_4) {
	if (!points_in || !points_out || !intrinsics_3x3 || !distortion_4) {
		return -1;
	}

	// Extract intrinsics
	float fx = intrinsics_3x3[0];
	float cx = intrinsics_3x3[2];
	float fy = intrinsics_3x3[4];
	float cy = intrinsics_3x3[5];

	// Helper: compute f(r) = theta_d where theta = atan(r) using compute_theta_d_from_r

	// For each point, invert r_dist = theta_d(r) to find r (undistorted normalized radius)
	for (size_t i = 0; i < n_points; i++) {
		float u_dist = points_in[2 * i + 0];
		float v_dist = points_in[2 * i + 1];

		// normalized distorted coords
		float x_dist = (u_dist - cx) / fx;
		float y_dist = (v_dist - cy) / fy;
		float r_dist = sqrtf(x_dist * x_dist + y_dist * y_dist);

		if (r_dist < 1e-8f) {
			// Map to center
			points_out[2 * i + 0] = u_dist;
			points_out[2 * i + 1] = v_dist;
			continue;
		}

		// Binary search to find r such that compute_theta_d(r) ~= r_dist
		float low = 0.0f;
		float high = (r_dist > 1.0f) ? r_dist : 1.0f;
		// Increase high until f(high) >= r_dist or until a large limit
	float fhigh = compute_theta_d_from_r(high, distortion_4);
		int safety = 0;
		while (fhigh < r_dist && safety < 60) {
			high *= 2.0f;
			fhigh = compute_theta_d_from_r(high, distortion_4);
			safety++;
			if (high > 1e6f) break;
		}

		// Binary search for ~1e-6 precision
		float mid = 0.0f;
		for (int iter = 0; iter < 60; ++iter) {
			mid = 0.5f * (low + high);
			float fmid = compute_theta_d_from_r(mid, distortion_4);
			float diff = fmid - r_dist;
			if (fabsf(diff) < 1e-6f) break;
			if (fmid < r_dist) {
				low = mid;
			} else {
				high = mid;
			}
		}

		float r_undist = mid;

		// Compute undistorted normalized coords preserving direction
		float scale = (r_undist / r_dist);
		float x_undist = x_dist * scale;
		float y_undist = y_dist * scale;

		// Convert back to pixel coordinates in undistorted image
		float u_undist = x_undist * fx + cx;
		float v_undist = y_undist * fy + cy;

		points_out[2 * i + 0] = u_undist;
		points_out[2 * i + 1] = v_undist;
	}

	return 0;
}
