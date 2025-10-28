#ifndef DGST_FILTERS_H
#define DGST_FILTERS_H

#include "stddef.h"
#include "stdint.h"
#include <stdint.h>

/**
 * @brief Applies a box filter to a 2D grayscale image using integral image
 * optimization.
 *
 * This function performs box filtering (simple blur) on a grayscale image.
 * The filter uses mirror padding at image boundaries to handle edge cases.
 *
 * @param input Pointer to the input grayscale image data (row-major order).
 *              Each pixel is a uint8_t value (0-255). Must not be NULL.
 * @param output Pointer to the output buffer for filtered image data.
 *               Must be pre-allocated with size width * height bytes.
 *               Must not be NULL and should not overlap with input.
 * @param width Width of the image in pixels. Must be > 0.
 * @param height Height of the image in pixels. Must be > 0.
 * @param filter_size Size of the square box filter kernel. Must be odd and
 * >= 1. Larger values create stronger blur effect.
 *
 * @return Returns 0 on success, negative values on error:
 *         -1: Invalid filter_size (even number or < 1)
 *         -2: Image too large (width * height > 4,000,000 pixels)
 *         -3: Memory allocation failure for internal buffers
 *
 * @note This function allocates temporary memory proportional to image size.
 *       Time complexity: O(width * height), independent of filter_size.
 *       Space complexity: O(width * height) for integral image buffer.
 *
 * @warning Undefined behavior if input/output pointers are NULL or if
 *          output buffer is smaller than width * height bytes.
 */
int32_t box_filter(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t filter_size);

/**
 * @brief Applies a Gaussian blur filter to a 2D grayscale image using separable convolution.
 *
 * This function performs Gaussian filtering (smooth blur) on a grayscale image using
 * an optimized separable convolution approach. The filter decomposes the 2D Gaussian
 * kernel into two 1D passes (horizontal + vertical) for improved performance.
 * Uses mirror padding at image boundaries to prevent edge artifacts.
 *
 * @param input Pointer to the input grayscale image data (row-major order).
 *              Each pixel is a uint8_t value (0-255). Must not be NULL.
 * @param output Pointer to the output buffer for filtered image data.
 *               Must be pre-allocated with size width * height bytes.
 *               Must not be NULL and should not overlap with input.
 * @param width Width of the image in pixels. Must be > 0.
 * @param height Height of the image in pixels. Must be > 0.
 * @param sigma Standard deviation of the Gaussian kernel. Must be > 0.0f.
 *              Larger values create stronger blur effect.
 *              Typical range: 0.5f to 10.0f for most applications.
 *
 * @return Returns 0 on success, negative values on error:
 *         -1: Invalid parameters (NULL pointers, zero dimensions, or sigma <= 0)
 *         -2: Kernel size too large (> 101 pixels, occurs when sigma > 16.7)
 *         -3: Image too large (width * height > 16,000,000 pixels)
 *         -4: Memory allocation failure for temporary buffers
 *
 * @note This function uses separable convolution for efficiency:
 *       - Kernel size automatically calculated as (6*sigma + 1), always odd
 *       - Time complexity: O(width * height * kernel_size), not O(kernel_size²)
 *       - Space complexity: O(width * height) for temporary image buffer
 *       - Parallelized using OpenMP for images larger than 64 pixels
 *       - Mirror boundary conditions preserve edge continuity
 *
 * @warning Undefined behavior if input/output pointers are NULL or if
 *          output buffer is smaller than width * height bytes.
 */
int32_t gaussian_filter(const uint8_t* input, uint8_t* output, size_t width, size_t height, float sigma);

/**
 * @brief Applies Canny edge detection algorithm to a pre-smoothed grayscale image.
 *
 * This function implements the Canny edge detection algorithm optimized for better
 * results and OpenCV compatibility:
 * 1. Gradient calculation using Sobel operators (3x3 kernels for Gx and Gy)
 * 2. Non-maximum suppression with linear interpolation along gradient direction
 * 3. Double thresholding to classify strong/weak edges
 * 4. Edge tracking by hysteresis using efficient stack-based approach
 *
 * NOTE: This function expects a pre-smoothed input image. For best results, apply
 * Gaussian smoothing (sigma ~1.4) to the image before calling this function.
 *
 * @param input Pointer to the input grayscale image data (row-major order).
 *              Should be pre-smoothed (e.g., with Gaussian filter).
 *              Each pixel is a uint8_t value (0-255). Must not be NULL.
 * @param output Pointer to the output buffer for binary edge map.
 *               Must be pre-allocated with size width * height bytes.
 *               Output is binary: 255 for detected edges, 0 for non-edges.
 *               Must not be NULL and should not overlap with input.
 * @param width Width of the image in pixels. Must be > 0.
 * @param height Height of the image in pixels. Must be > 0.
 * @param low_threshold Lower threshold for edge detection. Must be >= 0.0f.
 *                      Pixels with gradient magnitude between low_threshold
 *                      and high_threshold are classified as weak edges and
 *                      only kept if connected to strong edges via hysteresis.
 *                      Typical range: 20.0f to 50.0f.
 * @param high_threshold Upper threshold for edge detection. Must be >= low_threshold.
 *                       Pixels with gradient magnitude >= high_threshold are
 *                       automatically classified as strong edges.
 *                       Typical range: 50.0f to 150.0f.
 *                       Recommended ratio: high_threshold = 2-3 * low_threshold.
 *
 * @return Returns 0 on success, negative values on error:
 *         -1: Invalid parameters (NULL pointers, zero dimensions, high < low)
 *         -2: Image too large (width * height > 16,000,000 pixels)
 *         -3: Memory allocation failure for temporary buffers
 *
 * @note Algorithm implementation details:
 *       - Sobel operators: 3x3 kernels for horizontal/vertical gradients
 *       - Non-maximum suppression: Linear interpolation along gradient direction
 *       - Hysteresis: Stack-based edge tracing with 8-connectivity
 *       - Parallelized with OpenMP for gradient and NMS steps (if image > 64 pixels)
 *       - Image boundaries (1-pixel border) are left as 0 (non-edges)
 *       - Time complexity: O(width * height)
 *       - Space complexity: O(width * height) for intermediate buffers
 *
 * @warning Undefined behavior if input/output pointers are NULL or if
 *          output buffer is smaller than width * height bytes.
 *          For best results, pre-smooth the input image with Gaussian filter.
 */
int32_t canny_edge_detection(const uint8_t* input, uint8_t* output, size_t width, size_t height, float low_threshold, float high_threshold);

/**
 * @brief Applies Kannala-Brandt fisheye undistortion to a grayscale or color image.
 *
 * This function undistorts fisheye camera images using the Kannala-Brandt camera model.
 * It generates undistortion maps and applies remapping with bilinear interpolation.
 *
 * @param input Pointer to the input image data (row-major order).
 *              Each pixel is a uint8_t value (0-255). Must not be NULL.
 * @param output Pointer to the output buffer for undistorted image data.
 *               Must be pre-allocated with size width * height * channels bytes.
 *               Must not be NULL and should not overlap with input.
 * @param width Width of the image in pixels. Must be > 0.
 * @param height Height of the image in pixels. Must be > 0.
 * @param channels Number of color channels (1 for grayscale, 3 for RGB). Must be 1 or 3.
 * @param intrinsics_3x3 Pointer to camera intrinsic matrix (3x3, row-major order).
 *                       Contains [fx, 0, cx; 0, fy, cy; 0, 0, 1].
 *                       Must not be NULL.
 * @param distortion_4 Pointer to array of 4 distortion coefficients [k1, k2, k3, k4].
 *                     Must not be NULL.
 *
 * @return Returns 0 on success, negative values on error:
 *         -1: Invalid parameters (NULL pointers, zero dimensions, invalid channels)
 *         -2: Image too large (width * height > 16,000,000 pixels)
 *         -3: Memory allocation failure for temporary buffers
 *
 * @note Implementation details:
 *       - Uses Kannala-Brandt polynomial model for fisheye distortion
 *       - Bilinear interpolation for remapping
 *       - Parallelized with OpenMP for better performance
 *       - Time complexity: O(width * height * channels)
 *       - Space complexity: O(width * height * 2) for mapping tables
 *
 * @warning Undefined behavior if input/output pointers are NULL or if
 *          output buffer is smaller than width * height * channels bytes.
 */
int32_t kannala_brandt_undistort(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t channels, const float* intrinsics_3x3, const float* distortion_4);

#endif // DGST_FILTERS_H
