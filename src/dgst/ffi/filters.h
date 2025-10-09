#ifndef DGST_FILTERS_H
#define DGST_FILTERS_H

#include "stddef.h"
#include "stdint.h"
#include <stdint.h>

/**
 * @brief Computes the sum of all elements in a uint32_t array.
 *
 * This function iterates through the provided array and calculates
 * the sum of all elements.
 *
 * @param array Pointer to the array of uint32_t values to sum.
 *              Must not be NULL if length > 0.
 * @param length Number of elements in the array.
 *
 * @return The sum of all elements in the array as a uint32_t.
 *         Returns 0 if array is NULL or length is 0.
 *
 * @note This function may overflow if the sum exceeds UINT32_MAX.
 *       For large arrays or values, consider using a larger data type.
 *
 * @warning Undefined behavior if array is NULL and length > 0.
 */
uint32_t sum_array(const uint32_t *array, size_t length);

// ...existing code...

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
uint32_t box_filter(const uint8_t *input, uint8_t *output, size_t width,
                    size_t height, size_t filter_size);

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
int32_t gaussian_filter(const uint8_t *input, uint8_t *output, size_t width,
                       size_t height, float sigma);

#endif // DGST_FILTERS_H
