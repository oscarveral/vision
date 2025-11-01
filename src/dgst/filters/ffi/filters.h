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

/**
 * @brief Map points from distorted image pixel coordinates into undistorted image pixel coordinates.
 *
 * The function expects points in pixel coordinates (u, v) in the distorted image and
 * computes their corresponding (u, v) coordinates in the undistorted image using
 * the inverse of the Kannala-Brandt mapping. Points are provided as an array of
 * floats [u0, v0, u1, v1, ..., uN-1, vN-1]. The output array must have the same
 * length and will be filled with the mapped coordinates.
 *
 * @param points_in Pointer to input points array (length = n_points * 2). Must not be NULL.
 * @param points_out Pointer to output points array (length = n_points * 2). Must not be NULL.
 * @param n_points Number of points to map.
 * @param intrinsics_3x3 Pointer to camera intrinsic matrix (3x3, row-major order).
 * @param distortion_4 Pointer to array of 4 distortion coefficients [k1, k2, k3, k4].
 * @return 0 on success, negative on error.
 */
int32_t kannala_brandt_map_points_to_undistorted(const float* points_in, float* points_out, size_t n_points, const float* intrinsics_3x3, const float* distortion_4);

/**
 * @brief Computes phase congruency map for edge and feature detection.
 *
 * This function implements a phase congruency algorithm inspired by Kovesi's method.
 * It builds log-Gabor band-pass filters across multiple scales and orientations in
 * the frequency domain, computes quadrature responses via FFT, forms local energy,
 * and normalizes to produce a phase congruency map.
 *
 * Phase congruency is a dimensionless measure of feature significance that is invariant
 * to changes in image brightness or contrast, making it more robust than traditional
 * gradient-based edge detection methods.
 *
 * @param input Pointer to the input grayscale image data (row-major order).
 *              Each pixel is a uint8_t value (0-255). Must not be NULL.
 * @param output Pointer to the output buffer for phase congruency map.
 *               Must be pre-allocated with size width * height bytes.
 *               Output is uint8_t with values in [0, 255] where higher values indicate stronger features.
 *               Must not be NULL and should not overlap with input.
 * @param width Width of the image in pixels. Must be > 0 and a power of 2.
 * @param height Height of the image in pixels. Must be > 0 and a power of 2.
 * @param nscale Number of wavelet scales to use. Must be > 0.
 *               More scales provide better feature detection but increase computation.
 *               Typical value: 4.
 * @param norient Number of filter orientations. Must be > 0.
 *                More orientations provide better angular resolution but increase computation.
 *                Typical value: 6.
 * @param min_wavelength Wavelength of smallest scale filter in pixels. Must be > 0.
 *                       Should be >= 3.0 to avoid aliasing.
 *                       Typical value: 3.0.
 * @param mult Scaling factor between successive filters. Must be > 1.0.
 *             Each scale has wavelength mult times the previous scale.
 *             Typical value: 2.1.
 * @param sigma_onf Ratio of the standard deviation of the Gaussian describing
 *                  the log Gabor filter's transfer function in the frequency
 *                  domain to the filter center frequency. Must be > 0.
 *                  Controls filter bandwidth. Typical value: 0.55.
 * @param k Noise compensation gain factor. Must be >= 0.
 *          Used in noise energy estimation. Typical value: 2.0.
 * @param cut_off Fractional measure of frequency spread below which
 *                phase congruency values get penalized. Typical value: 0.5.
 * @param g Gain factor for sigmoid function used in noise compensation.
 *          Typical value: 10.0.
 * @param eps Small constant to avoid division by zero. Must be > 0.
 *            Typical value: 1e-8.
 *
 * @return Returns 0 on success, negative values on error:
 *         -1: Invalid parameters (NULL pointers, zero dimensions, non-power-of-2 size,
 *             or invalid algorithm parameters)
 *         -2: Image too large (width * height > 16,000,000 pixels)
 *         -3: Memory allocation failure for temporary buffers
 *
 * @note Implementation details:
 *       - Requires image dimensions to be powers of 2 for FFT
 *       - Uses Cooley-Tukey FFT algorithm (no external dependencies)
 *       - Log-Gabor filters provide better frequency coverage than Gabor
 *       - Parallelized with OpenMP for major computation steps
 *       - Time complexity: O(width * height * nscale * norient * log(width * height))
 *       - Space complexity: O(width * height) for intermediate buffers
 *       - Output is normalized and scaled to full uint8 range (0-255)
 *
 * @warning Undefined behavior if input/output pointers are NULL, if dimensions
 *          are not powers of 2, or if output buffer is smaller than width * height bytes.
 *          For best results, use power-of-2 dimensions (e.g., 256x256, 512x512).
 */
int32_t phase_congruency(const uint8_t* input, uint8_t* output, size_t width, size_t height, int32_t nscale, int32_t norient, float min_wavelength, float mult, float sigma_onf, float eps);

/**
 * @brief Threshold a uint8 image producing a uint8 binary output map with 0 or 255
 *
 * This function thresholds a uint8 image where input pixels are expected in [0,255].
 * Pixels are first normalized to [0,1] (value/255.0) and then compared to threshold.
 * Output pixels are set to 255 if value >= threshold, otherwise 0.
 *
 * @param input Pointer to input uint8 image (row-major), values in [0,255]
 * @param output Pointer to output uint8 image (row-major), same size as input
 * @param width Image width
 * @param height Image height
 * @param threshold Threshold value in [0,1]
 * @return 0 on success, negative on error:
 *         -1: Invalid parameters (NULL pointers, zero dimensions, threshold out of range)
 *         -2: Image too large (width * height > 16,000,000)
 */
int32_t threshold_filter(const uint8_t* input, uint8_t* output, size_t width, size_t height, float threshold);

/**
 * @brief Fits a line to edge points in a binary edge map using the RANSAC algorithm.
 *
 * This function identifies the best-fitting line in a binary edge map by iteratively
 * sampling edge points and evaluating inliers based on a distance threshold. The line
 * is represented in the general form Ax + By + C = 0.
 * @param input Pointer to the input binary edge map (row-major order).
 *                 Each pixel is a bool value (true for edge, false for non-edge). Must not be NULL.
 * @param width Width of the edge map in pixels. Must be > 0.
 * @param height Height of the edge map in pixels. Must be > 0.
 * @param distance_threshold Maximum distance from line to consider a point as an inlier. Must be > 0.0f.
 * @param max_iterations Maximum number of RANSAC iterations to perform. Must be > 0.
 * @param max_lsq_iterations Maximum number of least squares refinement iterations. Set to 0 to skip refinement.
 * @param min_inlier_count Minimum number of inliers required to accept a line. Must be > 0.
 * @param a Pointer to output parameter for line coefficient A. Must not be NULL.
 * @param b Pointer to output parameter for line coefficient B. Must not be NULL.
 * @param c Pointer to output parameter for line coefficient C. Must not be NULL.
 * @return Returns 0 on success, negative values on error:
 *         -1: Invalid parameters (NULL pointers, zero dimensions, non-positive thresholds)
 *         -2: Image too large (width * height > 16,000,000 pixels)
 *         -3: Memory allocation failure for temporary buffers
 *         -4: Not enough edge points to fit a line
 *         -5: No line found with sufficient inliers
 * @note Algorithm details:
 *       - Randomly samples pairs of edge points to define candidate lines
 *       - Counts inliers within distance_threshold for each candidate line
 *       - Retains the line with the highest inlier count exceeding min_inlier_count
 *       - Optionally refines the best line using least squares fitting on inliers
 *       - Time complexity: O(max_iterations * number_of_edge_points + max_lsq_iterations * number_of_inliers)
 *       - Space complexity: O(number_of_edge_points) for storing edge point coordinates
 * @warning Undefined behavior if input/output pointers are NULL or if
 *          output parameters a, b, c are NULL.
 */
int32_t ransac_line_fitting(const bool* input, size_t width, size_t height,
                            float distance_threshold, uint32_t max_iterations,
                            uint32_t max_lsq_iterations, uint32_t min_inlier_count,
                            float* a, float* b, float* c);

/**
 * @brief Fits a circle to edge points in a binary edge map using the RANSAC algorithm.
 *
 * This function identifies the best-fitting circle in a binary edge map by iteratively
 * sampling edge points and evaluating inliers based on a distance threshold. The circle
 * is represented by its center coordinates (center_x, center_y) and radius.
 * @param input Pointer to the input binary edge map (row-major order).
 *                 Each pixel is a bool value (true for edge, false for non-edge). Must not be NULL.
 * @param width Width of the edge map in pixels. Must be > 0.
 * @param height Height of the edge map in pixels. Must be > 0.
 * @param distance_threshold Maximum distance from circle to consider a point as an inlier. Must be > 0.0f.
 * @param max_iterations Maximum number of RANSAC iterations to perform. Must be > 0.
 * @param min_inlier_ratio Minimum ratio of inliers to radius to accept a circle. Recommended minimum is 3.
 * @param min_radius Minimum radius of the circle to be considered valid. Must be >= 0.0f.
 * @param max_radius Maximum radius of the circle to be considered valid. Must be > min_radius.
 * @param center_x Pointer to output parameter for circle center x-coordinate. Must not be NULL.
 * @param center_y Pointer to output parameter for circle center y-coordinate. Must not be NULL.
 * @param radius Pointer to output parameter for circle radius. Must not be NULL.
 * @return Returns 0 on success, negative values on error:
 *         -1: Invalid parameters (NULL pointers, zero dimensions, non-positive thresholds)
 *         -2: Image too large (width * height > 16,000,000 pixels)
 *         -3: Memory allocation failure for temporary buffers
 *         -4: Not enough edge points to fit a circle
 *         -5: No circle found with sufficient inliers
 * @note Algorithm details:
 *       - Randomly samples triplets of edge points to define candidate circles
 *       - Counts inliers within distance_threshold for each candidate circle
 *       - Retains the circle with the highest inlier count exceeding min_inlier_count
 *       - Time complexity: O(max_iterations * number_of_edge_points)
 *       - Space complexity: O(number_of_edge_points) for storing edge point coordinates
 * @warning Undefined behavior if input/output pointers are NULL or if
 *          output parameters center_x, center_y, radius are NULL.
 */
int32_t ransac_circle_fitting(const bool* input, size_t width, size_t height,
                              float distance_threshold, uint32_t max_iterations,
                              float min_inlier_ratio, float min_radius, float max_radius,
                              float* center_x, float* center_y, float* radius);
                            
#endif // DGST_FILTERS_H
