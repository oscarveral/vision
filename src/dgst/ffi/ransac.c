#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "filters.h"

static inline float point_line_distance(float x0, float y0, 
                                        float a, float b, float c) {
    return fabsf(a * x0 + b * y0 + c) / sqrtf(a * a + b * b);
}

int32_t ransac_line_fitting(
    const bool* input, 
    size_t width,
    size_t height,
    float distance_threshold, 
    uint32_t max_iterations,
    uint32_t max_lsq_iterations, 
    uint32_t min_inlier_count,
    float* a, float* b, float* c) {

    // Validate input parameters.
    if (!input || !a || !b || !c || width == 0 || height == 0 || 
        distance_threshold <= 0.0f || max_iterations == 0 || 
        min_inlier_count == 0) {
        return -1;
    }

    // Limit image size.
    if (width * height > 16000000UL) {
        return -2;
    } 

    // Allocate memory for edge point coordinates.
    float* xs = (float*) malloc(width * height * sizeof(float));
    float* ys = (float*) malloc(width * height * sizeof(float));
    if (!xs || !ys) {
        free(xs); free(ys);
        return -3;
    }

    // Count edge points and store their coordinates.
    size_t point_count = 0;
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            if (input[y * width + x]) {
                xs[point_count] = (float)x;
                ys[point_count] = (float)y;
                point_count++;
            }
        }
    }

    if (point_count < 2) {
        free(xs); free(ys);
        return -4;
    }

    // RANSAC loop.
    srand((unsigned) time(NULL));

    size_t best_inlier_count = 0;
    float best_a = 0.0f, best_b = 0.0f, best_c = 0.0f;

    for (uint32_t iter = 0; iter < max_iterations; iter++) {
        // Randomly select two distinct points.
        size_t idx1 = rand() % point_count;
        size_t idx2 = rand() % point_count;
        while (idx2 == idx1) {
            // Avoid selecting the same point.
            idx2 = rand() % point_count;
        }

        float x1 = xs[idx1];
        float y1 = ys[idx1];
        float x2 = xs[idx2];
        float y2 = ys[idx2];

        // Compute line coefficients Ax + By + C = 0.
        float A = y2 - y1;
        float B = x1 - x2;
        float C = x2 * y1 - x1 * y2;

        // Count inliers.
        size_t inlier_count = 0;
        for (size_t i = 0; i < point_count; i++) {
            float dist = point_line_distance(xs[i], ys[i], A, B, C);
            if (dist <= distance_threshold) {
                inlier_count++;
            }
        }

        // Update best line if current one is better.
        if (inlier_count > best_inlier_count && inlier_count >= min_inlier_count) {
            best_inlier_count = inlier_count;
            best_a = A;
            best_b = B;
            best_c = C;
        }
    }

    if (best_inlier_count == 0) {
        // No valid line found.
        free(xs); free(ys);
        return -5;
    }

    // Optional least squares refinement.
    if (max_lsq_iterations > 0) {
        for (uint32_t iter = 0; iter < max_lsq_iterations; iter++) {
            // Detect vertical lines to avoid division by zero.
            bool vertical = fabsf(best_b) < 1e-6f;

            // Accumulate sums for least squares.
            float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
            size_t inlier_count = 0;

            // Loop over points to find inliers.
            for (size_t i = 0; i < point_count; i++) {
                float x = xs[i];
                float y = ys[i];
                float dist = point_line_distance(xs[i], ys[i], best_a, best_b, best_c);
                
                if (dist <= distance_threshold) {

                    if (!vertical) {
                        // For non-vertical lines, fit y = mx + b.
                        sum_x += x;
                        sum_y += y;
                        sum_xy += x * y;
                        sum_x2 += x * x;
                    } else {
                        // For vertical lines, fit x = my + b.
                        sum_x += y;
                        sum_y += x;
                        sum_xy += y * x;
                        sum_x2 += y * y;
                    }
                    inlier_count++;
                }
            }

            if (inlier_count < 2) {
                break; // Not enough inliers to refine.
            }

            if (inlier_count < best_inlier_count) {
                // Stop if inlier count decreased.
                break;
            }

            float N = (float)inlier_count;
            float denominator = N * sum_x2 - sum_x * sum_x;
            if (fabsf(denominator) < 1e-6f) {
                break; // Avoid division by zero.
            }

            float m = (N * sum_xy - sum_x * sum_y) / denominator;
            float k = (sum_y - m * sum_x) / N;

            if (!vertical) {
                // Update line coefficients for y = mx + k.
                best_a = m;
                best_b = -1.0f;
                best_c = k;
            } else {
                // Update line coefficients for x = my + k.
                best_a = -1.0f;
                best_b = m;
                best_c = k;
            }
        }
    }

    free(xs); free(ys);
    *a = best_a;
    *b = best_b;
    *c = best_c;
    return 0;
}

int32_t ransac_circle_fitting(
    const bool* input, 
    size_t width,
    size_t height,
    float distance_threshold, 
    uint32_t max_iterations,
    float min_inlier_ratio,
    float min_radius,
    float max_radius,
    float* center_x, 
    float* center_y, 
    float* radius) {

    // Validate input parameters.
    if (!input || !center_x || !center_y || !radius || width == 0 || height == 0 || 
        distance_threshold <= 0.0f || max_iterations == 0 || 
        min_inlier_ratio <= 0.0f || max_radius <= min_radius || min_radius < 0.0f) {
        return -1;
    }

    // Limit image size.
    if (width * height > 16000000UL) {
        return -2;
    } 

    // Allocate memory for edge point coordinates.
    float* xs = (float*) malloc(width * height * sizeof(float));
    float* ys = (float*) malloc(width * height * sizeof(float));
    if (!xs || !ys) {
        free(xs); free(ys);
        return -3;
    }

    // Count edge points and store their coordinates.
    size_t point_count = 0;
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            if (input[y * width + x]) {
                xs[point_count] = (float)x;
                ys[point_count] = (float)y;
                point_count++;
            }
        }
    }

    if (point_count < 3) {
        free(xs); free(ys);
        return -4;
    }

    // RANSAC loop.
    srand((unsigned) time(NULL));

    float best_inlier_ratio = 0.0f;
    float best_cx = 0.0f, best_cy = 0.0f, best_r = 0.0f;
    
    for (size_t i=0; i<max_iterations; i++) {
        size_t idx1 = rand() % point_count;
        size_t idx2 = rand() % point_count;
        while (idx2 == idx1) {
            idx2 = rand() % point_count;
        }
        size_t idx3 = rand() % point_count;
        while (idx3 == idx1 || idx3 == idx2) {
            idx3 = rand() % point_count;
        }
        float x1 = xs[idx1], y1 = ys[idx1];
        float x2 = xs[idx2], y2 = ys[idx2];
        float x3 = xs[idx3], y3 = ys[idx3];

        // Check for collinearity.
        float A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2;
        if (fabsf(A) < 1e-6f) {
            continue; // Points are collinear.
        }

        // Compute circle center and radius.
        float B = (x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (x3 * x3 + y3 * y3) * (y2 - y1);
        float C = (x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (x3 * x3 + y3 * y3) * (x1 - x2);
        float cx = -B / (2 * A);
        float cy = -C / (2 * A);
        float r = sqrtf((cx - x1) * (cx - x1) + (cy - y1) * (cy - y1));

        if (r > max_radius || r < min_radius) {
            continue; // Skip circles that exceed max radius and min radius.
        }

        // Count inliers.
        size_t inlier_count = 0;
        for (size_t j = 0; j < point_count; j++) {
            float dist = fabsf(sqrtf((xs[j] - cx) * (xs[j] - cx) + (ys[j] - cy) * (ys[j] - cy)) - r);
            if (dist <= distance_threshold) {
                inlier_count++;
            }
        }

        // Update best circle if current one is better.
        float inlier_ratio = (float)inlier_count / r;
        if (inlier_ratio > best_inlier_ratio) {
            best_inlier_ratio = inlier_ratio;
            best_cx = cx;
            best_cy = cy;
            best_r = r;
        }
    }

    // Check if we found a valid circle.
    if (best_inlier_ratio >= min_inlier_ratio) {
        *center_x = best_cx;
        *center_y = best_cy;
        *radius = best_r;
        free(xs); free(ys);
        return 0;
    }

    free(xs); free(ys);
    return -5;
}
