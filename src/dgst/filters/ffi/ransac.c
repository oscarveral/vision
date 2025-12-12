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

    
    static int _ransac_seeded_line = 0;
    if (!_ransac_seeded_line) {
        srand((unsigned) time(NULL));
        _ransac_seeded_line = 1;
    }

    // RANSAC loop.
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

   
    static int _ransac_seeded_circle = 0;
    if (!_ransac_seeded_circle) {
        srand((unsigned) time(NULL));
        _ransac_seeded_circle = 1;
    }

     
    // RANSAC loop.

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

// Helper: compute 3x3 homography from 4 point correspondences using DLT
static int compute_homography_4pt(
    const float* src_pts,  // [x0,y0, x1,y1, x2,y2, x3,y3]
    const float* dst_pts,  // [x0,y0, x1,y1, x2,y2, x3,y3]
    float* H               // output 3x3 homography (row-major)
) {
    // Build 8x9 matrix A for DLT
    // For each correspondence (x,y) -> (x',y'):
    // [-x, -y, -1,  0,  0,  0, x*x', y*x', x']
    // [ 0,  0,  0, -x, -y, -1, x*y', y*y', y']
    
    float A[8][9];
    for (int i = 0; i < 4; i++) {
        float x = src_pts[2*i];
        float y = src_pts[2*i + 1];
        float xp = dst_pts[2*i];
        float yp = dst_pts[2*i + 1];
        
        A[2*i][0] = -x;
        A[2*i][1] = -y;
        A[2*i][2] = -1;
        A[2*i][3] = 0;
        A[2*i][4] = 0;
        A[2*i][5] = 0;
        A[2*i][6] = x * xp;
        A[2*i][7] = y * xp;
        A[2*i][8] = xp;
        
        A[2*i+1][0] = 0;
        A[2*i+1][1] = 0;
        A[2*i+1][2] = 0;
        A[2*i+1][3] = -x;
        A[2*i+1][4] = -y;
        A[2*i+1][5] = -1;
        A[2*i+1][6] = x * yp;
        A[2*i+1][7] = y * yp;
        A[2*i+1][8] = yp;
    }
    
    // Solve using simplified SVD-like approach for 8x9 system
    // We use Gaussian elimination to reduce to null space
    // Copy A to working matrix
    float M[8][9];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 9; j++) {
            M[i][j] = A[i][j];
        }
    }
    
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 8; col++) {
        // Find pivot
        int max_row = col;
        float max_val = fabsf(M[col][col]);
        for (int row = col + 1; row < 8; row++) {
            if (fabsf(M[row][col]) > max_val) {
                max_val = fabsf(M[row][col]);
                max_row = row;
            }
        }
        
        if (max_val < 1e-10f) {
            return -1; // Singular matrix
        }
        
        // Swap rows
        if (max_row != col) {
            for (int j = 0; j < 9; j++) {
                float tmp = M[col][j];
                M[col][j] = M[max_row][j];
                M[max_row][j] = tmp;
            }
        }
        
        // Eliminate below
        for (int row = col + 1; row < 8; row++) {
            float factor = M[row][col] / M[col][col];
            for (int j = col; j < 9; j++) {
                M[row][j] -= factor * M[col][j];
            }
        }
    }
    
    // Back substitution to get null space vector (column 9)
    float h[9];
    h[8] = 1.0f; // Set h9 = 1
    
    for (int i = 7; i >= 0; i--) {
        float sum = M[i][8]; // This is -h[8] coefficient
        for (int j = i + 1; j < 8; j++) {
            sum += M[i][j] * h[j];
        }
        if (fabsf(M[i][i]) < 1e-10f) {
            return -1;
        }
        h[i] = -sum / M[i][i];
    }
    
    // Normalize so h[8] = 1 (if not already)
    if (fabsf(h[8]) > 1e-10f) {
        for (int i = 0; i < 9; i++) {
            H[i] = h[i] / h[8];
        }
    } else {
        for (int i = 0; i < 9; i++) {
            H[i] = h[i];
        }
    }
    
    return 0;
}

// Helper: apply homography to a point
static inline void apply_homography(const float* H, float x, float y, 
                                    float* xp, float* yp) {
    float w = H[6] * x + H[7] * y + H[8];
    if (fabsf(w) < 1e-10f) {
        *xp = 0;
        *yp = 0;
        return;
    }
    *xp = (H[0] * x + H[1] * y + H[2]) / w;
    *yp = (H[3] * x + H[4] * y + H[5]) / w;
}

int32_t ransac_homography_fitting(
    const float* src_points,  // Nx2 source points [x0,y0, x1,y1, ...]
    const float* dst_points,  // Nx2 destination points [x0,y0, x1,y1, ...]
    size_t num_points,
    float distance_threshold,
    uint32_t max_iterations,
    uint32_t min_inlier_count,
    float* homography,        // Output 3x3 homography (row-major, 9 floats)
    bool* inlier_mask         // Output inlier mask (num_points bools), can be NULL
) {
    // Validate input parameters
    if (!src_points || !dst_points || !homography || num_points < 4 ||
        distance_threshold <= 0.0f || max_iterations == 0 || min_inlier_count < 4) {
        return -1;
    }

    if (num_points > 100000) {
        return -2; // Too many points
    }

    static int _ransac_seeded_homography = 0;
    if (!_ransac_seeded_homography) {
        srand((unsigned) time(NULL));
        _ransac_seeded_homography = 1;
    }

    size_t best_inlier_count = 0;
    float best_H[9] = {0};
    
    float sample_src[8]; // 4 points * 2 coords
    float sample_dst[8];
    float H[9];

    for (uint32_t iter = 0; iter < max_iterations; iter++) {
        // Randomly select 4 distinct points
        size_t indices[4];
        for (int i = 0; i < 4; i++) {
            bool unique;
            do {
                unique = true;
                indices[i] = rand() % num_points;
                for (int j = 0; j < i; j++) {
                    if (indices[i] == indices[j]) {
                        unique = false;
                        break;
                    }
                }
            } while (!unique);
        }

        // Extract sample points
        for (int i = 0; i < 4; i++) {
            sample_src[2*i] = src_points[2*indices[i]];
            sample_src[2*i+1] = src_points[2*indices[i]+1];
            sample_dst[2*i] = dst_points[2*indices[i]];
            sample_dst[2*i+1] = dst_points[2*indices[i]+1];
        }

        // Compute homography from 4 points
        if (compute_homography_4pt(sample_src, sample_dst, H) != 0) {
            continue; // Degenerate configuration
        }

        // Count inliers
        size_t inlier_count = 0;
        for (size_t i = 0; i < num_points; i++) {
            float x = src_points[2*i];
            float y = src_points[2*i+1];
            float xp_expected = dst_points[2*i];
            float yp_expected = dst_points[2*i+1];
            
            float xp, yp;
            apply_homography(H, x, y, &xp, &yp);
            
            float dx = xp - xp_expected;
            float dy = yp - yp_expected;
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist <= distance_threshold) {
                inlier_count++;
            }
        }

        // Update best if current is better
        if (inlier_count > best_inlier_count && inlier_count >= min_inlier_count) {
            best_inlier_count = inlier_count;
            for (int i = 0; i < 9; i++) {
                best_H[i] = H[i];
            }
        }
    }

    if (best_inlier_count < min_inlier_count) {
        return -3; // No valid homography found
    }

    // Copy best homography to output
    for (int i = 0; i < 9; i++) {
        homography[i] = best_H[i];
    }

    // Fill inlier mask if provided
    if (inlier_mask) {
        for (size_t i = 0; i < num_points; i++) {
            float x = src_points[2*i];
            float y = src_points[2*i+1];
            float xp_expected = dst_points[2*i];
            float yp_expected = dst_points[2*i+1];
            
            float xp, yp;
            apply_homography(best_H, x, y, &xp, &yp);
            
            float dx = xp - xp_expected;
            float dy = yp - yp_expected;
            float dist = sqrtf(dx*dx + dy*dy);
            
            inlier_mask[i] = (dist <= distance_threshold);
        }
    }

    return (int32_t)best_inlier_count;
}
