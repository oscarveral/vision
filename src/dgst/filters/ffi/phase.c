#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "filters.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Fast approximations for performance-critical paths
static inline float fast_exp(float x) {
	// Clamp for stability
	if (x < -10.0f) return 0.0f;
	if (x > 10.0f) return expf(10.0f);
	return expf(x);
}

static inline float fast_sqrt(float x) {
	// Use hardware sqrt which is fast on modern CPUs
	return sqrtf(x);
}

// Complex number structure.
typedef struct {
	float real;
	float imag;
} Complex;

// Helper functions for complex arithmetic.
static inline Complex complex_add(Complex a, Complex b) {
	Complex result = { a.real + b.real, a.imag + b.imag };
	return result;
}

static inline Complex complex_sub(Complex a, Complex b) {
	Complex result = { a.real - b.real, a.imag - b.imag };
	return result;
}

static inline Complex complex_mul(Complex a, Complex b) {
	Complex result = { a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real };
	return result;
}

static inline Complex complex_exp(float angle) {
	Complex result = { cosf(angle), sinf(angle) };
	return result;
}

// Optimized DFT with precomputed twiddle factors
static void dft_1d(Complex* data, size_t n, int32_t inverse) {
	Complex* temp = (Complex*)malloc(n * sizeof(Complex));
	Complex* twiddle = (Complex*)malloc(n * sizeof(Complex));
	if (!temp || !twiddle) {
		free(temp);
		free(twiddle);
		return;
	}
	
	memcpy(temp, data, n * sizeof(Complex));
	
	float sign = inverse ? 1.0f : -1.0f;
	float scale = inverse ? (1.0f / (float)n) : 1.0f;
	float base_angle = sign * 2.0f * (float)M_PI / (float)n;
	
	// Precompute twiddle factors for first row
	for (size_t j = 0; j < n; j++) {
		float angle = base_angle * (float)j;
		twiddle[j].real = cosf(angle);
		twiddle[j].imag = sinf(angle);
	}
	
	// Compute DFT using precomputed twiddles
	for (size_t k = 0; k < n; k++) {
		float sum_real = 0.0f;
		float sum_imag = 0.0f;
		
		// Unroll small cases
		if (n <= 4) {
			for (size_t j = 0; j < n; j++) {
				size_t idx = (k * j) % n;
				float wr = twiddle[idx].real;
				float wi = twiddle[idx].imag;
				sum_real += temp[j].real * wr - temp[j].imag * wi;
				sum_imag += temp[j].real * wi + temp[j].imag * wr;
			}
		} else {
			// Vectorizable loop
			for (size_t j = 0; j < n; j++) {
				size_t idx = (k * j) % n;
				float wr = twiddle[idx].real;
				float wi = twiddle[idx].imag;
				float tr = temp[j].real;
				float ti = temp[j].imag;
				sum_real += tr * wr - ti * wi;
				sum_imag += tr * wi + ti * wr;
			}
		}
		
		data[k].real = sum_real * scale;
		data[k].imag = sum_imag * scale;
	}
	
	free(temp);
	free(twiddle);
}

// Check if n is a power of 2.
static int is_power_of_2(size_t n) {
	return n > 0 && (n & (n - 1)) == 0;
}

// Bit reversal for FFT.
static void bit_reverse(Complex* data, size_t n) {
	size_t j = 0;
	for (size_t i = 0; i < n - 1; i++) {
		if (i < j) {
			Complex temp = data[i];
			data[i]		 = data[j];
			data[j]		 = temp;
		}
		size_t k = n >> 1;
		while (k <= j) {
			j -= k;
			k >>= 1;
		}
		j += k;
	}
}

// 1D FFT (in-place, Cooley-Tukey algorithm for power-of-2).
static void fft_1d_pow2(Complex* data, size_t n, int32_t inverse) {
	bit_reverse(data, n);

	float sign = inverse ? 1.0f : -1.0f;

	for (size_t s = 2; s <= n; s <<= 1) {
		float angle		 = sign * 2.0f * M_PI / s;
		Complex w		 = complex_exp(angle);
		size_t half_size = s >> 1;

		for (size_t k = 0; k < n; k += s) {
			Complex wn = { 1.0f, 0.0f };
			for (size_t j = 0; j < half_size; j++) {
				Complex t				= complex_mul(wn, data[k + j + half_size]);
				Complex u				= data[k + j];
				data[k + j]				= complex_add(u, t);
				data[k + j + half_size] = complex_sub(u, t);
				wn						= complex_mul(wn, w);
			}
		}
	}

	if (inverse) {
		float scale = 1.0f / n;
		for (size_t i = 0; i < n; i++) {
			data[i].real *= scale;
			data[i].imag *= scale;
		}
	}
}

// 1D FFT dispatcher (uses fast algorithm for power-of-2, DFT otherwise).
static void fft_1d(Complex* data, size_t n, int32_t inverse) {
	if (is_power_of_2(n)) {
		fft_1d_pow2(data, n, inverse);
	} else {
		dft_1d(data, n, inverse);
	}
}

// 2D FFT (row-by-row, then column-by-column) - optimized with blocking
static int32_t fft_2d(Complex* data, size_t rows, size_t cols, int32_t inverse) {
	// FFT along rows (good cache locality)
	#pragma omp parallel for if (rows * cols > 65536)
	for (size_t y = 0; y < rows; y++) {
		fft_1d(&data[y * cols], cols, inverse);
	}

	// Transpose for better cache locality in column FFT
	Complex* transposed = (Complex*)malloc(rows * cols * sizeof(Complex));
	if (!transposed) {
		return -3;
	}

	// Blocked transpose for cache efficiency
	const size_t BLOCK = 32;
	for (size_t y0 = 0; y0 < rows; y0 += BLOCK) {
		size_t y_end = (y0 + BLOCK < rows) ? y0 + BLOCK : rows;
		for (size_t x0 = 0; x0 < cols; x0 += BLOCK) {
			size_t x_end = (x0 + BLOCK < cols) ? x0 + BLOCK : cols;
			for (size_t y = y0; y < y_end; y++) {
				for (size_t x = x0; x < x_end; x++) {
					transposed[x * rows + y] = data[y * cols + x];
				}
			}
		}
	}

	// FFT along transposed rows (which are original columns)
	#pragma omp parallel for if (rows * cols > 65536)
	for (size_t x = 0; x < cols; x++) {
		fft_1d(&transposed[x * rows], rows, inverse);
	}

	// Transpose back with blocking
	for (size_t x0 = 0; x0 < cols; x0 += BLOCK) {
		size_t x_end = (x0 + BLOCK < cols) ? x0 + BLOCK : cols;
		for (size_t y0 = 0; y0 < rows; y0 += BLOCK) {
			size_t y_end = (y0 + BLOCK < rows) ? y0 + BLOCK : rows;
			for (size_t x = x0; x < x_end; x++) {
				for (size_t y = y0; y < y_end; y++) {
					data[y * cols + x] = transposed[x * rows + y];
				}
			}
		}
	}

	free(transposed);
	return 0;
}

// FFT shift (move zero frequency to center).
static void fft_shift(Complex* data, size_t rows, size_t cols) {
	size_t half_rows = rows >> 1;
	size_t half_cols = cols >> 1;

	// Swap quadrants.
	for (size_t y = 0; y < half_rows; y++) {
		for (size_t x = 0; x < half_cols; x++) {
			// Top-left <-> Bottom-right.
			Complex temp								   = data[y * cols + x];
			data[y * cols + x]							   = data[(y + half_rows) * cols + (x + half_cols)];
			data[(y + half_rows) * cols + (x + half_cols)] = temp;

			// Top-right <-> Bottom-left.
			temp							 = data[y * cols + (x + half_cols)];
			data[y * cols + (x + half_cols)] = data[(y + half_rows) * cols + x];
			data[(y + half_rows) * cols + x] = temp;
		}
	}
}

int32_t phase_congruency(const uint8_t* input, float* output, size_t width, size_t height, int32_t nscale, int32_t norient, float min_wavelength, float mult, float sigma_onf, float eps) {

	// Validate input parameters.
	if (!input || !output || width == 0 || height == 0) {
		return -1;
	}

	if (nscale <= 0 || norient <= 0 || min_wavelength <= 0.0f || mult <= 1.0f || sigma_onf <= 0.0f || eps <= 0.0f) {
		return -1;
	}

	// Limit image size.
	if (width * height > 16000000UL) {
		return -2;
	}

	size_t size = width * height;

	// Allocate memory for complex buffers.
	Complex* image_fft	  = (Complex*)malloc(size * sizeof(Complex));
	Complex* filtered	  = (Complex*)malloc(size * sizeof(Complex));
	float* pc_sum		  = (float*)calloc(size, sizeof(float));
	float* amplitude_sum  = (float*)calloc(size, sizeof(float));
	float* energy_orient  = (float*)malloc(size * sizeof(float));
	float* an			  = (float*)malloc(size * sizeof(float));
	float* sum_even		  = (float*)malloc(size * sizeof(float));
	float* sum_odd		  = (float*)malloc(size * sizeof(float));
	float* log_gabor	  = (float*)malloc(size * sizeof(float));
	float* angular_spread = (float*)malloc(size * sizeof(float));

	if (!image_fft || !filtered || !pc_sum || !amplitude_sum || !energy_orient || !an || !sum_even || !sum_odd || !log_gabor || !angular_spread) {
		free(image_fft);
		free(filtered);
		free(pc_sum);
		free(amplitude_sum);
		free(energy_orient);
		free(an);
		free(sum_even);
		free(sum_odd);
		free(log_gabor);
		free(angular_spread);
		return -3;
	}

	// Convert input to complex and compute FFT.
	for (size_t i = 0; i < size; i++) {
		image_fft[i].real = (float)input[i];
		image_fft[i].imag = 0.0f;
	}

	if (fft_2d(image_fft, height, width, 0) != 0) {
		free(image_fft);
		free(filtered);
		free(pc_sum);
		free(amplitude_sum);
		free(energy_orient);
		free(an);
		free(sum_even);
		free(sum_odd);
		free(log_gabor);
		free(angular_spread);
		return -3;
	}

	// Apply FFT shift.
	fft_shift(image_fft, height, width);

	// Precompute frequency grids and radius (avoid redundant calculations).
	float half_height = (float)height * 0.5f;
	float half_width  = (float)width * 0.5f;
	float inv_width = 1.0f / (float)width;
	float inv_height = 1.0f / (float)height;
	
	// Precompute theta and radius arrays
	float* theta_grid = (float*)malloc(size * sizeof(float));
	float* radius_grid = (float*)malloc(size * sizeof(float));
	if (!theta_grid || !radius_grid) {
		free(theta_grid);
		free(radius_grid);
		free(image_fft);
		free(filtered);
		free(pc_sum);
		free(amplitude_sum);
		free(energy_orient);
		free(an);
		free(sum_even);
		free(sum_odd);
		free(log_gabor);
		free(angular_spread);
		return -3;
	}
	
	#pragma omp parallel for if (size > 65536)
	for (size_t y = 0; y < height; y++) {
		float yf = (float)y - half_height;
		float yf_norm = yf * inv_height;
		for (size_t x = 0; x < width; x++) {
			float xf = (float)x - half_width;
			float xf_norm = xf * inv_width;
			size_t idx = y * width + x;
			
			theta_grid[idx] = atan2f(yf, xf);
			radius_grid[idx] = sqrtf(xf_norm * xf_norm + yf_norm * yf_norm);
		}
	}

	// Precompute angular parameters
	float angular_sigma = (float)M_PI / (float)norient * 1.2f;
	float angular_sigma2 = 2.0f * angular_sigma * angular_sigma;

	// Process each orientation.
	for (int32_t o = 0; o < norient; o++) {
		float angle = (float)o * (float)M_PI / (float)norient;
		float cos_angle = cosf(angle);
		float sin_angle = sinf(angle);

		// Compute angular spread for this orientation (optimized).
		#pragma omp parallel for if (size > 65536)
		for (size_t i = 0; i < size; i++) {
			float theta = theta_grid[i];
			float cos_theta = cosf(theta);
			float sin_theta = sinf(theta);
			
			// Compute angular distance using precomputed trig values.
			float ds = sin_theta * cos_angle - cos_theta * sin_angle;
			float dc = cos_theta * cos_angle + sin_theta * sin_angle;
			float dtheta = fabsf(atan2f(ds, dc));
			
			angular_spread[i] = fast_exp(-dtheta * dtheta / angular_sigma2);
		}

		// Initialize accumulators for this orientation.
		memset(sum_even, 0, size * sizeof(float));
		memset(sum_odd, 0, size * sizeof(float));
		memset(an, 0, size * sizeof(float));

		// Precompute log(sigma_onf) outside scale loop
		float log_sigma_onf = logf(sigma_onf);
		float log_sigma2 = 2.0f * log_sigma_onf * log_sigma_onf;

		// Process each scale.
		float wavelength = min_wavelength;
		for (int32_t s = 0; s < nscale; s++) {
			float fo = 1.0f / wavelength;

			// Build log-Gabor filter (optimized with precomputed values).
			#pragma omp parallel for if (size > 65536)
			for (size_t i = 0; i < size; i++) {
				float radius = radius_grid[i];
				
				// Handle DC component
				if (radius < 1e-8f) {
					log_gabor[i] = 0.0f;
					continue;
				}
				
				float log_rad = logf(radius / fo);
				float radial = fast_exp(-log_rad * log_rad / log_sigma2);
				
				log_gabor[i] = radial * angular_spread[i];
			}

			// Apply filter in frequency domain and compute inverse FFT in one pass
			memcpy(filtered, image_fft, size * sizeof(Complex));
			
			#pragma omp parallel for if (size > 65536)
			for (size_t i = 0; i < size; i++) {
				filtered[i].real *= log_gabor[i];
				filtered[i].imag *= log_gabor[i];
			}

			// Inverse FFT shift.
			fft_shift(filtered, height, width);

			// Inverse FFT.
			if (fft_2d(filtered, height, width, 1) != 0) {
				free(theta_grid);
				free(radius_grid);
				free(image_fft);
				free(filtered);
				free(pc_sum);
				free(amplitude_sum);
				free(energy_orient);
				free(an);
				free(sum_even);
				free(sum_odd);
				free(log_gabor);
				free(angular_spread);
				return -3;
			}

			// Accumulate even and odd responses (fused loop).
			#pragma omp parallel for if (size > 65536)
			for (size_t i = 0; i < size; i++) {
				float even = filtered[i].real;
				float odd = filtered[i].imag;
				
				sum_even[i] += even;
				sum_odd[i] += odd;
				
				// Compute amplitude directly
				float amp = fast_sqrt(even * even + odd * odd);
				an[i] += amp;
			}

			wavelength *= mult;
		}

		// Compute local energy and accumulate (fused loop).
		#pragma omp parallel for if (size > 65536)
		for (size_t i = 0; i < size; i++) {
			float energy = fast_sqrt(sum_even[i] * sum_even[i] + sum_odd[i] * sum_odd[i]);
			pc_sum[i] += energy;
			amplitude_sum[i] += an[i];
		}
	}

	// Free precomputed grids
	free(theta_grid);
	free(radius_grid);

	// Normalize and convert to uint8 (fused loop with SIMD-friendly pattern).
	float max_val = 0.0f;
	
	// First pass: normalize
	#pragma omp parallel for reduction(max:max_val) if (size > 65536)
	for (size_t i = 0; i < size; i++) {
		float denom = amplitude_sum[i] + eps;
		float val = pc_sum[i] / denom;
		// Clamp to [0, 1].
		val = (val < 0.0f) ? 0.0f : ((val > 1.0f) ? 1.0f : val);
		pc_sum[i] = val;
		if (val > max_val) {
			max_val = val;
		}
	}

	// Second pass: scale to [0,1] float mapping (preserve relative magnitudes)
	float inv_max = (max_val > eps) ? (1.0f / max_val) : 0.0f;
	#pragma omp parallel for if (size > 65536)
	for (size_t i = 0; i < size; i++) {
		// Output as float in [0,1]
		output[i] = pc_sum[i] * inv_max;
	}

	// Clean up.
	free(image_fft);
	free(filtered);
	free(pc_sum);
	free(amplitude_sum);
	free(energy_orient);
	free(an);
	free(sum_even);
	free(sum_odd);
	free(log_gabor);
	free(angular_spread);

	return 0;
}
