#include <stdint.h>
#include <stdlib.h>

#include "filters.h"

int32_t threshold_filter(const uint8_t* input, uint8_t* output, size_t width, size_t height, float threshold) {
    if (!input || !output || width == 0 || height == 0) {
        return -1;
    }

    if (!(threshold >= 0.0f && threshold <= 1.0f)) {
        return -1;
    }

    size_t npix = width * height;
    if (npix > 16000000UL) {
        return -2;
    }

    // Simple threshold loop. Cope with NaN by treating it as 0 (non-edge).
    // Input is uint8 in [0,255]. Convert to normalized float [0,1] before thresholding.
    // Output is uint8 binary image: 255 for >= threshold, 0 otherwise.
    for (size_t i = 0; i < npix; i++) {
        uint8_t vin = input[i];
        float v = (float)vin / 255.0f;
        // clamp tiny numeric errors to range
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        // set output to 255 if >= threshold, else 0
        output[i] = (v >= threshold) ? (uint8_t)255 : (uint8_t)0;
    }

    return 0;
}
