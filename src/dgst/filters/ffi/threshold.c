#include <stdint.h>
#include <stdlib.h>

#include "filters.h"

int32_t threshold_filter(const float* input, float* output, size_t width, size_t height, float threshold) {
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

    // Simple threshold loop. Cope with NaN by treating it as 0.0 (non-edge).
    for (size_t i = 0; i < npix; i++) {
        float v = input[i];
        // clamp tiny numeric errors to range
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        // set output to 1.0f if >= threshold, else 0.0f
        output[i] = (v >= threshold) ? 1.0f : 0.0f;
    }

    return 0;
}
