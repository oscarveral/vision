#include <stdint.h>
#include <stdlib.h>

#include "filters.h"

int32_t box_filter(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t filter_size) {
    // Validate filter_size.
    if (filter_size % 2 == 0 || filter_size < 1) {
        return -1;
    }
    // Limit image size to prevent overflows and excessive memory use.
    if (width*height > 4000000UL) {
        return -2;
    }

    // Precomute cumulative sum for efficiency.
    size_t half_size = filter_size / 2;
    uint32_t* cum_sum = (uint32_t*) calloc((width + 1) * (height + 1), sizeof(uint32_t));
    if (!cum_sum) {
        return -3;
    }

    for (size_t y = 1; y <= height; y++) {
        for (size_t x = 1; x <= width; x++) {
            cum_sum[y * (width + 1) + x] = input[(y - 1) * width + (x - 1)]
                                            + cum_sum[(y - 1) * (width + 1) + x]
                                            + cum_sum[y * (width + 1) + (x - 1)]
                                            - cum_sum[(y - 1) * (width + 1) + (x - 1)];
        }
    }

    // Apply box filter using the cumulative sum and mirror padding.
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            size_t x1 = (x < half_size) ? 0 : x - half_size;
            size_t y1 = (y < half_size) ? 0 : y - half_size;
            size_t x2 = (x + half_size >= width) ? width - 1 : x + half_size;
            size_t y2 = (y + half_size >= height) ? height - 1 : y + half_size;

            uint32_t area = (x2 - x1 + 1) * (y2 - y1 + 1);
            uint32_t sum = cum_sum[(y2 + 1) * (width + 1) + (x2 + 1)]
                           - cum_sum[(y1) * (width + 1) + (x2 + 1)]
                           - cum_sum[(y2 + 1) * (width + 1) + (x1)]
                           + cum_sum[(y1) * (width + 1) + (x1)];

            output[y * width + x] = (uint8_t)(sum / area);
        }
    }

    free(cum_sum);
    return 0;
}
