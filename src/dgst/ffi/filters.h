#ifndef DGST_FILTERS_H
#define DGST_FILTERS_H

#include "stdint.h"
#include "stddef.h"

uint32_t sum_array(const uint32_t* array, size_t length);
int box_filter(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t filter_size);

#endif // DGST_FILTERS_H