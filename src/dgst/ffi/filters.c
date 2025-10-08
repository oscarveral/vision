#include "filters.h"

uint32_t sum_array(const uint32_t* array, size_t length) {
    uint32_t sum = 0;
    for (size_t i = 0; i < length; i++) {
        sum += array[i];
    }
    return sum;
}
