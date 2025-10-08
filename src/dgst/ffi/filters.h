#ifndef DGST_FILTERS_H
#define DGST_FILTERS_H

#include "stddef.h"
#include "stdint.h"

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
uint32_t sum_array(const uint32_t* array, size_t length);

#endif // DGST_FILTERS_H
