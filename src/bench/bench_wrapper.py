#!/usr/bin/env python3

import os
import time
import cv2
import numpy as np
import statistics

TEST_DENIS_PATH = os.path.join(os.path.dirname(__file__), "images/denis.jpg")
img_denis = cv2.imread(TEST_DENIS_PATH, cv2.IMREAD_GRAYSCALE)
TEST_LENNA_PATH = os.path.join(os.path.dirname(__file__), "images/lenna.png")
img_lenna = cv2.imread(TEST_LENNA_PATH, cv2.IMREAD_GRAYSCALE)

from dgst.ffi.wrapper import box_filter
from dgst.ffi.wrapper import gaussian_filter

# --------------------------------------------------------------
# Benchmark box_filter.
# --------------------------------------------------------------

def box_filter_benchmark_ffi(image, filter_size, iterations=10):
    times = []
    for _ in range(iterations):
        start_time = time.time()
        output = box_filter(image, filter_size)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(
        f"Box Filter FFI (size={filter_size}) - Mean: {mean_time:.3f} ms, Std Dev: {std_dev:.3f} ms ({iterations} iterations)"
    )


def box_filter_benchmark_cv(image, filter_size, iterations=10):
    times = []
    for _ in range(iterations):
        start_time = time.time()
        kernel = np.ones((filter_size, filter_size), np.float32) / (
            filter_size * filter_size
        )
        filtered_cv = cv2.filter2D(image, -1, kernel)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(
        f"Box Filter CV (size={filter_size}) - Mean: {mean_time:.3f} ms, Std Dev: {std_dev:.3f} ms ({iterations} iterations)"
    )


# --------------------------------------------------------------
# Benchmark gaussian_filter.
# --------------------------------------------------------------


def gaussian_filter_benchmark_ffi(image, sigma, iterations=10):
    times = []
    for _ in range(iterations):
        start_time = time.time()
        output = gaussian_filter(image, sigma)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(
        f"Gaussian Filter FFI (sigma={sigma}) - Mean: {mean_time:.3f} ms, Std Dev: {std_dev:.3f} ms ({iterations} iterations)"
    )


def gaussian_filter_benchmark_cv(image, sigma, iterations=10):
    times = []
    for _ in range(iterations):
        start_time = time.time()
        filtered_cv = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(
        f"Gaussian Filter CV (sigma={sigma}) - Mean: {mean_time:.3f} ms, Std Dev: {std_dev:.3f} ms ({iterations} iterations)"
    )


if __name__ == "__main__":
    box_filter_sizes = [3, 5, 7, 9, 11, 15]
    gaussian_filter_sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 11.0, 15.0]

    iterations = 1000

    imgs = [img_lenna, img_denis]
    for img in imgs:
        if img is None:
            print("Error: Test image not found or could not be loaded.")
            continue
        print(f"\nBenchmarking on image with shape: {img.shape}\n")
        print("Benchmarking Box Filter:")
        for size in box_filter_sizes:
            box_filter_benchmark_ffi(img, size, iterations)
            box_filter_benchmark_cv(img, size, iterations)

        print("\nBenchmarking Gaussian Filter:")
        for sigma in gaussian_filter_sigmas:
            gaussian_filter_benchmark_ffi(img, sigma, iterations)
            gaussian_filter_benchmark_cv(img, sigma, iterations)
