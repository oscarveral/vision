#!/usr/bin/env python3

import os
import time
import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt

TEST_DENIS_PATH = os.path.join(os.path.dirname(__file__), "images/denis.jpg")
img_denis = cv2.imread(TEST_DENIS_PATH, cv2.IMREAD_GRAYSCALE)
TEST_LENNA_PATH = os.path.join(os.path.dirname(__file__), "images/lenna.png")
img_lenna = cv2.imread(TEST_LENNA_PATH, cv2.IMREAD_GRAYSCALE)

from dgst.ffi.wrapper import box_filter
from dgst.ffi.wrapper import gaussian_filter

# Storage for benchmark results
benchmark_results = {
    "box": {"ffi": {}, "cv": {}},
    "gaussian": {"ffi": {}, "cv": {}},
}

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
    return mean_time, std_dev


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
    return mean_time, std_dev


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
    return mean_time, std_dev


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
    return mean_time, std_dev


if __name__ == "__main__":
    box_filter_sizes = [3, 5, 7, 9, 11, 15]
    gaussian_filter_sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 11.0, 15.0]

    iterations = 1000

    imgs = [img_lenna, img_denis]
    img_names = ["Lenna (512x512)", "Denis (1920x1200)"]

    for img, img_name in zip(imgs, img_names):
        if img is None:
            print("Error: Test image not found or could not be loaded.")
            continue

        img_key = img_name
        benchmark_results["box"]["ffi"][img_key] = []
        benchmark_results["box"]["cv"][img_key] = []
        benchmark_results["gaussian"]["ffi"][img_key] = []
        benchmark_results["gaussian"]["cv"][img_key] = []

        print(f"\nBenchmarking on image: {img_name} - shape: {img.shape}\n")
        print("Benchmarking Box Filter:")
        for size in box_filter_sizes:
            mean_ffi, std_ffi = box_filter_benchmark_ffi(img, size, iterations)
            mean_cv, std_cv = box_filter_benchmark_cv(img, size, iterations)
            benchmark_results["box"]["ffi"][img_key].append(
                (size, mean_ffi, std_ffi)
            )
            benchmark_results["box"]["cv"][img_key].append(
                (size, mean_cv, std_cv)
            )

        print("\nBenchmarking Gaussian Filter:")
        for sigma in gaussian_filter_sigmas:
            mean_ffi, std_ffi = gaussian_filter_benchmark_ffi(
                img, sigma, iterations
            )
            mean_cv, std_cv = gaussian_filter_benchmark_cv(
                img, sigma, iterations
            )
            benchmark_results["gaussian"]["ffi"][img_key].append(
                (sigma, mean_ffi, std_ffi)
            )
            benchmark_results["gaussian"]["cv"][img_key].append(
                (sigma, mean_cv, std_cv)
            )

    # Visualización de resultados de benchmark
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Box Filter - Lenna
    ax = axes[0, 0]
    lenna_key = img_names[0]
    sizes = [x[0] for x in benchmark_results["box"]["ffi"][lenna_key]]
    times_ffi = [x[1] for x in benchmark_results["box"]["ffi"][lenna_key]]
    times_cv = [x[1] for x in benchmark_results["box"]["cv"][lenna_key]]
    std_ffi = [x[2] for x in benchmark_results["box"]["ffi"][lenna_key]]
    std_cv = [x[2] for x in benchmark_results["box"]["cv"][lenna_key]]

    ax.errorbar(
        sizes,
        times_ffi,
        yerr=std_ffi,
        marker="o",
        label="FFI (C)",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.errorbar(
        sizes,
        times_cv,
        yerr=std_cv,
        marker="s",
        label="OpenCV",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Filter Size", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title(f"Box Filter - {lenna_key}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Box Filter - Denis
    ax = axes[0, 1]
    denis_key = img_names[1]
    sizes = [x[0] for x in benchmark_results["box"]["ffi"][denis_key]]
    times_ffi = [x[1] for x in benchmark_results["box"]["ffi"][denis_key]]
    times_cv = [x[1] for x in benchmark_results["box"]["cv"][denis_key]]
    std_ffi = [x[2] for x in benchmark_results["box"]["ffi"][denis_key]]
    std_cv = [x[2] for x in benchmark_results["box"]["cv"][denis_key]]

    ax.errorbar(
        sizes,
        times_ffi,
        yerr=std_ffi,
        marker="o",
        label="FFI (C)",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.errorbar(
        sizes,
        times_cv,
        yerr=std_cv,
        marker="s",
        label="OpenCV",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Filter Size", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title(f"Box Filter - {denis_key}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Gaussian Filter - Lenna
    ax = axes[1, 0]
    sigmas = [x[0] for x in benchmark_results["gaussian"]["ffi"][lenna_key]]
    times_ffi = [x[1] for x in benchmark_results["gaussian"]["ffi"][lenna_key]]
    times_cv = [x[1] for x in benchmark_results["gaussian"]["cv"][lenna_key]]
    std_ffi = [x[2] for x in benchmark_results["gaussian"]["ffi"][lenna_key]]
    std_cv = [x[2] for x in benchmark_results["gaussian"]["cv"][lenna_key]]

    ax.errorbar(
        sigmas,
        times_ffi,
        yerr=std_ffi,
        marker="o",
        label="FFI (C)",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.errorbar(
        sigmas,
        times_cv,
        yerr=std_cv,
        marker="s",
        label="OpenCV",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Sigma", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title(
        f"Gaussian Filter - {lenna_key}", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Gaussian Filter - Denis
    ax = axes[1, 1]
    sigmas = [x[0] for x in benchmark_results["gaussian"]["ffi"][denis_key]]
    times_ffi = [x[1] for x in benchmark_results["gaussian"]["ffi"][denis_key]]
    times_cv = [x[1] for x in benchmark_results["gaussian"]["cv"][denis_key]]
    std_ffi = [x[2] for x in benchmark_results["gaussian"]["ffi"][denis_key]]
    std_cv = [x[2] for x in benchmark_results["gaussian"]["cv"][denis_key]]

    ax.errorbar(
        sigmas,
        times_ffi,
        yerr=std_ffi,
        marker="o",
        label="FFI (C)",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.errorbar(
        sigmas,
        times_cv,
        yerr=std_cv,
        marker="s",
        label="OpenCV",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Sigma", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title(
        f"Gaussian Filter - {denis_key}", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calcular y mostrar speedups
    print("\n" + "=" * 80)
    print("ANÁLISIS DE SPEEDUP (FFI vs OpenCV)")
    print("=" * 80)

    for img_name in img_names:
        print(f"\n{img_name}:")
        print("-" * 60)

        print("\nBox Filter:")
        for i, size in enumerate(box_filter_sizes):
            ffi_time = benchmark_results["box"]["ffi"][img_name][i][1]
            cv_time = benchmark_results["box"]["cv"][img_name][i][1]
            speedup = cv_time / ffi_time
            status = "FFI wins" if speedup > 1 else "OpenCV wins"
            print(f"  Size {size:2d}: Speedup = {speedup:.2f}x ({status})")

        print("\nGaussian Filter:")
        for i, sigma in enumerate(gaussian_filter_sigmas):
            ffi_time = benchmark_results["gaussian"]["ffi"][img_name][i][1]
            cv_time = benchmark_results["gaussian"]["cv"][img_name][i][1]
            speedup = cv_time / ffi_time
            status = "FFI wins" if speedup > 1 else "OpenCV wins"
            print(f"  Sigma {sigma:4.1f}: Speedup = {speedup:.2f}x ({status})")
