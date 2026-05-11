/*
 * vector_add_f32.cpp — Float vector addition, multi-threaded.
 *
 * CP-3 (HW Roadmap): Validates the full chain:
 *   C++ source → clang (CUDA split) → llc-zfinx (RISC-V) →
 *   patch_hostside → g++ → libverigpu_runtime.so → Verilator →
 *   4 RISC-V cores executing FADD.S in parallel
 *
 * This is the float equivalent of vector_add.cpp (integers).
 * Each thread computes one element: out[tid] = a[tid] + b[tid]
 * With 8 elements and 4 cores, it runs in 2 batches.
 *
 * Run:
 *   examples/cpp_single_source/run.sh vector_add_f32
 */

#include "gpu_runtime.h"
#include <iostream>
#include <cassert>
#include <cmath>

__device__ uint32_t __thread_id() {
    uint32_t tid;
    asm volatile("addi %0, x5, 0" : "=r"(tid));
    return tid;
}

__global__ void vector_add_f32(float *a, float *b, float *out, uint32_t n) {
    uint32_t tid = __thread_id();
    if (tid < n) {
        out[tid] = a[tid] + b[tid];  // compiles to FADD.S on RISC-V zfinx
    }
}

int main(int argc, char **argv, char **env) {
    std::cout << "=== CP-3 (HW Roadmap): Float Vector Add via Verilator Runtime ===" << std::endl;

    gpuCreateContext();

    uint32_t n = 8;
    float a[]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[]   = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    float out[8] = {0};

    // Allocate GPU memory
    void *gpu_a   = gpuMalloc(n * sizeof(float));
    void *gpu_b   = gpuMalloc(n * sizeof(float));
    void *gpu_out = gpuMalloc(n * sizeof(float));

    // Copy input data to GPU
    gpuCopyToDevice(gpu_a, a, n * sizeof(float));
    gpuCopyToDevice(gpu_b, b, n * sizeof(float));

    // Launch kernel: 8 threads (2 batches of 4 cores)
    vector_add_f32<<<dim3(8, 1, 1), dim3(1, 1, 1)>>>(
        (float *)gpu_a, (float *)gpu_b, (float *)gpu_out, n);

    // Copy results back
    gpuCopyFromDevice(out, gpu_out, n * sizeof(float));

    // Verify results
    int pass = 1;
    for (uint32_t i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        std::cout << "  out[" << i << "] = " << out[i]
                  << "  (expected " << expected << ")";
        if (fabsf(out[i] - expected) < 0.001f) {
            std::cout << "  OK" << std::endl;
        } else {
            std::cout << "  MISMATCH!" << std::endl;
            pass = 0;
        }
    }

    std::cout << std::endl;
    if (pass) {
        std::cout << "========================================" << std::endl;
        std::cout << "  ALL 8 FLOAT ADDITIONS CORRECT" << std::endl;
        std::cout << "  FADD.S via Verilator runtime works!" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "FAIL: some results are wrong" << std::endl;
    }

    gpuDestroyContext();
    return pass ? 0 : 1;
}
