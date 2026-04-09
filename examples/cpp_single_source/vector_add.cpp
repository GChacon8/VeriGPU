#include "gpu_runtime.h"
#include <iostream>
#include <cassert>

// __thread_id() reads register t0 (x5) where compute_unit
// loaded the thread ID during clr
__device__ uint32_t __thread_id() {
    uint32_t tid;
    asm volatile("addi %0, x5, 0" : "=r"(tid));
    return tid;
}

__global__ void vector_add(uint32_t *a, uint32_t *b, uint32_t *out, uint32_t n) {
    uint32_t tid = __thread_id();
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char **argv, char **env) {
    gpuCreateContext();

    uint32_t n = 8;
    uint32_t a[] = {10, 20, 30, 40, 50, 60, 70, 80};
    uint32_t b[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t out[8];

    void *gpu_a   = gpuMalloc(n * sizeof(uint32_t));
    void *gpu_b   = gpuMalloc(n * sizeof(uint32_t));
    void *gpu_out = gpuMalloc(n * sizeof(uint32_t));

    gpuCopyToDevice(gpu_a, a, n * sizeof(uint32_t));
    gpuCopyToDevice(gpu_b, b, n * sizeof(uint32_t));

    // 8 threads, 1 thread per block, 8 blocks
    vector_add<<<dim3(8,1,1), dim3(1,1,1)>>>(
        (uint32_t*)gpu_a, (uint32_t*)gpu_b, (uint32_t*)gpu_out, n);

    gpuCopyFromDevice(out, gpu_out, n * sizeof(uint32_t));

    for (uint32_t i = 0; i < n; i++) {
        std::cout << "out[" << i << "] = " << out[i] << std::endl;
        assert(out[i] == a[i] + b[i]);
    }

    gpuDestroyContext();
    return 0;
}
