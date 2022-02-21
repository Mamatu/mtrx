#include "cuda_alloc.hpp"
#include "mtrxCublas/status_handler.hpp"
#include <cuda.h>

namespace mtrx {

void CudaAlloc::malloc(void **devPtr, size_t size) {
  handleStatus(cudaMalloc(devPtr, size));
}

void CudaAlloc::free(void *devPtr) { handleStatus(cudaFree(devPtr)); }

void CudaAlloc::memcpyKernelToHost(void *dst, const void *src, size_t count) {
  handleStatus(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

} // namespace mtrx
