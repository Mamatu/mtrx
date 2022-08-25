#ifndef MTRX_CUBLAS_CUDA_ALLOC_HPP
#define MTRX_CUBLAS_CUDA_ALLOC_HPP

#include <cuda.h>
#include <mtrxCublas/impl/alloc.hpp>

namespace mtrx {

/**
 * @brief Abstract interface for alloc/dealloc/memcpy primitives.
 * It is used in code which can be compiled into device/host form
 */
class CudaAlloc : public Alloc {
public:
  void malloc(void **devPtr, size_t size) override;

  void free(void *devPtr) override;

  void memcpyKernelToHost(void *dst, const void *src, size_t count) override;
};
} // namespace mtrx

#endif
