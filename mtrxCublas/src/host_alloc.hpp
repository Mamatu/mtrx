#ifndef MTRX_CUBLAS_HOST_ALLOC_HPP
#define MTRX_CUBLAS_HOST_ALLOC_HPP

#include "alloc.hpp"
#include <cuda.h>

namespace mtrx {

/**
 * @brief Abstract interface for alloc/dealloc/memcpy primitives.
 * It is used in code which can be compiled into device/host form
 */
class HostAlloc : public Alloc {
public:
  void malloc(void **devPtr, size_t size) override;

  void free(void *devPtr) override;

  void memcpyKernelToHost(void *dst, const void *src, size_t count) override;
};
} // namespace mtrx

#endif
