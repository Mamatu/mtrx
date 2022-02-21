#ifndef MTRX_CUBLAS_ALLOC_HPP
#define MTRX_CUBLAS_ALLOC_HPP

#include <stddef.h>

namespace mtrx {

/**
 * @brief Abstract interface for alloc/dealloc/memcpy primitives.
 * It is used in code which can be compiled into device/host form
 */
class Alloc {
public:
  virtual void malloc(void **devPtr, size_t size) = 0;
  virtual void free(void *devPtr) = 0;
  virtual void memcpyKernelToHost(void *, const void *, size_t) = 0;
};
} // namespace mtrx

#endif
