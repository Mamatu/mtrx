#include "mtrxCublas/status_handler.hpp"
#include <mtrxCublas/impl/host_alloc.hpp>

namespace mtrx {

void HostAlloc::malloc(void **devPtr, size_t size) {
  void *ptr = ::malloc(size);
  *devPtr = ptr;
}

void HostAlloc::free(void *devPtr) { ::free(devPtr); }

void HostAlloc::memcpyKernelToHost(void *dst, const void *src, size_t count) {
  memcpy(dst, src, count);
}

} // namespace mtrx
