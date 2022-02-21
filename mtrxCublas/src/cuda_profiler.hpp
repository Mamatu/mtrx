#ifndef MTRX_CUBLAS_CUDA_PROFILER_HPP
#define MTRX_CUBLAS_CUDA_PROFILER_HPP

#include <cuda_profiler_api.h>
#include <mtrxCublas/status_handler.hpp>
#include <spdlog/spdlog.h>

namespace mtrx {

inline void startProfiler() {
  spdlog::info("Start profiling section...");
  handleStatus(cudaProfilerStart());
}

inline void stopProfiler() {
  handleStatus(cudaProfilerStop());
  spdlog::info("Stop profiling section.");
}

class Profiler final {
public:
  Profiler() { startProfiler(); }
  ~Profiler() { stopProfiler(); }
};

} // namespace mtrx

#endif
