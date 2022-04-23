#ifndef MTRX_TEST_HPP
#define MTRX_TEST_HPP

#include <gtest/gtest.h>

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#ifdef CUBLAS_NVPROF_TESTS
#include "cuda_profiler.hpp"
#endif

namespace mtrx
{

class Test : public testing::Test {
public:
  virtual void setUp()
  {}

  virtual void tearDown()
  {}

  void SetUp() override {
#ifdef CUBLAS_NVPROF_TESTS
    startProfiler();
#endif

#define LOG_LEVEL_CRITICAL 0
#define LOG_LEVEL_ERROR 1
#define LOG_LEVEL_INFO 2
#define LOG_LEVEL_DEBUG 3
#define LOG_LEVEL_TRACE 4

    spdlog::level::level_enum testLogLevel = spdlog::level::critical;
#if (MTRX_CUBLAS_TEST_LOG_LEVEL == LOG_LEVEL_CRITICAL)
    testLogLevel = spdlog::level::critical;
#elif (MTRX_CUBLAS_TEST_LOG_LEVEL == LOG_LEVEL_ERROR)
    testLogLevel = spdlog::level::err;
#elif (MTRX_CUBLAS_TEST_LOG_LEVEL == LOG_LEVEL_INFO)
    testLogLevel = spdlog::level::info;
#elif (MTRX_CUBLAS_TEST_LOG_LEVEL == LOG_LEVEL_DEBUG)
    testLogLevel = spdlog::level::debug;
#elif (MTRX_CUBLAS_TEST_LOG_LEVEL == LOG_LEVEL_TRACE)
    testLogLevel = spdlog::level::trace;
#endif
    spdlog::set_level(testLogLevel);
    setUp();
  }

  void TearDown() override {
#ifdef CUBLAS_NVPROF_TESTS
    stopProfiler();
#endif
    tearDown();
  }
};

}
#endif
