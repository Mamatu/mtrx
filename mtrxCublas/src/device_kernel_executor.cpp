/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of mtrx.
 *
 * mtrx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * mtrx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with mtrx.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "device_kernel_executor.hpp"
#include "device_properties_loader.hpp"
#include "sys_pathes_parser.hpp"
#include <filesystem>
#include <mtrxCublas/status_handler.hpp>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdio.h>
#include <vector>

namespace mtrx {
void DeviceKernelExecutor::Init() {
  static bool wasInit = false;
  if (wasInit == false) {
    wasInit = true;
    cuInit(0);
  }
}

DeviceKernelExecutor::DeviceKernelExecutor()
    : IKernelExecutor(loadDeviceProperties(0)), m_image(nullptr),
      m_cuModule(nullptr) {
  Init();
  load(CUBIN_FILE_NAME);
}

void DeviceKernelExecutor::loadCuModule() {
  if (nullptr != m_image && nullptr == m_cuModule) {
    spdlog::info("Load module from image {}", m_image);
    handleStatus(cuModuleLoadData(&m_cuModule, m_image));
  } else if (!m_path.empty() && nullptr == m_cuModule) {
    spdlog::info("Load module from path {}", m_path);
    handleStatus(cuModuleLoad(&m_cuModule, m_path.c_str()));
  }
}

void DeviceKernelExecutor::unloadCuModule() {
  if (m_cuModule != nullptr) {
    handleStatus(cuModuleUnload(m_cuModule));
    m_cuModule = nullptr;
  }
}

void DeviceKernelExecutor::unload() { unloadCuModule(); }

void DeviceKernelExecutor::setImage(void *image) {
  this->m_image = image;
  loadCuModule();
}

bool DeviceKernelExecutor::setImageFromPathes(const Pathes &pathes) {
  const auto &path = pathes[0];
  spdlog::info("Load module from path {}", path);
  m_path = path;
  return true;
}

bool DeviceKernelExecutor::_run(const std::string &functionName) {
  CUresult status = CUDA_SUCCESS;

  loadCuModule();

  CUfunction cuFunction = nullptr;
  if (nullptr != m_cuModule) {
    handleStatus(
        cuModuleGetFunction(&cuFunction, m_cuModule, functionName.c_str()));

    const RunParams &ep = getRunParams();

    handleStatus(cuLaunchKernel(
        cuFunction, ep.blocksCount[0], ep.blocksCount[1], ep.blocksCount[2],
        ep.threadsCount[0], ep.threadsCount[1], ep.threadsCount[2],
        ep.sharedMemSize, nullptr, const_cast<void **>(this->getParams()),
        nullptr));
  } else {
    if (nullptr != m_image && m_path.empty()) {
      std::stringstream sstream;
      sstream << "Module is incorrect. Image is not loaded. " << m_cuModule
              << ", " << m_image;
      throw std::runtime_error(sstream.str());
    } else if (!m_path.empty()) {
      std::stringstream sstream;
      sstream << "Module is incorrect. Image is not loaded. " << m_cuModule
              << ", " << m_path.c_str();
      throw std::runtime_error(sstream.str());
    }
  }

  return status == 0;
}

DeviceKernelExecutor::~DeviceKernelExecutor() {
  unloadCuModule();
  releaseImage();
}

char *DeviceKernelExecutor::Load(FileUnique &&f) {
  if (f) {
    fseek(f.get(), 0, SEEK_END);
    size_t size = ftell(f.get());
    fseek(f.get(), 0, SEEK_SET);
    char *data = static_cast<char *>(malloc(size * sizeof(char)));
    memset(data, 0, size * sizeof(char));
    spdlog::info("Reading file with size {}", size);
    constexpr size_t count = 1;
    size_t returnedCount = fread(data, size * sizeof(char), count, f.get());
    if (returnedCount != count) {
      spdlog::error("Returned elements count {0} is not equal to expected {1} "
                    "Error: {2} Eof: {3}",
                    returnedCount, count, ferror(f.get()), feof(f.get()));
      return nullptr;
    }
    return data;
  }
  return nullptr;
}

char *DeviceKernelExecutor::Load(const fs::path &path) {
  std::unique_ptr<FILE, std::function<void(FILE *)>> f(
      fopen(path.c_str(), "rb"), [](FILE *f) { fclose(f); });
  return Load(std::move(f));
}

char *DeviceKernelExecutor::Load(const fs::path &path,
                                 const mtrx::Pathes &sysPathes) {
  spdlog::info("Sys pathes to find cubin {}", sysPathes);
  for (size_t idx = 0; idx < sysPathes.size(); ++idx) {
    auto sysPath = sysPathes[idx];
    fs::path p = fs::path(sysPathes[idx]) / path;
    spdlog::info("Trying to load {}", p.u8string());
    if (fs::exists(p)) {
      auto *data = Load(p);
      if (data != nullptr) {
        spdlog::info("Loaded {}", p.u8string());
        return data;
      } else {
        spdlog::warn("None data loaded from {}", p.u8string());
      }
    } else {
      spdlog::warn("Path {} doesn't exist", p.u8string());
    }
  }
  return nullptr;
}

Pathes DeviceKernelExecutor::FindExist(const fs::path &path,
                                       const Pathes &sysPathes) {
  Pathes pathes;
  for (size_t idx = 0; idx < sysPathes.size(); ++idx) {
    auto sysPath = sysPathes[idx];
    fs::path p = fs::path(sysPathes[idx]) / path;
    if (fs::exists(p)) {
      pathes.push_back(p);
    }
  }
  return pathes;
}

Pathes DeviceKernelExecutor::GetSysPathes() {
  return mtrx::loadSysPathes(ENVIRONMENT_VARIABLE);
}

bool DeviceKernelExecutor::load(const fs::path &path) {
  const auto &pathes = FindExist(path, GetSysPathes());
  return setImageFromPathes(pathes);
}

bool DeviceKernelExecutor::load(const mtrx::Pathes &pathes) {
  for (const auto &path : pathes) {
    if (load(path)) {
      return true;
    }
  }
  return false;
}

void DeviceKernelExecutor::releaseImage() {
  if (m_image != nullptr) {
    free(m_image);
    m_image = nullptr;
  }
}
} // namespace mtrx
