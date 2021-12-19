/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <filesystem>
#include <mtrxCublas/status_handler.hpp>

#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <sstream>
#include <stdio.h>
#include <vector>

#include "kernel_executor.hpp"

namespace mtrx {
void KernelExecutor::Init() {
  static bool wasInit = false;
  if (wasInit == false) {
    wasInit = true;
    cuInit(0);
  }
}

KernelExecutor::KernelExecutor() : m_image(nullptr), m_cuModule(nullptr) {}

void KernelExecutor::loadCuModule() {
  if (nullptr != m_image && nullptr == m_cuModule) {
    printf("Load module from image = %p", m_image);
    handleStatus(cuModuleLoadData(&m_cuModule, m_image));
  }
}

void KernelExecutor::unloadCuModule() {
  if (m_cuModule != nullptr) {
    handleStatus(cuModuleUnload(m_cuModule));
    m_cuModule = nullptr;
  }
}

void KernelExecutor::unload() { unloadCuModule(); }

void KernelExecutor::setImage(void *image) {
  this->m_image = image;
  printf("image = %p", m_image);
  loadCuModule();
}

bool KernelExecutor::_run(const std::string &functionName) {
  CUresult status = CUDA_SUCCESS;

  if (nullptr == m_image && m_path.length() == 0) {
    printf("Error: image and path not defined. Function name: %s. Probalby was "
           "not executed load() method. \n",
           functionName.c_str());
  }

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

KernelExecutor::~KernelExecutor() {
  unloadCuModule();
  releaseImage();
}

std::byte *KernelExecutor::Load(FILE *f) {
  if (f) {
    fseek(f, 0, SEEK_END);
    long int size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::byte *data = new std::byte[size];
    memset(data, 0, size);
    fread(data, size, 1, f);
    fclose(f);
    return data;
  }
  return nullptr;
}

std::byte *KernelExecutor::Load(const fs::path &path) {
  FILE *f = fopen(path.c_str(), "rb");
  if (f != nullptr) {
    auto *data = Load(f);
    return data;
  }
  return nullptr;
}

std::byte *KernelExecutor::Load(const fs::path &path,
                                const KernelExecutor::Pathes &sysPathes) {
  for (size_t idx = 0; idx < sysPathes.size(); ++idx) {
    auto p = sysPathes[idx] / path;
    printf("%s", p.c_str());
    if (!fs::exists(p)) {
      return nullptr;
    }
    auto *data = Load(p);
    if (data != nullptr) {
      return data;
    }
  }
  return nullptr;
}

KernelExecutor::Pathes KernelExecutor::GetSysPathes() {
  Pathes pathes;
  auto split = [&pathes](const std::string &env, char c) {
    size_t pindex = 0;
    size_t index = 0;
    while ((index = env.find(c, pindex)) != std::string::npos) {
      pathes.push_back(env.substr(pindex, index - pindex));
    }
    pathes.push_back(env.substr(pindex, env.length() - pindex));
  };
  char *cenvs = ::getenv(ENVIRONMENT_VARIABLE);
  if (cenvs != nullptr) {
    std::string env = cenvs;
    split(env, ':');
  }
  return pathes;
}

bool KernelExecutor::load(const fs::path &path) {
  const auto &pathes = GetSysPathes();
  setImage(Load(path, pathes));
  if (m_image == nullptr) {
    printf("Cannot load %s.", path.c_str());
  } else {
    printf("Loaded %s.", path.c_str());
  }
  return m_image != nullptr;
}

bool KernelExecutor::load(const KernelExecutor::Pathes &pathes) {
  for (const auto &path : pathes) {
    if (load(path)) {
      return true;
    }
  }
  return false;
}

void KernelExecutor::releaseImage() {
  if (m_image != nullptr) {
    printf("~image = %p", m_image);
    std::byte *data = static_cast<std::byte *>(m_image);
    delete[] data;
    m_image = nullptr;
  }
}
} // namespace mtrx
