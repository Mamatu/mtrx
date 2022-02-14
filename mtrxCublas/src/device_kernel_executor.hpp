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

#ifndef MTRX_CUBLAS_KERNEL_EXECUTOR_H
#define MTRX_CUBLAS_KERNEL_EXECUTOR_H

#include "ikernel_executor.hpp"
#include "sys_pathes_parser.hpp"

#include <cuda.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace mtrx {
namespace fs = std::filesystem;
class DeviceKernelExecutor : public IKernelExecutor {
public:
  using Strings = std::vector<std::string>;
  DeviceKernelExecutor(int device);
  virtual ~DeviceKernelExecutor();

  static void Init();

  bool load(const fs::path &path = "");
  bool load(const Pathes &pathes);

  void unload();

protected:
  bool _run(const std::string &functionName) override;

private:
  constexpr static auto ENVIRONMENT_VARIABLE = "MTRX_CUBIN_PATHES";
  constexpr static auto CUBIN_FILE_NAME = "libmtrxCuda.cubin";
  using FileUnique = std::unique_ptr<FILE, std::function<void(FILE *)>>;

  CUdevice m_device;
  void *m_image = nullptr;
  std::string m_path;

  CUmodule m_cuModule;

  void releaseImage();
  void unloadCuModule();
  void loadCuModule();
  void setImage(void *image);
  bool setImageFromPathes(const Pathes &pathes);

  static char *Load(FileUnique &&f);
  static char *Load(const fs::path &path);
  static char *Load(const fs::path &path, const Pathes &sysPathes);
  static Pathes FindExist(const fs::path &path, const Pathes &sysPathes);

  static Pathes GetSysPathes();
};
} // namespace mtrx

#endif
