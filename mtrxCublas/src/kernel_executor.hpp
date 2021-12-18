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
#include <cuda.h>
#include <filesystem>
#include <string>
#include <vector>

namespace mtrx {
namespace fs = std::filesystem;
class KernelExecutor : public IKernelExecutor {
public:
  using Strings = std::vector<std::string>;
  using Pathes = std::vector<fs::path>;
  KernelExecutor();
  virtual ~KernelExecutor();

  static void Init();

  bool load(const fs::path &path);
  bool load(const Pathes &pathes);

  void unload();

protected:
  bool _run(const std::string &functionName) override;

private:
  void *m_image;
  std::string m_path;
  CUmodule m_cuModule;

  void releaseImage();
  void unloadCuModule();
  void loadCuModule();
  void setImage(void *image);

  static std::byte *Load(FILE *f);
  static std::byte *Load(const fs::path &path);
  static std::byte *Load(const fs::path &path, const Pathes &sysPathes);

  constexpr static auto ENVIRONMENT_VARIABLE = "MTRX_CUBIN_PATHES";
  static Pathes GetSysPathes();
};
} // namespace mtrx

#endif
