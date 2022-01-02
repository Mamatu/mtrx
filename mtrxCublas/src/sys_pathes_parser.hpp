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

#ifndef MTRX_SYS_PATHES_PARSER_HPP
#define MTRX_SYS_PATHES_PARSER_HPP

#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace mtrx {
using Pathes = std::vector<std::string>;
Pathes loadSysPathes(const std::string &ev);
Pathes parseSysPathes(const std::string &pathes);

std::string toString(const mtrx::Pathes &pathes);
} // namespace mtrx

inline std::ostream &operator<<(std::ostream &os, const mtrx::Pathes &pathes) {
  return os << mtrx::toString(pathes);
}

template <> struct fmt::formatter<mtrx::Pathes> : formatter<std::string> {
  template <typename FormatContext>
  auto format(const mtrx::Pathes &pathes, FormatContext &ctx) {
    return formatter<string_view>::format(mtrx::toString(pathes), ctx);
  }
};

#endif
