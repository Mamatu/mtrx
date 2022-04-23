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

#ifndef MTRX_DIM3_PARSER_HPP
#define MTRX_DIM3_PARSER_HPP

#include <sstream>
#include <string>

#include <spdlog/spdlog.h>
#include <vector_types.h>

template <> struct fmt::formatter<dim3> : formatter<std::string> {
  template <typename FormatContext>
  auto format(const dim3 &dim, FormatContext &ctx) {
    std::stringstream sstream;
    sstream << dim.x << ", " << dim.y << ", " << dim.z;
    return formatter<string_view>::format(sstream.str(), ctx);
  }
};

#endif
