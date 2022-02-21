/*
 * Copyright 2021 - 2022 Marcin Matula
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

#ifndef MTRX_CORE_CHECKERS_HPP
#define MTRX_CORE_CHECKERS_HPP

#include "types.hpp"
#include <sstream>
#include <string>
#include <vector>

namespace mtrx {

template <typename T> void checkIfNotNull(T *t, const std::string_view &label) {
  if (t == nullptr) {
    std::stringstream sstream;
    sstream << label << " is null pointer";
    throw std::runtime_error(sstream.str());
  }
}

#define MTRX_CHECK_IF_NOT_NULL(a) checkIfNotNull(a, MTRX_TO_STRING(a))

template <typename T>
void check(const std::vector<T> &data, const std::vector<std::string> &labels) {
  if (data.size() != labels.size()) {
    std::stringstream sstream;
    sstream << "All argument containers must have the same size!";
    throw std::runtime_error(sstream.str());
  }

  for (size_t idx = 0; idx < data.size(); ++idx) {
    for (size_t idx1 = idx; idx1 < data.size(); ++idx1) {
      if (data[idx] != data[idx1]) {
        std::stringstream sstream;
        sstream << labels[idx] << " and " << labels[idx1];
        sstream << " have different dims";
        throw std::runtime_error(sstream.str());
      }
    }
  }
}

void check(const std::vector<ValueType> &valueTypes,
           const std::vector<std::string> &labels);
void check(ValueType valueType1, ValueType valueType2,
           const std::string &label1, const std::string &label2);

template <typename T>
void check(const std::vector<std::pair<T, T>> &dims,
           const std::vector<std::string> &labels) {
  check<std::pair<T, T>>(dims, labels);
}

template <typename T>
void check(const std::pair<T, T> &dims1, const std::pair<T, T> &dims2,
           const std::string &label1, const std::string &label2) {
  check(std::vector<std::pair<T, T>>({dims1, dims2}),
        std::vector<std::string>({label1, label2}));
}

} // namespace mtrx

#endif
