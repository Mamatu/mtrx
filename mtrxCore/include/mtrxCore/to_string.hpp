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

#ifndef MTRX_CORE_TO_STRING_HPP
#define MTRX_CORE_TO_STRING_HPP

#include "types.hpp"
#include <sstream>
#include <string>

namespace mtrx {
inline std::string toString(ValueType valueType) {
  switch (valueType) {
  case ValueType::FLOAT:
    return "FLOAT";
  case ValueType::DOUBLE:
    return "DOUBLE";
  case ValueType::FLOAT_COMPLEX:
    return "FLOAT_DOUBLE";
  case ValueType::DOUBLE_COMPLEX:
    return "DOUBLE_DOUBLE";
  case ValueType::NOT_DEFINED:
    return "NOT_DEFINED";
  };
  return "NOT_DEFINED";
}

inline std::string toString(Operation operation) {
  switch (operation) {
  case Operation::OP_N:
    return "OP_N";
  case Operation::OP_T:
    return "OP_T";
  case Operation::OP_C:
    return "OP_C";
  };
  return "NOT_DEFINED";
}

inline std::string toString(FillMode fillMode) {
  switch (fillMode) {
  case FillMode::FULL:
    return "FULL";
  case FillMode::LOWER:
    return "LOWER";
  case FillMode::UPPER:
    return "UPPER";
  };
  return "NOT_DEFINED";
}

inline std::string toString(SideMode sideMode) {
  switch (sideMode) {
  case SideMode::LEFT:
    return "LEFT";
  case SideMode::RIGHT:
    return "RIGHT";
  };
  return "NOT_DEFINED";
}

template <typename Container> std::string toString(Container &&container) {
  std::stringstream sstream;
  sstream << "[";
  for (size_t idx = 0; idx < container.size(); ++idx) {
    sstream << container[idx];
    if (idx < container.size() - 1) {
      sstream << ", ";
    }
  }
  sstream << "]";
  return sstream.str();
}

} // namespace mtrx

#endif
