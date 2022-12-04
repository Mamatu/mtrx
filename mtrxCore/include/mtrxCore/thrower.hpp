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

#ifndef MTRX_CORE_THROWER_HPP
#define MTRX_CORE_THROWER_HPP

#include <sstream>
#include <stdexcept>

namespace mtrx {

class Thrower final {
public:
  Thrower() {}

  Thrower(const Thrower &) = delete;
  Thrower(Thrower &&) = delete;
  Thrower &operator=(const Thrower &) = delete;
  Thrower &operator=(Thrower &&) = delete;

  void throw_exception() const { throw std::runtime_error(m_in.str()); }

  template <typename InCallback> void throw_exception(InCallback &&inCallback) {
    inCallback(m_in);
    throw std::runtime_error(m_in.str());
  }

private:
  std::stringstream m_in;
};

template <typename InCallback>
inline void throw_exception(InCallback &&inCallback) {
  Thrower thrower;
  thrower.throw_exception(inCallback);
}

template <typename InCallback>
inline void throw_exception_ifnot(bool assert_cond, InCallback &&inCallback) {
  if (!assert_cond) {
    throw_exception(std::move(inCallback));
  }
}

} // namespace mtrx
#endif
