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

#ifndef MTRX_CORE_STATUS_HANDLER_HPP
#define MTRX_CORE_STATUS_HANDLER_HPP

#include <functional>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include <mtrxCore/to_string.hpp>

namespace mtrx {
template <typename Status, typename ToStringT = std::function<std::string(Status)>>
void handleStatus(
    Status status, const Status success, ToStringT &&toStringT = [](Status status) -> std::string { return toString(status); }) {
  auto exception = [toStringT = std::forward<ToStringT>(toStringT)](auto status) {
    std::stringstream sstream;
    sstream << "Error: " << toStringT(status);
    spdlog::error(sstream.str());
    throw std::runtime_error(sstream.str());
  };

  if (status == success) {
    return;
  }

  exception(status);
}
} // namespace mtrx

#endif
