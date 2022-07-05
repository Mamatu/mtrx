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

#include <mtrxCore/checkers.hpp>

namespace mtrx {

void checkIfAllEqual(const std::vector<ValueType> &valueTypes,
                     const std::vector<std::string> &labels) {
  checkIfAllEqual<ValueType>(valueTypes, labels);
}

void checkIfAllEqual(ValueType valueType1, ValueType valueType2,
                     const std::string &label1, const std::string &label2) {
  checkIfAllEqual(std::vector<ValueType>({valueType1, valueType2}),
                  std::vector<std::string>({label1, label2}));
}

} // namespace mtrx
