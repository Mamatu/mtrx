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

#ifndef MTRX_CUBLAS_CUDA_MATH_UTILS_HPP
#define MTRX_CUBLAS_CUDA_MATH_UTILS_HPP

#include "cuda_core.hpp"
#include <cuComplex.h>

__device__ __inline__ float cuda_magnitude(cuComplex a) {
  return sqrt(a.x * a.x + a.y * a.y);
}

__device__ __inline__ double cuda_magnitude(cuDoubleComplex a) {
  return sqrt(a.x * a.x + a.y * a.y);
}

__device__ __inline__ bool cuda_isHigher(float a, float b) { return a > b; }

__device__ __inline__ bool cuda_isHigher(double a, double b) { return a > b; }

__device__ __inline__ bool cuda_isHigher(cuComplex a, cuComplex b) {
  return cuda_magnitude(a) > cuda_magnitude(b);
}

__device__ __inline__ bool cuda_isHigher(cuDoubleComplex a, cuDoubleComplex b) {
  return cuda_magnitude(a) > cuda_magnitude(b);
}

__device__ __inline__ bool cuda_isLower(float a, float b) { return a < b; }

__device__ __inline__ bool cuda_isLower(double a, double b) { return a < b; }

__device__ __inline__ bool cuda_isLower(cuComplex a, cuComplex b) {
  return cuda_magnitude(a) < cuda_magnitude(b);
}

__device__ __inline__ bool cuda_isLower(cuDoubleComplex a, cuDoubleComplex b) {
  return cuda_magnitude(a) < cuda_magnitude(b);
}

__device__ __inline__ bool cuda_isAbsHigher(float a, float b) {
  return abs(a) > abs(b);
}

__device__ __inline__ bool cuda_isAbsHigher(double a, double b) {
  return abs(a) > abs(b);
}

__device__ __inline__ bool cuda_isAbsHigher(cuComplex a, cuComplex b) {
  return cuda_magnitude(a) > cuda_magnitude(b);
}

__device__ __inline__ bool cuda_isAbsHigher(cuDoubleComplex a,
                                            cuDoubleComplex b) {
  return cuda_magnitude(a) > cuda_magnitude(b);
}

__device__ __inline__ bool cuda_isAbsLower(float a, float b) {
  return abs(a) < abs(b);
}

__device__ __inline__ bool cuda_isAbsLower(double a, double b) {
  return abs(a) < abs(b);
}

__device__ __inline__ bool cuda_isAbsLower(cuComplex a, cuComplex b) {
  return cuda_magnitude(a) < cuda_magnitude(b);
}

__device__ __inline__ bool cuda_isAbsLower(cuDoubleComplex a,
                                           cuDoubleComplex b) {
  return cuda_magnitude(a) < cuda_magnitude(b);
}

#endif
