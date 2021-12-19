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

#include "cuda_scale_trace.hpp"

__global__ void CUDAKernel_SF_scaleTrace(int m, int n, float* matrix, int lda, float factor)
{
  cuda_SF_scaleTrace(m, n, matrix, lda, factor);
}

__global__ void CUDAKernel_SD_scaleTrace(int m, int n, double* matrix, int lda, double factor)
{
  cuda_SD_scaleTrace(m, n, matrix, lda, factor);
}

__global__ void CUDAKernel_CF_scaleTrace(int m, int n, cuComplex* matrix, int lda, cuComplex factor)
{
  cuda_CF_scaleTrace(m, n, matrix, lda, factor);
}

__global__ void CUDAKernel_CD_scaleTrace(int m, int n, cuDoubleComplex* matrix, int lda, cuDoubleComplex factor)
{
  cuda_CD_scaleTrace(m, n, matrix, lda, factor);
}


