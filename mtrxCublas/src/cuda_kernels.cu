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
#include "cuda_is_ul_triangular.hpp"
#include "cuda_reduce.hpp"

extern "C" __global__ void CudaKernel_SF_scaleTrace(int m, int n, float* matrix, int lda, float factor)
{
  cuda_SF_scaleTrace(m, n, matrix, lda, factor);
}

extern "C" __global__ void CudaKernel_SD_scaleTrace(int m, int n, double* matrix, int lda, double factor)
{
  cuda_SD_scaleTrace(m, n, matrix, lda, factor);
}

extern "C" __global__ void CudaKernel_CF_scaleTrace(int m, int n, cuComplex* matrix, int lda, cuComplex factor)
{
  cuda_CF_scaleTrace(m, n, matrix, lda, factor);
}

extern "C" __global__ void CudaKernel_CD_scaleTrace(int m, int n, cuDoubleComplex* matrix, int lda, cuDoubleComplex factor)
{
  cuda_CD_scaleTrace(m, n, matrix, lda, factor);
}

extern "C" __global__ void CudaKernel_SF_isUpperTriangular(int m, int n, float* matrix, int lda, float delta, int* reductionResults)
{
  cuda_isUpperTriangular(m, n, matrix, lda, delta, reductionResults);
}

extern "C" __global__ void CudaKernel_SD_isUpperTriangular(int m, int n, double* matrix, int lda, double delta, int* reductionResults)
{
  cuda_isUpperTriangular(m, n, matrix, lda, delta, reductionResults);
}

extern "C" __global__ void CudaKernel_CF_isUpperTriangular(int m, int n, cuComplex* matrix, int lda, cuComplex delta, int* reductionResults)
{
  cuda_isUpperTriangular(m, n, matrix, lda, delta, reductionResults);
}

extern "C" __global__ void CudaKernel_CD_isUpperTriangular(int m, int n, cuDoubleComplex* matrix, int lda, cuDoubleComplex delta, int* reductionResults)
{
  cuda_isUpperTriangular(m, n, matrix, lda, delta, reductionResults);
}

extern "C" __global__ void CudaKernel_SF_isLowerTriangular(int m, int n, float* matrix, int lda, float delta, int* reductionResults)
{
  cuda_isLowerTriangular(m, n, matrix, lda, delta, reductionResults);
}

extern "C" __global__ void CudaKernel_SD_isLowerTriangular(int m, int n, double* matrix, int lda, double delta, int* reductionResults)
{
  cuda_isLowerTriangular(m, n, matrix, lda, delta, reductionResults);
}

extern "C" __global__ void CudaKernel_CF_isLowerTriangular(int m, int n, cuComplex* matrix, int lda, cuComplex delta, int* reductionResults)
{
  cuda_isLowerTriangular(m, n, matrix, lda, delta, reductionResults);
}

extern "C" __global__ void CudaKernel_CD_isLowerTriangular(int m, int n, cuDoubleComplex* matrix, int lda, cuDoubleComplex delta, int* reductionResults)
{
  cuda_isLowerTriangular(m, n, matrix, lda, delta, reductionResults);
}

extern "C" __global__ void CudaKernel_SI_reduceShm(int m, int n, int* array, int lda, int* reductionResults)
{
  cuda_reduce_shm<int>(m, n, array, lda, reductionResults);
}
//
//extern "C" __global__ void CudaKernel_SF_reduceShm(int m, int n, float* array, int lda, float* reductionResults)
//{
//  cuda_reduce_shm<float>(m, n, array, lda, reductionResults);
//}
//
//extern "C" __global__ void CudaKernel_SD_reduceShm(int m, int n, double* array, int lda, double* reductionResults)
//{
//  cuda_reduce_shm<double>(m, n, array, lda, reductionResults);
//}
//
//extern "C" __global__ void CudaKernel_CF_reduceShm(int m, int n, cuComplex* array, int lda, cuComplex* reductionResults)
//{
//  cuda_reduce_shm<cuComplex>(m, n, array, lda, reductionResults);
//}
//
//extern "C" __global__ void CudaKernel_CD_reduceShm(int m, int n, cuDoubleComplex* array, int lda, cuDoubleComplex* reductionResults)
//{
//  cuda_reduce_shm<cuDoubleComplex>(m, n, array, lda, reductionResults);
//}
