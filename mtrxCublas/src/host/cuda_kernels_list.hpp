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

#ifndef MTRX_CUBLAS_CUDA_KERNELS_LIST_H
#define MTRX_CUBLAS_CUDA_KERNELS_LIST_H

#include <cuComplex.h>

void HOSTKernel_SF_scaleTrace(int m, int n, float *matrix, int lda,
                              float factor);

void proxy_HOSTKernel_SF_scaleTrace(const void **params);

void HOSTKernel_SD_scaleTrace(int m, int n, double *matrix, int lda,
                              double factor);

void proxy_HOSTKernel_SD_scaleTrace(const void **params);

void HOSTKernel_CF_scaleTrace(int m, int n, cuComplex *matrix, int lda,
                              cuComplex factor);

void proxy_HOSTKernel_CF_scaleTrace(const void **params);

void HOSTKernel_CD_scaleTrace(int m, int n, cuDoubleComplex *matrix, int lda,
                              cuDoubleComplex factor);

void proxy_HOSTKernel_CD_scaleTrace(const void **params);

#endif
