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

#ifndef MTRX_CUBLAS_KERNELS_H
#define MTRX_CUBLAS_KERNELS_H

#include <cuComplex.h>

void Kernel_SF_scaleTrace(int dim, float *matrix, int lda, float factor);
void Kernel_SD_scaleTrace(int dim, double *matrix, int lda, double factor);
void Kernel_CF_scaleTrace(int dim, cuComplex *matrix, int lda,
                          cuComplex factor);
void Kernel_CD_scaleTrace(int dim, cuDoubleComplex *matrix, int lda,
                          cuDoubleComplex factor);

#endif
