/*******************************************************************************
* Copyright Codeplay Software
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

// Buffer APIs

void dotc(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void dotc(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void dotu(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void dotu(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void iamax(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void iamax(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void iamax(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void iamax(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void iamin(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void iamin(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void iamin(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void iamin(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<double, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &result) {
    CALL_SYCLBLAS_FN(::blas::_asum, queue, n, x, incx, result);
}

void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &result) {
    CALL_SYCLBLAS_FN(::blas::_asum, queue, n, x, incx, result);
}

void axpy(sycl::queue &queue, std::int64_t n, float alpha, sycl::buffer<float, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy) {
    CALL_SYCLBLAS_FN(::blas::_axpy, queue, n, alpha, x, incx, y, incy);
}

void axpy(sycl::queue &queue, std::int64_t n, double alpha, sycl::buffer<double, 1> &x,
          std::int64_t incx, sycl::buffer<double, 1> &y, std::int64_t incy) {
    CALL_SYCLBLAS_FN(::blas::_axpy, queue, n, alpha, x, incx, y, incy);
}

void axpy(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void axpy(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void axpby(sycl::queue &queue, std::int64_t n, float alpha, sycl::buffer<float, 1> &x,
           std::int64_t incx, float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
    // TODO: UNCHECKED
}

void axpby(sycl::queue &queue, std::int64_t n, double alpha, sycl::buffer<double, 1> &x,
           std::int64_t incx, double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
    // TODO: UNCHECKED
}

void axpby(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
    // TODO: UNCHECKED
}

void axpby(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
    // TODO: UNCHECKED
}

void copy(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &y, std::int64_t incy) {
    CALL_SYCLBLAS_FN(::blas::_copy, queue, n, x, incx, y, incy);
}

void copy(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    CALL_SYCLBLAS_FN(::blas::_copy, queue, n, x, incx, y, incy);
}

void copy(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void copy(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void dot(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
         sycl::buffer<float, 1> &y, std::int64_t incy, sycl::buffer<float, 1> &result) {
    CALL_SYCLBLAS_FN(::blas::_dot, queue, n, x, incx, y, incy, result);
}

void dot(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
         sycl::buffer<double, 1> &y, std::int64_t incy, sycl::buffer<double, 1> &result) {
    CALL_SYCLBLAS_FN(::blas::_dot, queue, n, x, incx, y, incy, result);
}

void dot(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
         sycl::buffer<float, 1> &y, std::int64_t incy, sycl::buffer<double, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void sdsdot(sycl::queue &queue, std::int64_t n, float sb, sycl::buffer<float, 1> &x,
            std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
            sycl::buffer<float, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}
void nrm2(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void nrm2(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<double, 1> &result) {
    throw std::runtime_error("Not implemented for syclblas");
}

void nrm2(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &result) {
    CALL_SYCLBLAS_FN(::blas::_nrm2, queue, n, x, incx, result);
}

void nrm2(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &result) {
    CALL_SYCLBLAS_FN(::blas::_nrm2, queue, n, x, incx, result);
}

void rot(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
         std::int64_t incx, sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
         float s) {
    throw std::runtime_error("Not implemented for syclblas");
}

void rot(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
         std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c,
         double s) {
    throw std::runtime_error("Not implemented for syclblas");
}

void rot(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
         sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s) {
    CALL_SYCLBLAS_FN(::blas::_rot, queue, n, x, incx, y, incy, c, s);
}

void rot(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
         sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s) {
    CALL_SYCLBLAS_FN(::blas::_rot, queue, n, x, incx, y, incy, c, s);
}

void rotg(sycl::queue &queue, sycl::buffer<float, 1> &a, sycl::buffer<float, 1> &b,
          sycl::buffer<float, 1> &c, sycl::buffer<float, 1> &s) {
    throw std::runtime_error("Not implemented for syclblas");
}

void rotg(sycl::queue &queue, sycl::buffer<double, 1> &a, sycl::buffer<double, 1> &b,
          sycl::buffer<double, 1> &c, sycl::buffer<double, 1> &s) {
    throw std::runtime_error("Not implemented for syclblas");
}

void rotg(sycl::queue &queue, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &b, sycl::buffer<float, 1> &c,
          sycl::buffer<std::complex<float>, 1> &s) {
    throw std::runtime_error("Not implemented for syclblas");
}

void rotg(sycl::queue &queue, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &b, sycl::buffer<double, 1> &c,
          sycl::buffer<std::complex<double>, 1> &s) {
    throw std::runtime_error("Not implemented for syclblas");
}

void rotm(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &y, std::int64_t incy, sycl::buffer<float, 1> &param) {
    throw std::runtime_error("Not implemented for syclblas");
}

void rotm(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &y, std::int64_t incy, sycl::buffer<double, 1> &param) {
    throw std::runtime_error("Not implemented for syclblas");
}

void rotmg(sycl::queue &queue, sycl::buffer<float, 1> &d1, sycl::buffer<float, 1> &d2,
           sycl::buffer<float, 1> &x1, float y1, sycl::buffer<float, 1> &param) {
    throw std::runtime_error("Not implemented for syclblas");
}

void rotmg(sycl::queue &queue, sycl::buffer<double, 1> &d1, sycl::buffer<double, 1> &d2,
           sycl::buffer<double, 1> &x1, double y1, sycl::buffer<double, 1> &param) {
    throw std::runtime_error("Not implemented for syclblas");
}

void scal(sycl::queue &queue, std::int64_t n, float alpha, sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    CALL_SYCLBLAS_FN(::blas::_scal, queue, n, alpha, x, incx);
}

void scal(sycl::queue &queue, std::int64_t n, double alpha, sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    CALL_SYCLBLAS_FN(::blas::_scal, queue, n, alpha, x, incx);
}

void scal(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void scal(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void scal(sycl::queue &queue, std::int64_t n, float alpha, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void scal(sycl::queue &queue, std::int64_t n, double alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void swap(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &y, std::int64_t incy) {
    CALL_SYCLBLAS_FN(::blas::_swap, queue, n, x, incx, y, incy);
}

void swap(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    CALL_SYCLBLAS_FN(::blas::_swap, queue, n, x, incx, y, incy);
}

void swap(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void swap(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

// USM APIs

sycl::event dotc(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                 std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                 std::complex<float> *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event dotc(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                 std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                 std::complex<double> *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event dotu(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                 std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                 std::complex<float> *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event dotu(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                 std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                 std::complex<double> *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event iamax(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                  std::int64_t *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event iamax(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                  std::int64_t *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event iamax(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                  std::int64_t incx, std::int64_t *result,
                  const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event iamax(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                  std::int64_t incx, std::int64_t *result,
                  const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event iamin(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                  std::int64_t *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event iamin(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                  std::int64_t *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event iamin(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                  std::int64_t incx, std::int64_t *result,
                  const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event iamin(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                  std::int64_t incx, std::int64_t *result,
                  const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event asum(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                 std::int64_t incx, float *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event asum(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                 std::int64_t incx, double *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event asum(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                 float *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event asum(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                 double *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event axpy(sycl::queue &queue, std::int64_t n, float alpha, const float *x, std::int64_t incx,
                 float *y, std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event axpy(sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                 std::int64_t incx, double *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event axpy(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event axpy(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}
sycl::event axpby(sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                  std::int64_t incx, const float beta, float *y, std::int64_t incy,
                  const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event axpby(sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                  std::int64_t incx, const double beta, double *y, std::int64_t incy,
                  const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event axpby(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                  const std::complex<float> *x, std::int64_t incx, const std::complex<float> beta,
                  std::complex<float> *y, std::int64_t incy,
                  const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event axpby(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                  const std::complex<double> *x, std::int64_t incx, const std::complex<double> beta,
                  std::complex<double> *y, std::int64_t incy,
                  const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event copy(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event copy(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event copy(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                 std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event copy(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                 std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}
sycl::event dot(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                const float *y, std::int64_t incy, float *result,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event dot(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                const double *y, std::int64_t incy, double *result,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event dot(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                const float *y, std::int64_t incy, double *result,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event sdsdot(sycl::queue &queue, std::int64_t n, float sb, const float *x, std::int64_t incx,
                   const float *y, std::int64_t incy, float *result,
                   const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event nrm2(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                 std::int64_t incx, float *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event nrm2(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                 std::int64_t incx, double *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event nrm2(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                 float *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event nrm2(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                 double *result, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rot(sycl::queue &queue, std::int64_t n, std::complex<float> *x, std::int64_t incx,
                std::complex<float> *y, std::int64_t incy, float c, float s,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rot(sycl::queue &queue, std::int64_t n, std::complex<double> *x, std::int64_t incx,
                std::complex<double> *y, std::int64_t incy, double c, double s,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rot(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                std::int64_t incy, float c, float s, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rot(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
                std::int64_t incy, double c, double s,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rotg(sycl::queue &queue, float *a, float *b, float *c, float *s,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rotg(sycl::queue &queue, double *a, double *b, double *c, double *s,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rotg(sycl::queue &queue, std::complex<float> *a, std::complex<float> *b, float *c,
                 std::complex<float> *s, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rotg(sycl::queue &queue, std::complex<double> *a, std::complex<double> *b, double *c,
                 std::complex<double> *s, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rotm(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                 std::int64_t incy, float *param, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rotm(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
                 std::int64_t incy, double *param, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rotmg(sycl::queue &queue, float *d1, float *d2, float *x1, float y1, float *param,
                  const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event rotmg(sycl::queue &queue, double *d1, double *d2, double *x1, double y1, double *param,
                  const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event scal(sycl::queue &queue, std::int64_t n, float alpha, float *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event scal(sycl::queue &queue, std::int64_t n, double alpha, double *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event scal(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                 std::complex<float> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event scal(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                 std::complex<double> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event scal(sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event scal(sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event swap(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event swap(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event swap(sycl::queue &queue, std::int64_t n, std::complex<float> *x, std::int64_t incx,
                 std::complex<float> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event swap(sycl::queue &queue, std::int64_t n, std::complex<double> *x, std::int64_t incx,
                 std::complex<double> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}
