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

void gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          float alpha, sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
    CALL_SYCLBLAS_FN(::blas::_gemv, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
    CALL_SYCLBLAS_FN(::blas::_gemv, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, float alpha, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          sycl::buffer<float, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, double alpha, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void ger(sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1> &x,
         std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy, sycl::buffer<float, 1> &a,
         std::int64_t lda) {
    CALL_SYCLBLAS_FN(::blas::_ger, queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void ger(sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
         sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &y,
         std::int64_t incy, sycl::buffer<double, 1> &a, std::int64_t lda) {
    CALL_SYCLBLAS_FN(::blas::_ger, queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    throw std::runtime_error("Not implemented for syclblas");
}

void gerc(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    throw std::runtime_error("Not implemented for syclblas");
}

void geru(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    throw std::runtime_error("Not implemented for syclblas");
}

void geru(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    throw std::runtime_error("Not implemented for syclblas");
}

void hbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void hbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void hemv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void hemv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void her(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    throw std::runtime_error("Not implemented for syclblas");
}

void her(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    throw std::runtime_error("Not implemented for syclblas");
}
void her2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    throw std::runtime_error("Not implemented for syclblas");
}

void her2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    throw std::runtime_error("Not implemented for syclblas");
}

void hpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void hpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void hpr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1> &a) {
    throw std::runtime_error("Not implemented for syclblas");
}

void hpr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1> &a) {
    throw std::runtime_error("Not implemented for syclblas");
}

void hpr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a) {
    throw std::runtime_error("Not implemented for syclblas");
}

void hpr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a) {
    throw std::runtime_error("Not implemented for syclblas");
}

void sbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          float alpha, sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void sbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void symv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx,
          float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
    CALL_SYCLBLAS_FN(::blas::_symv, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void symv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
    CALL_SYCLBLAS_FN(::blas::_symv, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void syr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &a,
         std::int64_t lda) {
    CALL_SYCLBLAS_FN(::blas::_syr, queue, upper_lower, n, alpha, x, incx, a, lda);
}

void syr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &a,
         std::int64_t lda) {
    CALL_SYCLBLAS_FN(::blas::_syr, queue, upper_lower, n, alpha, x, incx, a, lda);
}

void syr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &y,
          std::int64_t incy, sycl::buffer<float, 1> &a, std::int64_t lda) {
    CALL_SYCLBLAS_FN(::blas::_syr2, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void syr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &y,
          std::int64_t incy, sycl::buffer<double, 1> &a, std::int64_t lda) {
    CALL_SYCLBLAS_FN(::blas::_syr2, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void spmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &a, sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          sycl::buffer<float, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void spmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &a, sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    throw std::runtime_error("Not implemented for syclblas");
}

void spr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &a) {
    throw std::runtime_error("Not implemented for syclblas");
}

void spr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &a) {
    throw std::runtime_error("Not implemented for syclblas");
}

void spr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &y,
          std::int64_t incy, sycl::buffer<float, 1> &a) {
    throw std::runtime_error("Not implemented for syclblas");
}

void spr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &y,
          std::int64_t incy, sycl::buffer<double, 1> &a) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}
void tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    CALL_SYCLBLAS_FN(::blas::_trmv, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a, std::int64_t lda,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    CALL_SYCLBLAS_FN(::blas::_trmv, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}
void trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a, std::int64_t lda,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

void trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw std::runtime_error("Not implemented for syclblas");
}

// USM APIs

sycl::event gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                 float beta, float *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 double alpha, const double *a, std::int64_t lda, const double *x,
                 std::int64_t incx, double beta, double *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                 const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                 std::complex<float> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                 const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                 std::complex<double> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, float alpha, const float *a, std::int64_t lda,
                 const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, double alpha, const double *a, std::int64_t lda,
                 const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                 const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                 std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                 const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                 std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event ger(sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha, const float *x,
                std::int64_t incx, const float *y, std::int64_t incy, float *a, std::int64_t lda,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event ger(sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha, const double *x,
                std::int64_t incx, const double *y, std::int64_t incy, double *a, std::int64_t lda,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event gerc(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
                 std::int64_t incy, std::complex<float> *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event gerc(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
                 std::int64_t incy, std::complex<double> *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event geru(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
                 std::int64_t incy, std::complex<float> *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event geru(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
                 std::int64_t incy, std::complex<double> *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event hbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                 const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                 std::complex<float> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event hbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                 const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                 std::complex<double> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event hemv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                 const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                 std::complex<float> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event hemv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                 const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                 std::complex<double> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event her(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event her(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event her2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                 const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                 std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event her2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                 const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                 std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event hpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float> *a,
                 const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                 std::complex<float> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event hpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double> *a,
                 const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                 std::complex<double> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event hpr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event hpr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event hpr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                 const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event hpr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                 const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event sbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                 float beta, float *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event sbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 double alpha, const double *a, std::int64_t lda, const double *x,
                 std::int64_t incx, double beta, double *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event symv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                 const float *a, std::int64_t lda, const float *x, std::int64_t incx, float beta,
                 float *y, std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event symv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                 const double *a, std::int64_t lda, const double *x, std::int64_t incx, double beta,
                 double *y, std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event syr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                const float *x, std::int64_t incx, float *a, std::int64_t lda,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event syr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                const double *x, std::int64_t incx, double *a, std::int64_t lda,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event syr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                 const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                 std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event syr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                 const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
                 std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event spmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                 const float *a, const float *x, std::int64_t incx, float beta, float *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event spmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                 const double *a, const double *x, std::int64_t incx, double beta, double *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event spr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                const float *x, std::int64_t incx, float *a,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event spr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                const double *x, std::int64_t incx, double *a,
                const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event spr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                 const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event spr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                 const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
                 std::int64_t lda, float *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
                 std::int64_t lda, double *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                 const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                 const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}
sycl::event tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
                 std::int64_t lda, float *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
                 std::int64_t lda, double *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                 const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                 const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, float *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, double *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
                 std::complex<float> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
                 std::complex<double> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, float *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, double *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
                 std::complex<float> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
                 std::complex<double> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, std::int64_t lda,
                 float *x, std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, std::int64_t lda,
                 double *x, std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
                 std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
                 std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, std::int64_t lda,
                 float *x, std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, std::int64_t lda,
                 double *x, std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
                 std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}

sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
                 std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for syclblas");
}
