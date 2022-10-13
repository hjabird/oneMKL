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
    throw unimplemented("blas", "gemv", " for complex");
}

void gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "gemv", " for complex");
}

void gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, float alpha, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          sycl::buffer<float, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "gbmv", "");
}

void gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, double alpha, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "gbmv", "");
}

void gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "gbmv", "");
}

void gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "gbmv", "");
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
    throw unimplemented("blas", "gerc", "");
}

void gerc(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    throw unimplemented("blas", "gerc", "");
}

void geru(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    throw unimplemented("blas", "geru", "");
}

void geru(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    throw unimplemented("blas", "geru", "");
}

void hbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "hbmv", "");
}

void hbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "hbmv", "");
}

void hemv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "hemv", "");
}

void hemv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "hemv", "");
}

void her(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    throw unimplemented("blas", "her", "");
}

void her(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    throw unimplemented("blas", "her", "");
}
void her2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    throw unimplemented("blas", "her2", "");
}

void her2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    throw unimplemented("blas", "her2", "");
}

void hpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "hpmv", "");
}

void hpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "hpmv", "");
}

void hpr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1> &a) {
    throw unimplemented("blas", "hpr", "");
}

void hpr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1> &a) {
    throw unimplemented("blas", "hpr", "");
}

void hpr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a) {
    throw unimplemented("blas", "hpr2", "");
}

void hpr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a) {
    throw unimplemented("blas", "hpr2", "");
}

void sbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          float alpha, sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "sbmv", "");
}

void sbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "sbmv", "");
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
    throw unimplemented("blas", "spmv", "");
}

void spmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &a, sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    throw unimplemented("blas", "spmv", "");
}

void spr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &a) {
    throw unimplemented("blas", "spr", "");
}

void spr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &a) {
    throw unimplemented("blas", "spr", "");
}

void spr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &y,
          std::int64_t incy, sycl::buffer<float, 1> &a) {
    throw unimplemented("blas", "spr2", "");
}

void spr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &y,
          std::int64_t incy, sycl::buffer<double, 1> &a) {
    throw unimplemented("blas", "spr2", "");
}

void tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tbmv", "");
}

void tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tbmv", "");
}

void tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tbmv", "");
}

void tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tbmv", "");
}

void tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tbsv", "");
}

void tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tbsv", "");
}

void tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tbsv", "");
}

void tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tbsv", "");
}

void tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tpmv", "");
}

void tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tpmv", "");
}

void tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tpmv", "");
}

void tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tpmv", "");
}
void tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tpsv", "");
}

void tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tpsv", "");
}

void tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tpsv", "");
}

void tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "tpsv", "");
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
    throw unimplemented("blas", "trmv", " for complex");
}

void trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "trmv", " for complex");
}
void trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "trsv", "");
}

void trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a, std::int64_t lda,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "trsv", "");
}

void trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "trsv", "");
}

void trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    throw unimplemented("blas", "trsv", "");
}

// USM APIs

sycl::event gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                 float beta, float *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv", " for USM");
}

sycl::event gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 double alpha, const double *a, std::int64_t lda, const double *x,
                 std::int64_t incx, double beta, double *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv", " for USM");
}

sycl::event gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                 const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                 std::complex<float> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv", " for USM");
}

sycl::event gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                 const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                 std::complex<double> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv", " for USM");
}

sycl::event gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, float alpha, const float *a, std::int64_t lda,
                 const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gbmv", " for USM");
}

sycl::event gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, double alpha, const double *a, std::int64_t lda,
                 const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gbmv", " for USM");
}

sycl::event gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                 const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                 std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gbmv", " for USM");
}

sycl::event gbmv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                 const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                 std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gbmv", " for USM");
}

sycl::event ger(sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha, const float *x,
                std::int64_t incx, const float *y, std::int64_t incy, float *a, std::int64_t lda,
                const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "ger", " for USM");
}

sycl::event ger(sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha, const double *x,
                std::int64_t incx, const double *y, std::int64_t incy, double *a, std::int64_t lda,
                const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "ger", " for USM");
}

sycl::event gerc(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
                 std::int64_t incy, std::complex<float> *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gerc", " for USM");
}

sycl::event gerc(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
                 std::int64_t incy, std::complex<double> *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gerc", " for USM");
}

sycl::event geru(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
                 std::int64_t incy, std::complex<float> *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "geru", " for USM");
}

sycl::event geru(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
                 std::int64_t incy, std::complex<double> *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "geru", " for USM");
}

sycl::event hbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                 const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                 std::complex<float> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hbmv", " for USM");
}

sycl::event hbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                 const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                 std::complex<double> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hbmv", " for USM");
}

sycl::event hemv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                 const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                 std::complex<float> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hemv", " for USM");
}

sycl::event hemv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                 const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                 std::complex<double> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hemv", " for USM");
}

sycl::event her(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "her", " for USM");
}

sycl::event her(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "her", " for USM");
}

sycl::event her2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                 const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                 std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "her2", " for USM");
}

sycl::event her2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                 const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                 std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "her2", " for USM");
}

sycl::event hpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float> *a,
                 const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                 std::complex<float> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hpmv", " for USM");
}

sycl::event hpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double> *a,
                 const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                 std::complex<double> *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hpmv", " for USM");
}

sycl::event hpr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hpr", " for USM");
}

sycl::event hpr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hpr", " for USM");
}

sycl::event hpr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                 const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hpr2", " for USM");
}

sycl::event hpr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                 const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hpr2", " for USM");
}

sycl::event sbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                 float beta, float *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "sbmv", " for USM");
}

sycl::event sbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 double alpha, const double *a, std::int64_t lda, const double *x,
                 std::int64_t incx, double beta, double *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "sbmv", " for USM");
}

sycl::event symv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                 const float *a, std::int64_t lda, const float *x, std::int64_t incx, float beta,
                 float *y, std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "symv", " for USM");
}

sycl::event symv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                 const double *a, std::int64_t lda, const double *x, std::int64_t incx, double beta,
                 double *y, std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "symv", " for USM");
}

sycl::event syr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                const float *x, std::int64_t incx, float *a, std::int64_t lda,
                const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr", " for USM");
}

sycl::event syr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                const double *x, std::int64_t incx, double *a, std::int64_t lda,
                const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr", " for USM");
}

sycl::event syr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                 const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                 std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr2", " for USM");
}

sycl::event syr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                 const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
                 std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr2", " for USM");
}

sycl::event spmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                 const float *a, const float *x, std::int64_t incx, float beta, float *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "spmv", " for USM");
}

sycl::event spmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                 const double *a, const double *x, std::int64_t incx, double beta, double *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "spmv", " for USM");
}

sycl::event spr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                const float *x, std::int64_t incx, float *a,
                const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "spr", " for USM");
}

sycl::event spr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                const double *x, std::int64_t incx, double *a,
                const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "spr", " for USM");
}

sycl::event spr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                 const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "spr2", " for USM");
}

sycl::event spr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
                 const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "spr2", " for USM");
}

sycl::event tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
                 std::int64_t lda, float *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tbmv", " for USM");
}

sycl::event tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
                 std::int64_t lda, double *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tbmv", " for USM");
}

sycl::event tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                 const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tbmv", " for USM");
}

sycl::event tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                 const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tbmv", " for USM");
}
sycl::event tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
                 std::int64_t lda, float *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tbsv", " for USM");
}

sycl::event tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
                 std::int64_t lda, double *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tbsv", " for USM");
}

sycl::event tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                 const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tbsv", " for USM");
}

sycl::event tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                 const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tbsv", " for USM");
}

sycl::event tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, float *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tpmv", " for USM");
}

sycl::event tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, double *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tpmv", " for USM");
}

sycl::event tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
                 std::complex<float> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tpmv", " for USM");
}

sycl::event tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
                 std::complex<double> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tpmv", " for USM");
}

sycl::event tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, float *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tpsv", " for USM");
}

sycl::event tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, double *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tpsv", " for USM");
}

sycl::event tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
                 std::complex<float> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tpsv", " for USM");
}

sycl::event tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
                 std::complex<double> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tpsv", " for USM");
}

sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, std::int64_t lda,
                 float *x, std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmv", " for USM");
}

sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, std::int64_t lda,
                 double *x, std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmv", " for USM");
}

sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
                 std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmv", " for USM");
}

sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
                 std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmv", " for USM");
}

sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, std::int64_t lda,
                 float *x, std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsv", " for USM");
}

sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, std::int64_t lda,
                 double *x, std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsv", " for USM");
}

sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
                 std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsv", " for USM");
}

sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
                 std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsv", " for USM");
}
