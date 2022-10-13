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

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          sycl::buffer<float, 1> &c, std::int64_t ldc) {
    CALL_SYCLBLAS_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc);
}

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          sycl::buffer<double, 1> &c, std::int64_t ldc) {
    CALL_SYCLBLAS_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc);
}

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemm", " for complex");
}

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemm", " for complex");
}

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
          sycl::buffer<sycl::half, 1> &a, std::int64_t lda, sycl::buffer<sycl::half, 1> &b,
          std::int64_t ldb, sycl::half beta, sycl::buffer<sycl::half, 1> &c, std::int64_t ldc) {
    CALL_SYCLBLAS_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc);
}

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<sycl::half, 1> &a, std::int64_t lda, sycl::buffer<sycl::half, 1> &b,
          std::int64_t ldb, float beta, sycl::buffer<float, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemm", " for different argument data types");
}

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<oneapi::mkl::bfloat16, 1> &a, std::int64_t lda,
          sycl::buffer<oneapi::mkl::bfloat16, 1> &b, std::int64_t ldb, float beta,
          sycl::buffer<float, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemm", " for bfloat16");
}

void symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &b, std::int64_t ldb, float beta, sycl::buffer<float, 1> &c,
          std::int64_t ldc) {
    throw unimplemented("blas", "symm", "");
}

void symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          sycl::buffer<double, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "symm", "");
}

void symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "symm", "");
}

void symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "symm", "");
}

void hemm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "hemm", "");
}

void hemm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "hemm", "");
}

void syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
          float beta, sycl::buffer<float, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syrk", "");
}

void syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1> &a,
          std::int64_t lda, double beta, sycl::buffer<double, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syrk", "");
}

void syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syrk", "");
}

void syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syrk", "");
}

void herk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, float alpha, sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, float beta, sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "herk", "");
}

void herk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, double alpha, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, double beta, sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    throw unimplemented("blas", "herk", "");
}

void syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
           sycl::buffer<float, 1> &b, std::int64_t ldb, float beta, sycl::buffer<float, 1> &c,
           std::int64_t ldc) {
    throw unimplemented("blas", "syr2k", "");
}

void syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1> &a,
           std::int64_t lda, sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           sycl::buffer<double, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syr2k", "");
}

void syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syr2k", "");
}

void syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syr2k", "");
}

void her2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
           sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "her2k", "");
}

void her2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
           sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "her2k", "");
}

void trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          float alpha, sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b,
          std::int64_t ldb) {
    throw unimplemented("blas", "trmm", "");
}

void trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
          std::int64_t ldb) {
    throw unimplemented("blas", "trmm", "");
}

void trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    throw unimplemented("blas", "trmm", "");
}

void trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    throw unimplemented("blas", "trmm", "");
}

void trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          float alpha, sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b,
          std::int64_t ldb) {
    CALL_SYCLBLAS_FN(::blas::_trsm, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb);
}

void trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
          std::int64_t ldb) {
    CALL_SYCLBLAS_FN(::blas::_trsm, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb);
}

void trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    throw unimplemented("blas", "trsm", " for complex");
}

void trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    throw unimplemented("blas", "trsm", " for complex");
}

void gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, float alpha,
           sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b, std::int64_t ldb,
           float beta, sycl::buffer<float, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemmt", "");
}

void gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
           sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, sycl::buffer<double, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemmt", "");
}

void gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemmt", "");
}

void gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemmt", "");
}

void gemm_bias(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
               oneapi::mkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
               sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "");
}

void gemm_bias(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
               oneapi::mkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
               sycl::buffer<int8_t, 1> &b, std::int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "");
}

void gemm_bias(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
               oneapi::mkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, sycl::buffer<uint8_t, 1> &a, std::int64_t lda, uint8_t ao,
               sycl::buffer<int8_t, 1> &b, std::int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "");
}

void gemm_bias(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
               oneapi::mkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, sycl::buffer<uint8_t, 1> &a, std::int64_t lda, uint8_t ao,
               sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "");
}

// USM APIs

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float *a,
                 std::int64_t lda, const float *b, std::int64_t ldb, float beta, float *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double *a,
                 std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                 const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                 std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                 const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                 std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                 const sycl::half *a, std::int64_t lda, const sycl::half *b, std::int64_t ldb,
                 sycl::half beta, sycl::half *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const sycl::half *a,
                 std::int64_t lda, const sycl::half *b, std::int64_t ldb, float beta, float *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                 const oneapi::mkl::bfloat16 *a, std::int64_t lda, const oneapi::mkl::bfloat16 *b,
                 std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

sycl::event symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda,
                 const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "symm", " for USM");
}

sycl::event symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, double alpha, const double *a, std::int64_t lda,
                 const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "symm", " for USM");
}

sycl::event symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                 std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "symm", " for USM");
}

sycl::event symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                 std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "symm", " for USM");
}

sycl::event hemm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                 std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hemm", " for USM");
}

sycl::event hemm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                 std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hemm", " for USM");
}

sycl::event syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                 float beta, float *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk", " for USM");
}

sycl::event syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
                 double beta, double *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk", " for USM");
}

sycl::event syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t k, std::complex<float> alpha,
                 const std::complex<float> *a, std::int64_t lda, std::complex<float> beta,
                 std::complex<float> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk", " for USM");
}

sycl::event syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t k, std::complex<double> alpha,
                 const std::complex<double> *a, std::int64_t lda, std::complex<double> beta,
                 std::complex<double> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk", " for USM");
}

sycl::event herk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t k, float alpha, const std::complex<float> *a,
                 std::int64_t lda, float beta, std::complex<float> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "herk", " for USM");
}

sycl::event herk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t k, double alpha, const std::complex<double> *a,
                 std::int64_t lda, double beta, std::complex<double> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "herk", " for USM");
}

sycl::event syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                  const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr2k", " for USM");
}

sycl::event syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
                  const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr2k", " for USM");
}

sycl::event syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t k, std::complex<float> alpha,
                  const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                  std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                  std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr2k", " for USM");
}

sycl::event syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t k, std::complex<double> alpha,
                  const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                  std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                  std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr2k", " for USM");
}

sycl::event her2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t k, std::complex<float> alpha,
                  const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                  std::int64_t ldb, float beta, std::complex<float> *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "her2k", " for USM");
}

sycl::event her2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t k, std::complex<double> alpha,
                  const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                  std::int64_t ldb, double beta, std::complex<double> *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "her2k", " for USM");
}

sycl::event trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, float alpha, const float *a, std::int64_t lda, float *b,
                 std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmm", " for USM");
}

sycl::event trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, double alpha, const double *a, std::int64_t lda, double *b,
                 std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmm", " for USM");
}

sycl::event trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                 std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmm", " for USM");
}

sycl::event trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                 std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmm", " for USM");
}

sycl::event trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, float alpha, const float *a, std::int64_t lda, float *b,
                 std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm", " for USM");
}

sycl::event trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, double alpha, const double *a, std::int64_t lda, double *b,
                 std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm", " for USM");
}

sycl::event trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                 std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm", " for USM");
}

sycl::event trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                 std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm", " for USM");
}

sycl::event gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                  oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, float alpha,
                  const float *a, std::int64_t lda, const float *b, std::int64_t ldb, float beta,
                  float *c, std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", " for USM");
}

sycl::event gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                  oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
                  const double *a, std::int64_t lda, const double *b, std::int64_t ldb, double beta,
                  double *c, std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", " for USM");
}

sycl::event gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                  oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
                  std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                  const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                  std::complex<float> *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", " for USM");
}

sycl::event gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                  oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
                  std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                  const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                  std::complex<double> *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", " for USM");
}

sycl::event gemm_bias(sycl::queue &queue, oneapi::mkl::transpose transa,
                      oneapi::mkl::transpose transb, oneapi::mkl::offset offsetc, std::int64_t m,
                      std::int64_t n, std::int64_t k, float alpha, const std::int8_t *a,
                      std::int64_t lda, std::int8_t ao, const std::uint8_t *b, std::int64_t ldb,
                      std::uint8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                      const std::int32_t *co, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", " for USM");
}

sycl::event gemm_bias(sycl::queue &queue, oneapi::mkl::transpose transa,
                      oneapi::mkl::transpose transb, oneapi::mkl::offset offsetc, std::int64_t m,
                      std::int64_t n, std::int64_t k, float alpha, const std::int8_t *a,
                      std::int64_t lda, std::int8_t ao, const std::int8_t *b, std::int64_t ldb,
                      std::int8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                      const std::int32_t *co, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", " for USM");
}

sycl::event gemm_bias(sycl::queue &queue, oneapi::mkl::transpose transa,
                      oneapi::mkl::transpose transb, oneapi::mkl::offset offsetc, std::int64_t m,
                      std::int64_t n, std::int64_t k, float alpha, const std::uint8_t *a,
                      std::int64_t lda, std::uint8_t ao, const std::int8_t *b, std::int64_t ldb,
                      std::int8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                      const std::int32_t *co, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", " for USM");
}

sycl::event gemm_bias(sycl::queue &queue, oneapi::mkl::transpose transa,
                      oneapi::mkl::transpose transb, oneapi::mkl::offset offsetc, std::int64_t m,
                      std::int64_t n, std::int64_t k, float alpha, const std::uint8_t *a,
                      std::int64_t lda, std::uint8_t ao, const std::uint8_t *b, std::int64_t ldb,
                      std::uint8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                      const std::int32_t *co, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", " for USM");
}
