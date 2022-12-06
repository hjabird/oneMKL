/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef _ONEMKL_DFT_MKLCPU_HPP_
#define _ONEMKL_DFT_MKLCPU_HPP_

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/dft/detail/types_impl.hpp"

namespace oneapi {
namespace mkl {
namespace dft {

namespace detail {
// Predeclarations
class commit_impl;

template <precision prec, domain dom>
class descriptor;
} // namespace detail

namespace mklcpu {
template <dft::detail::precision prec, dft::detail::domain dom>
ONEMKL_EXPORT dft::detail::commit_impl* create_commit(dft::detail::descriptor<prec, dom>& desc);

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif // _ONEMKL_DFT_MKLCPU_HPP_
