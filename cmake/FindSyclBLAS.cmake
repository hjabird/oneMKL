#===============================================================================
# Copyright Codeplay Software
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
#===============================================================================

# Try to find the SyclBLAS library.
#
# If the library is found then the `SyclBLAS::SyclBLAS` target will be exported
# with the required include directories.
#
# Sets the following variables:
#   SyclBLAS_FOUND        - whether the system has SyclBLAS
#   SyclBLAS_INCLUDE_DIRS - the SyclBLAS include directory

find_path(SyclBLAS_INCLUDE_DIRS
  NAMES sycl_blas.h
  PATH_SUFFIXES include
  HINTS ${SyclBLAS_DIR}
  DOC "The SyclBLAS include directory"
)

find_library(SyclBLAS_LIBRARIES 
  NAMES sycl_blas libsycl_blas
  HINTS ${SyclBLAS_DIR}
  PATH_SUFFIXES "lib"
  DOC "The SyclBLAS shared library"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SyclBLAS
  FOUND_VAR SyclBLAS_FOUND
  REQUIRED_VARS SyclBLAS_INCLUDE_DIRS
                SyclBLAS_LIBRARIES
)

mark_as_advanced(SyclBLAS_FOUND
                 SyclBLAS_LIBRARIES
)

if(SyclBLAS_FOUND)
  add_library(SyclBLAS::SyclBLAS UNKNOWN IMPORTED)
  set_target_properties(SyclBLAS::SyclBLAS PROPERTIES
    IMPORTED_LOCATION "${SyclBLAS_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${SyclBLAS_INCLUDE_DIRS}"
  )
endif()
