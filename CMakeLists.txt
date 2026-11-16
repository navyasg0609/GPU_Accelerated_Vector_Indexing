cmake_minimum_required(VERSION 3.18)

# Must be set before project(): CMake otherwise fills in the compiler default (sm_52).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
  endif()
endif()

project(gpu_vector_search LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VS_SYNC_CHECK "Synchronize after every kernel launch to localize CUDA errors" OFF)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_library(vsearch STATIC
  src/common.cpp
  src/io.cpp
  src/cpu_ref.cpp
  src/device_info.cu
  src/distance.cu
  src/topk.cu
  src/flat_index.cu
  src/ivf_index.cu
  src/kmeans.cu
  src/bench.cu
)
target_include_directories(vsearch PUBLIC include)
# OpenMP is passed as plain flags/libraries so that nvcc can do the final link.
target_compile_options(vsearch PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>"
  "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
  "$<$<COMPILE_LANGUAGE:CXX>:-march=native>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
target_link_libraries(vsearch PUBLIC CUDA::cudart CUDA::cublas ${OpenMP_CXX_LIBRARIES})
if(VS_SYNC_CHECK)
  target_compile_definitions(vsearch PUBLIC VS_SYNC_CHECK)
endif()

foreach(app flat_search build_ivf ivf_bench kernel_bench ivf_query)
  add_executable(${app} apps/${app}.cu)
  target_link_libraries(${app} PRIVATE vsearch)
endforeach()

add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE vsearch)

enable_testing()
add_test(NAME correctness COMMAND test_correctness)
