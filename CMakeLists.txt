cmake_minimum_required(VERSION 3.20)
project(InferenceScheduler VERSION 1.0.0 LANGUAGES CXX)

option(IS_BUILD_CUDA "Build the CUDA backend and hardware tests" ON)
option(IS_BUILD_TESTS "Build tests" ON)
option(IS_BUILD_EXAMPLES "Build examples" ON)
option(IS_BUILD_BENCHMARKS "Build benchmarks" ON)
option(IS_BUILD_CLI "Build the CLI and apps" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/WX>)
endif()

if(IS_BUILD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()

set(IS_LIB_SRC
  src/error.cpp
  src/enums.cpp
  src/persistence.cpp
  src/scheduler.cpp
  src/net.cpp
  src/cpu_executor.cpp
  src/local_driver.cpp
)
if(IS_BUILD_CUDA)
  list(APPEND IS_LIB_SRC src/cuda_executor.cu)
endif()

add_library(inference_scheduler STATIC ${IS_LIB_SRC})
target_include_directories(inference_scheduler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(inference_scheduler PUBLIC cxx_std_20)
set_target_properties(inference_scheduler PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME InferenceScheduler)
if(IS_BUILD_CUDA)
  target_link_libraries(inference_scheduler PUBLIC CUDA::cudart)
  target_compile_options(inference_scheduler PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
  target_compile_options(inference_scheduler PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>)
  # NVCC's generated .cudafe stub emits MSVC C4211 ('redefined extern to static')
  # for code we do not author; disable that single codegen warning only.
  target_compile_options(inference_scheduler PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>)
endif()

add_library(InferenceScheduler::InferenceScheduler ALIAS inference_scheduler)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS inference_scheduler
  EXPORT InferenceSchedulerTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/inference_scheduler DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
  cmake/InferenceSchedulerConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/InferenceSchedulerConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/InferenceScheduler"
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/InferenceSchedulerConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(EXPORT InferenceSchedulerTargets
  FILE InferenceSchedulerTargets.cmake
  NAMESPACE InferenceScheduler::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/InferenceScheduler"
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/InferenceSchedulerConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/InferenceSchedulerConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/InferenceScheduler"
)

if(IS_BUILD_BENCHMARKS)
  add_executable(scheduler_bench benchmarks/scheduler_bench.cpp)
  target_link_libraries(scheduler_bench PRIVATE inference_scheduler)
  target_include_directories(scheduler_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

enable_testing()

if(IS_BUILD_EXAMPLES)
  foreach(e basic_admission fairness dynamic_batching prefill_decode cancellation_retry explain_decision)
    add_executable(example_${e} examples/${e}.cpp)
    target_link_libraries(example_${e} PRIVATE inference_scheduler)
    target_include_directories(example_${e} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endforeach()
endif()

if(IS_BUILD_TESTS)
  set(IS_TEST_PROGRAMS smoke adversarial_test admission_test fairness_test deadline_test cancel_retry_test batch_test stale_test state_machine_test persistence_test property_test)
  foreach(t ${IS_TEST_PROGRAMS})
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE inference_scheduler)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
  if(WIN32)
    add_test(NAME multiprocess_proof COMMAND powershell -ExecutionPolicy Bypass -File ${CMAKE_CURRENT_SOURCE_DIR}/scripts/multiprocess_proof.ps1 -Port 29840)
  endif()
endif()

if(IS_BUILD_CUDA AND IS_BUILD_TESTS)
  add_executable(test_cuda tests/cuda_test.cpp)
  target_link_libraries(test_cuda PRIVATE inference_scheduler)
  target_include_directories(test_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME cuda_test COMMAND test_cuda)
endif()

if(IS_BUILD_CLI)
  add_executable(inference_scheduler_coordinator src/coordinator_app.cpp)
  target_link_libraries(inference_scheduler_coordinator PRIVATE inference_scheduler)
  target_include_directories(inference_scheduler_coordinator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

  add_executable(inference_scheduler_worker src/worker_app.cpp)
  target_link_libraries(inference_scheduler_worker PRIVATE inference_scheduler)
  target_include_directories(inference_scheduler_worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

  add_executable(inference_scheduler_client src/client_app.cpp)
  target_link_libraries(inference_scheduler_client PRIVATE inference_scheduler)
  target_include_directories(inference_scheduler_client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

  add_executable(inference_scheduler_cli src/cli.cpp)
  target_link_libraries(inference_scheduler_cli PRIVATE inference_scheduler)
  target_include_directories(inference_scheduler_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()