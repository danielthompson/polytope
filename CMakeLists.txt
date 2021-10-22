cmake_minimum_required(VERSION 3.10)

# Options. Turn on with 'cmake -Dmyvarname=ON'.
option(tests "Build all tests." OFF) # Makes boolean 'tests' available.
set(CMAKE_VERBOSE_MAKEFILE ON)

# Make PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR, and PROJECT_NAME available.
set(PROJECT_NAME polytope)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# https://stackoverflow.com/questions/51756562/obtaining-the-cuda-include-dir-in-c-targets-with-native-cuda-support-cmake
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wno-error=unused-function -pthread -fstandalone-debug")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wno-error=unused-function -pthread")

# Address Sanitizer in debug builds
#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-scope")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-scope")

#SET(CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61")
#SET(CMAKE_CUDA_FLAGS " -arch=sm_61")

set(ispc_options -g --pic)

SET(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
SET(CMAKE_CUDA_FLAGS_RELEASE "--generate-line-info")

IF (CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(ispc_options --pic -O3)
  # ELSE()
  #    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
ENDIF (CMAKE_BUILD_TYPE MATCHES RELEASE)

# set (ISPC_EXECUTABLE $<TARGET_FILE:ispc>)

# see http://david-grs.github.io/cpp-clang-travis-cmake-gtest-coveralls-appveyor/
SET(COVERAGE OFF CACHE BOOL "Coverage")

# If you want your own include/ directory, set this, and then you can do
# include_directories(${COMMON_INCLUDES}) in other CMakeLists.txt files.
# set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)

################################
# Normal Libraries & Executables
################################

add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/src/cpu/shapes/mesh_intersect.o
    COMMAND ispc ${ispc_options} -h ${CMAKE_SOURCE_DIR}/src/cpu/shapes/mesh_intersect.h -o ${CMAKE_SOURCE_DIR}/src/cpu/shapes/mesh_intersect.o ${CMAKE_SOURCE_DIR}/src/cpu/shapes/mesh_intersect.ispc
    MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/src/cpu/shapes/mesh_intersect.ispc
    BYPRODUCTS ${CMAKE_SOURCE_DIR}/src/cpu/shapes/mesh_intersect.h)

set(POLYTOPE_LIB_SOURCES
    lib/lodepng.cpp
    lib/lodepng.h
    src/cpu/constants.h
    src/cpu/cameras/abstract_camera.h
    src/cpu/cameras/camera_settings.h
    src/cpu/cameras/perspective_camera.cpp
    src/cpu/cameras/perspective_camera.h
    src/cpu/films/abstract_film.h
    src/cpu/films/png_film.cpp
    src/cpu/films/png_film.h
    src/cpu/filters/abstract_filter.h
    src/cpu/filters/box_filter.cpp
    src/cpu/filters/box_filter.h
    src/cpu/filters/triangle_filter.cpp
    src/cpu/filters/triangle_filter.h
    src/cpu/integrators/abstract_integrator.h
    src/cpu/integrators/abstract_integrator.h
    src/cpu/integrators/DebugIntegrator.cpp
    src/cpu/integrators/DebugIntegrator.h
    src/cpu/integrators/PathTraceIntegrator.cpp
    src/cpu/integrators/PathTraceIntegrator.h
    src/cpu/lights/AbstractLight.h
    src/cpu/lights/PointLight.h
    src/common/parsers/pbrt_parser.cpp
    src/common/parsers/pbrt_parser.h
    src/cpu/runners/runner.h
    src/cpu/runners/runner.cpp
    src/cpu/samplers/samplers.h
    src/cpu/samplers/samplers.cpp
    src/cpu/scenes/scene.cpp
    src/cpu/scenes/scene.h
    src/cpu/scenes/skyboxes/AbstractSkybox.h
    src/cpu/scenes/skyboxes/ColorSkybox.cpp
    src/cpu/scenes/skyboxes/ColorSkybox.h
    src/cpu/shading/Material.h
    src/cpu/shading/brdf/abstract_brdf.h
    src/cpu/shading/brdf/lambert_brdf.h
    src/cpu/shading/brdf/mirror_brdf.h
    src/cpu/shading/brdf/glossy_brdf.h
    src/cpu/shading/spectrum.h
    src/cpu/structures/intersection.h
    src/cpu/structures/matrix.cpp
    src/cpu/structures/matrix.h
    src/common/structures/point2.h
    src/cpu/structures/ray.h
    src/cpu/structures/Sample.h
    src/cpu/structures/transform.cpp
    src/cpu/structures/transform.h
    src/common/utilities/Common.h
    src/common/utilities/GeometryCalculations.h
    src/common/utilities/GlobalDefines.h
    src/common/utilities/Logger.cpp
    src/common/utilities/Logger.h
    src/common/utilities/Options.h
    src/common/utilities/OptionsParser.cpp
    src/common/utilities/OptionsParser.h
    src/common/parsers/abstract_file_parser.cpp
    src/common/parsers/abstract_file_parser.h
    src/cpu/structures/Vectors.cpp
    src/cpu/structures/Vectors.h
    src/cpu/structures/bounding_box.cpp
    src/cpu/structures/bounding_box.h
    src/common/parsers/mesh_parsers.h
    src/common/parsers/mesh_parsers.cpp
    src/cpu/shapes/tesselators.cpp
    src/cpu/shapes/tesselators.h
    src/cpu/shapes/mesh.cpp
    src/cpu/shapes/mesh.h
    src/cpu/shapes/mesh_intersect.h
    src/cpu/shapes/mesh_intersect.o
    src/cpu/acceleration/bvh.cpp
    src/cpu/acceleration/bvh.h
    src/cpu/structures/stats.h
    src/common/utilities/thread_pool.h
    src/common/utilities/thread_pool.cpp
    src/cpu/shading/texture.h
    )

add_executable(${PROJECT_NAME} src/cpu/main.cpp
    )

add_library(polytope_gl_lib SHARED ${POLYTOPE_LIB_SOURCES})
add_library(polytope_lib SHARED ${POLYTOPE_LIB_SOURCES})

target_compile_definitions(polytope_gl_lib PUBLIC POLYTOPEGL)

add_executable(polytope-gl
    src/gl/main.cpp
    src/gl/gl_renderer.cpp
    src/gl/gl_renderer.h
    src/gl/bb_vao.h
    src/gl/mesh_vao.h
    src/gl/quad_sphere/quad_sphere_vao.h
    src/gl/line/line.h
    src/gl/shaders.h
    src/gl/shaders.cpp src/gl/gl_recorder.h)

target_compile_definitions(polytope-gl PUBLIC POLYTOPEGL)

add_library(polytope_cuda_lib
    src/cuda/check_error.h
    src/cuda/context.cpp
    src/cuda/context.h
    src/cuda/kernels/common_device_functions.cuh
    src/cuda/kernels/path_tracer.cu
    src/cuda/kernels/path_tracer.cuh
    src/cuda/png_output.cpp
    src/cuda/png_output.h
    src/cuda/mesh/cuda_pinned_allocator.h
    src/cuda/mesh/cuda_mesh_soa.cpp
    src/cuda/mesh/cuda_mesh_soa.h
    )

add_executable(polytope_cuda
    src/cuda/main.cpp
    )

set_target_properties(
    polytope_cuda
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# Key idea: SEPARATE OUT your main() function into its own file so it can be its
# own executable. Separating out main() means you can add this library to be
# used elsewhere.

if (COVERAGE)
  target_compile_options(polytope_lib PRIVATE --coverage)
  target_link_libraries(polytope_lib PRIVATE --coverage)
endif ()

# link in pthreads on linux
if (UNIX)
  if (APPLE)
    find_package(glfw3 3.3 REQUIRED)
  else ()
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

    find_package(glfw3 3.2 REQUIRED)
  endif ()

  target_link_libraries(polytope-gl PRIVATE glfw)

  find_package(glbinding PATHS /usr/share/glbinding/ REQUIRED)
  target_link_libraries(polytope-gl PRIVATE glbinding::glbinding)

  # find_package(glm REQUIRED)
  # target_link_libraries(polytope PRIVATE glm)

  target_include_directories(
      polytope_cuda
      PRIVATE
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  target_link_libraries(
      polytope_cuda
      PRIVATE
      ${CUDART_LIBRARY}
  )

  target_include_directories(
      polytope_cuda_lib
      PRIVATE
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  target_link_libraries(
      polytope_cuda_lib
      PRIVATE
      ${CUDART_LIBRARY}
  )
endif ()

target_link_libraries(${PROJECT_NAME} PUBLIC polytope_lib)
target_link_libraries(polytope-gl PUBLIC polytope_gl_lib)
target_link_libraries(polytope_cuda PUBLIC polytope_lib)
target_link_libraries(polytope_cuda PUBLIC polytope_cuda_lib)

set_property(TARGET polytope_cuda PROPERTY CUDA_ARCHITECTURES 61)
set_property(TARGET polytope_cuda_lib PROPERTY CUDA_ARCHITECTURES 61)

################################
# Testing
################################
# if (tests)
if (APPLE)
  add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
  add_definitions(-D__GLIBCXX__)
endif (APPLE)

# This adds another subdirectory, which has 'project(gtest)'.
add_subdirectory(lib/googletest-1.10.0)

enable_testing()

# Include the gtest library. gtest_SOURCE_DIR is available due to
# 'project(gtest)' above.
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

##############
# Unit Tests
##############
add_executable(runUnitTests
    tests/TestBoundingBox.cpp
    tests/TestCommon.cpp
    tests/TestIntersectionTests.cpp
    tests/TestMatrix4x4.cpp
    tests/TestPoint.cpp
    tests/TestVector.cpp
    tests/TestTransform.cpp
    tests/test_bvh.cpp
    tests/parsers/test_obj_parser.cpp
    tests/parsers/pbrt/test_lex.cpp
    tests/parsers/pbrt/test_parse.cpp
    tests/parsers/pbrt/test_scan.cpp
    tests/parsers/pbrt/test_textures.cpp
    tests/shading/test_mirror_brdf.cpp
    tests/parsers/test_ply_parser.cpp
    tests/shapes/test_mesh.cpp
    tests/shapes/test_mesh_linear_soa.cpp
    tests/shapes/TestTesselators.cpp
    tests/cuda/test_bvh_traversal.cpp
    tests/thread_pool_tests.cpp
    tests/parsers/pbrt/test_instancing.cpp
    tests/triangle_filter_tests.cpp)

target_include_directories(
    runUnitTests
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# generate coverage data, see http://david-grs.github.io/cpp-clang-travis-cmake-gtest-coveralls-appveyor/
# if (COVERAGE)
#    target_compile_options(runUnitTests PRIVATE --coverage)
#    target_link_libraries(runUnitTests PRIVATE --coverage)
# endif()

# Standard linking to gtest stuff.
target_link_libraries(runUnitTests PRIVATE gtest gtest_main)

# Extra linking for the project.
target_link_libraries(runUnitTests PRIVATE polytope_lib polytope_cuda_lib)

# This is so you can do 'make tests' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
add_test(NAME that-test-I-made COMMAND runUnitTests)

# You can also omit NAME and COMMAND. The second argument could be some other
# tests executable.
add_test(that-other-test-I-made runUnitTests)
# endif()