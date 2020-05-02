cmake_minimum_required(VERSION 3.7)

# Options. Turn on with 'cmake -Dmyvarname=ON'.
option(tests "Build all tests." OFF) # Makes boolean 'tests' available.

# Make PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR, and PROJECT_NAME available.
set(PROJECT_NAME polytope)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wno-error=unused-function -pthread")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wno-error=unused-function -pthread")

set(ispc_options -g --pic)

IF(CMAKE_BUILD_TYPE MATCHES RELEASE)
   set(ispc_options --pic -O3)
ENDIF(CMAKE_BUILD_TYPE MATCHES RELEASE)

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
        OUTPUT ${CMAKE_SOURCE_DIR}/src/shapes/triangle_mesh_ispc.o
        COMMAND ispc ${ispc_options} -h ${CMAKE_SOURCE_DIR}/src/shapes/triangle_mesh_ispc.h -o ${CMAKE_SOURCE_DIR}/src/shapes/triangle_mesh_ispc.o ${CMAKE_SOURCE_DIR}/src/shapes/triangle_mesh.ispc
        MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/src/shapes/triangle_mesh.ispc
        BYPRODUCTS ${CMAKE_SOURCE_DIR}/src/shapes/triangle_mesh_ispc.h
        DEPENDS ${CMAKE_SOURCE_DIR}/src/shapes/triangle_mesh.ispc)

add_library(polytope_lib
      lib/lodepng.cpp
      lib/lodepng.h
      src/Constants.h
      src/cameras/AbstractCamera.cpp
      src/cameras/AbstractCamera.h
      src/cameras/CameraSettings.h
      src/cameras/PerspectiveCamera.cpp
      src/cameras/PerspectiveCamera.h
      src/films/AbstractFilm.h
      src/films/PNGFilm.cpp
      src/films/PNGFilm.h
      src/filters/AbstractFilter.h
      src/filters/BoxFilter.cpp
      src/filters/BoxFilter.h
      src/integrators/AbstractIntegrator.h
      src/integrators/AbstractIntegrator.h
      src/integrators/DebugIntegrator.cpp
      src/integrators/DebugIntegrator.h
      src/integrators/PathTraceIntegrator.cpp
      src/integrators/PathTraceIntegrator.h
      src/lights/AbstractLight.h
      src/lights/PointLight.h
      src/lights/ShapeLight.h
      src/parsers/PBRTFileParser.cpp
      src/parsers/PBRTFileParser.h
      src/runners/AbstractRunner.cpp
      src/runners/AbstractRunner.h
      src/runners/PixelRunner.cpp
      src/runners/PixelRunner.h
      src/runners/TileRunner.cpp
      src/runners/TileRunner.h
        src/samplers/samplers.h
        src/samplers/samplers.cpp
        src/scenes/AbstractScene.cpp
      src/scenes/AbstractScene.h
      src/scenes/BVHScene.h
      src/scenes/NaiveScene.cpp
      src/scenes/NaiveScene.h
      src/scenes/SceneBuilder.cpp
      src/scenes/SceneBuilder.h
      src/scenes/skyboxes/AbstractSkybox.h
      src/scenes/skyboxes/ColorSkybox.cpp
      src/scenes/skyboxes/ColorSkybox.h
      src/shading/Material.h
      src/shading/ReflectanceSpectrum.cpp
      src/shading/ReflectanceSpectrum.h
      src/shading/SpectralPowerDistribution.cpp
      src/shading/SpectralPowerDistribution.h
      src/shading/brdf/AbstractBRDF.cpp
      src/shading/brdf/AbstractBRDF.h
      src/shading/brdf/LambertBRDF.cpp
      src/shading/brdf/LambertBRDF.h
      src/shading/brdf/MirrorBRDF.cpp
      src/shading/brdf/MirrorBRDF.h
      src/shapes/AbstractShape.cpp
      src/shapes/AbstractShape.h
      src/structures/Intersection.cpp
      src/structures/Intersection.h
      src/structures/Matrix4x4.cpp
      src/structures/Matrix4x4.h
      src/structures/Point2.cpp
      src/structures/Point2.h
      src/structures/Ray.cpp
      src/structures/Ray.h
      src/structures/Sample.cpp
      src/structures/Sample.h
      src/structures/Transform.cpp
      src/structures/Transform.h
      src/utilities/Common.h
      src/utilities/GeometryCalculations.h
      src/utilities/GlobalDefines.h
      src/utilities/Logger.cpp
      src/utilities/Logger.h
      src/utilities/Options.h
      src/utilities/OptionsParser.cpp
      src/utilities/OptionsParser.h
      src/parsers/mesh/OBJParser.cpp
      src/parsers/mesh/OBJParser.h
        src/shapes/triangle.cpp
        src/shapes/triangle.h 
      src/shapes/triangle_mesh_ispc.h 
      src/shapes/triangle_mesh_ispc.o
      src/parsers/AbstractFileParser.cpp
      src/parsers/AbstractFileParser.h
      src/structures/Vectors.cpp
      src/structures/Vectors.h
      src/structures/BoundingBox.cpp
      src/structures/BoundingBox.h
      src/parsers/mesh/PLYParser.cpp
      src/parsers/mesh/PLYParser.h
      src/parsers/mesh/AbstractMeshParser.h
      src/shapes/Tesselators.cpp
      src/shapes/Tesselators.h
      src/exporters/OBJExporter.cpp
      src/exporters/OBJExporter.h
      src/scenes/BVHScene.cpp)

set_source_files_properties(~/repos/polytope/src/shapes/triangle_mesh_ispc.h PROPERTIES OBJECT_DEPENDS ~/repos/polytope/src/shapes/triangle_mesh.ispc)

#target_link_libraries(polytope_lib PRIVATE ~/repos/polytope/src/shapes/triangle_mesh_ispc.o)

add_executable(${PROJECT_NAME} src/Main.cpp
      src/platforms/win32-cygwin.h
      src/gl/GLRenderer.cpp
      src/gl/GLRenderer.h)
# Key idea: SEPARATE OUT your main() function into its own file so it can be its
# own executable. Separating out main() means you can add this library to be
# used elsewhere.

if (COVERAGE)
   target_compile_options(polytope_lib PRIVATE --coverage)
   target_link_libraries(polytope_lib PRIVATE --coverage)
endif()

# link in pthreads on linux
if (UNIX)
   if (APPLE)
      find_package(glfw3 3.3 REQUIRED)
   else()
      set(THREADS_PREFER_PTHREAD_FLAG ON)
      find_package(Threads REQUIRED)
      target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

      find_package(glfw3 3.2 REQUIRED)
   endif()

   target_link_libraries(polytope PRIVATE glfw)

   find_package(glbinding REQUIRED)
   target_link_libraries(polytope PRIVATE glbinding::glbinding)

   find_package(glm REQUIRED)
   target_link_libraries(polytope PRIVATE glm)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC polytope_lib)

################################
# Testing
################################
# if (tests)
if (APPLE)
   add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
   add_definitions(-D__GLIBCXX__)
endif (APPLE)

# This adds another subdirectory, which has 'project(gtest)'.
add_subdirectory(lib/googletest-1.7.0)

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
      tests/TestMirrorBRDF.cpp
        tests/parsers/test_obj_parser.cpp
        tests/parsers/pbrt/test_scan.cpp
        tests/parsers/pbrt/test_parse.cpp
        tests/parsers/pbrt/test_lex.cpp
        tests/parsers/test_ply_parser.cpp
      tests/TestPoint.cpp
      tests/TestTransform.cpp
      tests/TestTriangleMesh.cpp
      tests/TestVector.cpp
      tests/TestTesselators.cpp
      tests/TestTriangleSplitting.cpp
        tests/TestTriangleMeshISPC.cpp)

# generate coverage data, see http://david-grs.github.io/cpp-clang-travis-cmake-gtest-coveralls-appveyor/
# if (COVERAGE)
#    target_compile_options(runUnitTests PRIVATE --coverage)
#    target_link_libraries(runUnitTests PRIVATE --coverage)
# endif()

# Standard linking to gtest stuff.
target_link_libraries(runUnitTests PRIVATE gtest gtest_main)

# Extra linking for the project.
target_link_libraries(runUnitTests PRIVATE polytope_lib)

# This is so you can do 'make tests' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
add_test(NAME that-test-I-made COMMAND runUnitTests)

# You can also omit NAME and COMMAND. The second argument could be some other
# tests executable.
add_test(that-other-test-I-made runUnitTests)
# endif()