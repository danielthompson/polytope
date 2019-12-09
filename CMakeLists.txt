cmake_minimum_required(VERSION 3.7)

# Options. Turn on with 'cmake -Dmyvarname=ON'.
option(tests "Build all tests." OFF) # Makes boolean 'tests' available.

# Make PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR, and PROJECT_NAME available.
set(PROJECT_NAME polytope)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_CXX_FLAGS "-O0 -g -Wno-error=unused-function")

# If you want your own include/ directory, set this, and then you can do
# include_directories(${COMMON_INCLUDES}) in other CMakeLists.txt files.
# set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)

################################
# Normal Libraries & Executables
################################
add_library(polytope_lib
      src/structures/Point.cpp
      src/structures/Point.h
      src/Constants.h
      src/structures/Vector.cpp
      src/structures/Vector.h
      src/structures/Matrix4x4.h
      src/structures/Matrix4x4.cpp
      src/structures/Transform.h
      src/structures/Transform.cpp
      src/scenes/AbstractScene.cpp
      src/scenes/AbstractScene.h
      src/structures/Ray.cpp
      src/structures/Ray.h
      src/cameras/AbstractCamera.cpp
      src/cameras/AbstractCamera.h
      src/shapes/AbstractShape.cpp
      src/shapes/AbstractShape.h
      src/shapes/Sphere.cpp
      src/shapes/Sphere.h
      src/cameras/CameraSettings.h
      src/cameras/PerspectiveCamera.cpp
      src/cameras/PerspectiveCamera.h
      src/structures/Intersection.cpp
      src/structures/Intersection.h
      src/structures/Normal.cpp
      src/structures/Normal.h
      src/runners/AbstractRunner.h
      src/runners/AbstractRunner.cpp
      src/runners/PixelRunner.cpp
      src/runners/PixelRunner.h
      src/samplers/AbstractSampler.h
      src/structures/Point2.cpp
      src/structures/Point2.h
      src/samplers/CenterSampler.cpp
      src/samplers/CenterSampler.h
      src/scenes/NaiveScene.cpp
      src/scenes/NaiveScene.h
      src/scenes/SceneBuilder.cpp
      src/scenes/SceneBuilder.h
      src/shading/Material.h
      src/shading/brdf/AbstractBRDF.h
      src/shading/brdf/MirrorBRDF.cpp
      src/shading/brdf/MirrorBRDF.h
      src/shading/SpectralPowerDistribution.cpp
      src/shading/SpectralPowerDistribution.h
      src/shading/ReflectanceSpectrum.cpp
      src/shading/ReflectanceSpectrum.h
      src/lights/AbstractLight.h
      src/lights/PointLight.h
      src/integrators/AbstractIntegrator.cpp
      src/integrators/AbstractIntegrator.h
      src/structures/Sample.cpp
      src/structures/Sample.h
      src/integrators/PathTraceIntegrator.cpp
      src/integrators/PathTraceIntegrator.h
      lib/lodepng.cpp
      lib/lodepng.h
      src/films/AbstractFilm.h
      src/films/PNGFilm.cpp
      src/films/PNGFilm.h
      src/integrators/DebugIntegrator.cpp
      src/integrators/DebugIntegrator.h
      src/utilities/GeometryCalculations.h
      src/samplers/GridSampler.cpp
      src/samplers/GridSampler.h
      src/filters/AbstractFilter.cpp
      src/filters/AbstractFilter.h
      src/filters/BoxFilter.cpp
      src/filters/BoxFilter.h
      src/runners/TileRunner.cpp
      src/runners/TileRunner.h
      src/utilities/Logger.cpp
      src/utilities/Logger.h
      src/Tracer.cpp src/Tracer.h
      src/utilities/OptionsParser.cpp
      src/utilities/OptionsParser.h
      src/utilities/Options.h
      src/shading/brdf/LambertBRDF.cpp
      src/shading/brdf/LambertBRDF.h
      src/shading/brdf/AbstractBRDF.cpp
      src/samplers/HaltonSampler.cpp
      src/samplers/HaltonSampler.h
      src/parsers/PBRTFileParser.cpp
      src/parsers/PBRTFileParser.h
      src/lights/ShapeLight.h
      src/utilities/Common.h)
add_executable(${PROJECT_NAME} src/Main.cpp)
# Key idea: SEPARATE OUT your main() function into its own file so it can be its
# own executable. Separating out main() means you can add this library to be
# used elsewhere.

# link in pthreads on linux
if (UNIX AND NOT APPLE)
   set(THREADS_PREFER_PTHREAD_FLAG ON)
   find_package(Threads REQUIRED)
   target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

target_link_libraries(${PROJECT_NAME} polytope_lib)

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
      tests/Matrix4x4.cpp
      tests/Point.cpp
      tests/Transform.cpp
      tests/MirrorBRDF.cpp
      tests/TestFileParser.cpp)

# Standard linking to gtest stuff.
target_link_libraries(runUnitTests gtest gtest_main)

# Extra linking for the project.
target_link_libraries(runUnitTests polytope_lib)

# This is so you can do 'make tests' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
add_test(NAME that-test-I-made COMMAND runUnitTests)

# You can also omit NAME and COMMAND. The second argument could be some other
# tests executable.
add_test(that-other-test-I-made runUnitTests)
# endif()