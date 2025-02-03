cmake_minimum_required(VERSION 3.10)
project(JanusLinearProject)

# Option to build as header-only
option(HEADER_ONLY "Build as header-only library" ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


#Set the build type to Release if not specified
set(CMAKE_BUILD_TYPE Debug  )

# Find Python3 and NumPy
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
# Find GTest
find_package(GTest REQUIRED)
# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/matplotlib-cpp)
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)

# Set the global runtime output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set flags for debug and release
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Find libtorch. Modify this line to find your own libtorch installation.
set(Torch_DIR "/home/panos/Applications/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Add pthread
find_package(Threads REQUIRED)

# Include the Torch include directories
include_directories(${TORCH_INCLUDE_DIRS})

# Define the source directory for headers
set(HEADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/cpp")
set(HEADER_DEST_DIR "/usr/local/include/janus")

# Install headers to the system include directory
install(DIRECTORY ${HEADER_SOURCE_DIR}/ DESTINATION ${HEADER_DEST_DIR}/ FILES_MATCHING PATTERN "*.hpp")

# Add subdirectory for tests (if you have any test cases)
# You can create a tests/ directory and add it here
add_subdirectory(examples/cpp)
add_subdirectory(tests/cpp)
