cmake_minimum_required(VERSION 3.10)
project(JanusLinearProject)

# Option to build as header-only
option(HEADER_ONLY "Build as header-only library" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Find Python3 and NumPy
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Include directories
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

# Set the global runtime output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set flags for debug and release
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/matplotlib-cpp)
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)

enable_testing()

# Find libtorch. Modify this line to find your own libtorch installation.
set(Torch_DIR "/home/panos/Applications/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Add pthread
find_package(Threads REQUIRED)

# Define the source directory for headers
set(HEADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/cpp")
set(HEADER_DEST_DIR "/usr/local/include/janus")

# Add libraries if not header-only
if(NOT HEADER_ONLY)
    add_library(radauted SHARED src/cpp/radauted.cpp)
    target_include_directories(radauted PUBLIC ${CMAKE_SOURCE_DIR}/src/cpp)
    target_link_libraries(radauted PRIVATE ${TORCH_LIBRARIES})

    add_library(radaute SHARED src/cpp/radaute.cpp)
    target_include_directories(radaute PUBLIC ${CMAKE_SOURCE_DIR}/src/cpp)
    target_link_libraries(radaute PRIVATE ${TORCH_LIBRARIES})
endif()

# Add subdirectory for tests and examples
add_subdirectory(tests/cpp)
add_subdirectory(examples/cpp)

# Handle header-only mode
if(HEADER_ONLY)
    add_definitions(-DHEADER_ONLY)
    # Pass -DHEADER_ONLY to the compiler as an argument
    add_compile_definitions(HEADER_ONLY)
    # Install headers only
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include/janus FILES_MATCHING PATTERN "*.hpp")
else()
    # Install libraries and headers
    install(TARGETS radauted radaute ARCHIVE DESTINATION lib)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include/janus FILES_MATCHING PATTERN "*.hpp")
endif()
