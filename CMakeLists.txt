cmake_minimum_required(VERSION 3.10)
project(JanusLinearProject)
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


# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "Suppress developer warnings")
# Set debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
# Add DEBUG definition
#add_definitions(-DDEBUG)
# Include directories
include_directories(${CMAKE_SOURCE_DIR}/matplotlib-cpp)
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)



enable_testing()
# Find libtorch.  Modify this line to find your own libtorch installation.
set(Torch_DIR "/home/panos/Applications/libtorch/share/cmake/Torch")
# Find libtorch
find_package(Torch REQUIRED)

# Add pthread
find_package(Threads REQUIRED)

# Define the source directory for headers
set(HEADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/cpp")
set(HEADER_DEST_DIR "/usr/local/include/janus")

# Custom command to copy headers
add_custom_target(copy_headers ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HEADER_DEST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${HEADER_SOURCE_DIR} ${HEADER_DEST_DIR}
    COMMENT "Copying header files to ${HEADER_DEST_DIR}"
)

# Add subdirectory for tests
add_subdirectory(tests/cpp)
add_subdirectory(examples/cpp)

# Install rules for header files
install(DIRECTORY ${HEADER_SOURCE_DIR}/ DESTINATION include/janus FILES_MATCHING PATTERN "*.hpp")
