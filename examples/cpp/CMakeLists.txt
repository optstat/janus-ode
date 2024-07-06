cmake_minimum_required(VERSION 3.10)

project(VanDerPolExample)

# Include directories from parent
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/examples/cpp)

# Find libtorch
find_package(Torch REQUIRED)




# Specify the executable
set(SOURCE_FILE radaute_hamiltonian_example.cpp)

add_executable(vdpexample ${SOURCE_FILE})
# Link libraries
target_link_libraries(vdpexample "${TORCH_LIBRARIES}")
target_link_libraries(vdpexample Python3::Python Python3::NumPy)


# Ensure the executable is built when running 'make run_vdpexample'
add_custom_target(run_vdpexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdpexample
)