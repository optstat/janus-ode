cmake_minimum_required(VERSION 3.10)

project(VanDerPolExample)

# Find libtorch
find_package(Torch REQUIRED)

# Specify the executable
set(SOURCE_FILE radaute_hamiltonian_example.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_executable(vdpexample ${SOURCE_FILE})

# Link libtorch
target_link_libraries(vdpexample "${TORCH_LIBRARIES}")

# Ensure the executable is built when running 'make run_vdpexample'
add_custom_target(run_vdpexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdpexample
)