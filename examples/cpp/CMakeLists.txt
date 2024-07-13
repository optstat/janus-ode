cmake_minimum_required(VERSION 3.10)

project(VanDerPolExample)

# Include directories from parent
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/examples/cpp)

# Find libtorch
find_package(Torch REQUIRED)




# Specify the executable
set(SOURCE_FILE1 radaute_hamiltonian_example.cpp)
set(SOURCE_FILE2 radaute_example.cpp)
set(SOURCE_FILE3 radauted_example.cpp)
set(SOURCE_FILE4 safe_gradients_example.cpp)

add_executable(vdphamexample ${SOURCE_FILE1})
# Link libraries
target_link_libraries(vdphamexample "${TORCH_LIBRARIES}")
target_link_libraries(vdphamexample Python3::Python Python3::NumPy)

add_executable(vdpexample ${SOURCE_FILE2})
# Link libraries for the second executable
target_link_libraries(vdpexample "${TORCH_LIBRARIES}")
target_link_libraries(vdpexample Python3::Python Python3::NumPy)

add_executable(vdpdexample ${SOURCE_FILE3})
# Link libraries for the second executable
target_link_libraries(vdpdexample "${TORCH_LIBRARIES}")
target_link_libraries(vdpdexample Python3::Python Python3::NumPy)


add_executable(safegradexample ${SOURCE_FILE4})
# Link libraries for the second executable
target_link_libraries(safegradexample "${TORCH_LIBRARIES}")
target_link_libraries(safegradexample Python3::Python Python3::NumPy)

# Ensure the executable is built when running 'make vdphamexample'
add_custom_target(run_vdphamexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdphamexample
)

# Ensure the executable is built when running 'make vdpexample'
add_custom_target(run_vdpexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdpexample
)

# Ensure the executable is built when running 'make vdpdexample'
add_custom_target(run_vdpdexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdpdexample
)

# Ensure the executable is built when running 'make safegradexample'
add_custom_target(run_safegradexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target safegradexample
)