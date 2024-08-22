cmake_minimum_required(VERSION 3.10)

project(VanDerPolExample)

# Include directories from parent
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/examples/cpp)

# Find libtorch
find_package(Torch REQUIRED)

# Set compiler flags based on the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")





# Specify the executable
set(SOURCE_FILE1 radaute_hamiltonian_example.cpp)
set(SOURCE_FILE2 radaute_example.cpp)
set(SOURCE_FILE3 radauted_example.cpp)
set(SOURCE_FILE4 safe_gradients_example.cpp)
set(SOURCE_FILE5 radauted_vdp_norm_example.cpp)
set(SOURCE_FILE6 radaute_example_reverse.cpp)
set(SOURCE_FILE7 radauted_example_reverse.cpp)

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

add_executable(vdpnormexample ${SOURCE_FILE5})
# Link libraries for the second executable
target_link_libraries(vdpnormexample "${TORCH_LIBRARIES}")
target_link_libraries(vdpnormexample Python3::Python Python3::NumPy)

add_executable(vdp_example_reverse ${SOURCE_FILE6})
# Link libraries for the second executable
target_link_libraries(vdp_example_reverse "${TORCH_LIBRARIES}")
target_link_libraries(vdp_example_reverse Python3::Python Python3::NumPy)

add_executable(vdp_example_dual_reverse ${SOURCE_FILE7})
# Link libraries for the second executable
target_link_libraries(vdp_example_dual_reverse "${TORCH_LIBRARIES}")
target_link_libraries(vdp_example_dual_reverse Python3::Python Python3::NumPy)


# Ensure the executable is built when running 'make vdphamexample'
add_custom_target(create_vdphamexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdphamexample
)

# Ensure the executable is built when running 'make vdpexample'
add_custom_target(create_vdpexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdpexample
)

# Ensure the executable is built when running 'make vdpdexample'
add_custom_target(create_vdpdexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdpdexample
)

# Ensure the executable is built when running 'make safegradexample'
add_custom_target(create_safegradexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target safegradexample
)

add_custom_target(create_vdpnormexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdpnormexample
)

add_custom_target(create_vdpexample_reverse
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdp_example_reverse
)

add_custom_target(create_vdpexample_dual_reverse
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdp_example_dual_reverse
)