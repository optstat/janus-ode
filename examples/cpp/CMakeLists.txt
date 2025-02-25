cmake_minimum_required(VERSION 3.10)

project(OdeExamples)
#
# --- (1) Option to create/use a Python virtual environment
#
option(CREATE_VENV "Create and use a Python virtual environment for building" OFF)

#
# --- (2) If requested, create a venv in the build directory and install PyTorch
#
if(CREATE_VENV)
    message(STATUS "Creating Python virtual environment in: ${CMAKE_BINARY_DIR}/venv")

    # Create the venv (adjust 'python3' to 'python' or your interpreter if needed)
    execute_process(
        COMMAND python3 -m venv venv
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    message(STATUS "Upgrading pip in the new environment ...")
    execute_process(
        COMMAND ${CMAKE_BINARY_DIR}/venv/bin/python -m pip install --upgrade pip
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    message(STATUS "Installing PyTorch wheel into the new environment ...")
    # You can specify a version, CUDA vs. CPU, etc. e.g. "torch==2.0.0"
    execute_process(
        COMMAND ${CMAKE_BINARY_DIR}/venv/bin/pip install torch
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    #
    # Point CMake's find_package(Python3) to use the venv interpreter
    #
    set(Python3_EXECUTABLE
        "${CMAKE_BINARY_DIR}/venv/bin/python"
        CACHE FILEPATH "Python interpreter within the newly created venv"
        FORCE
    )

    #
    # If you want CMake to find Torch via the pip-installed package, you usually
    # need to set Torch_DIR to the folder containing TorchConfig.cmake.
    # Typical location on Linux with Python 3.9 might be:
    #   build/venv/lib/python3.9/site-packages/torch/share/cmake/Torch
    #
    # Adjust python version (3.8, 3.10, etc.) as needed:
    #
    # set(Torch_DIR
    #     "${CMAKE_BINARY_DIR}/venv/lib/python3.9/site-packages/torch/share/cmake/Torch"
    #     CACHE PATH "Path to TorchConfig.cmake inside the virtualenv" FORCE
    # )
endif()


# Include directories from parent
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/examples/cpp)

# Find libtorch
find_package(Torch REQUIRED)

#Add Petc to the project.  CHANGE THIS TO YOUR PETSC DIRECTORY
include_directories("/opt/petsc/include")
link_directories("/opt/petsc/lib")

#Add Sundials to the project.  CHANGE THIS TO YOUR SUNDIALS DIRECTORY
include_directories("/opt/sundials-7.2.1/include")
link_directories("/opt/sundials-7.2.1/lib")



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
set(SOURCE_FILE8 radauted_vdp_aug_example.cpp)
set(SOURCE_FILE9 radaut_example.cpp)
set(SOURCE_FILE10 for_sens_vdp_example.cpp)
set(SOURCE_FILE11 for_sens_u2d_act_example.cpp)

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

add_executable(radauted_vdp_aug_example ${SOURCE_FILE8})
# Link libraries for the second executable
target_link_libraries(radauted_vdp_aug_example "${TORCH_LIBRARIES}")
target_link_libraries(radauted_vdp_aug_example Python3::Python Python3::NumPy)

add_executable(radaut_example ${SOURCE_FILE9})
# Link libraries for the second executable
target_link_libraries(radaut_example "${TORCH_LIBRARIES}")
target_link_libraries(radaut_example Python3::Python Python3::NumPy)

add_executable(for_sens_vdp_example ${SOURCE_FILE10})
# Link to cvodes
#-lsundials_sunmatrixdense  -lsundials_sunlinsoldense
target_link_libraries(for_sens_vdp_example sundials_cvodes sundials_nvecserial sundials_sunmatrixdense sundials_sunlinsoldense sundials_core) 

add_executable(for_sens_u2d_act_example ${SOURCE_FILE11})
# Link to cvodes
#-lsundials_sunmatrixdense  -lsundials_sunlinsoldense   
target_link_libraries(for_sens_u2d_act_example "${TORCH_LIBRARIES}")
target_link_libraries(for_sens_u2d_act_example sundials_cvodes sundials_nvecserial sundials_sunmatrixdense sundials_sunlinsoldense sundials_core)

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

add_custom_target(create_radauted_vdp_aug_example
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target radauted_vdp_aug_example
)

add_custom_target(create_radaut_example
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target radaut_example
)


add_custom_target(create_for_sens_vdp_example
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target for_sens_vdp_example
)

add_custom_target(create_for_sens_u2d_act_example
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target for_sens_u2d_act_example
)