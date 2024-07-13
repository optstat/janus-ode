# Inherit include directories from the parent
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/matplotlib-cpp)


# Find libtorch
find_package(Torch REQUIRED)

# Specify the executable
set(SOURCE_FILE ode_solver_test.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(odetest ${SOURCE_FILE})

# Include GTest
find_package(GTest REQUIRED)
include_directories(src/cpp ${GTEST_INCLUDE_DIRS})

# Link GTest
target_link_libraries(odetest ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
target_link_libraries(odetest Python3::Python Python3::NumPy)


# Link libtorch
target_link_libraries(odetest "${TORCH_LIBRARIES}")
set_property(TARGET odetest PROPERTY CXX_STANDARD 17)

# Set C++ standard
set_property(TARGET odetest PROPERTY CXX_STANDARD 17)


# Ensure the test executable is built when running 'make test'
add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target lineartest
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
)


