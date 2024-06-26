# Find libtorch
find_package(Torch REQUIRED)

# Specify the executable
set(SOURCE_FILE linear_solver_test.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(lineartest ${SOURCE_FILE})

# Include GTest
find_package(GTest REQUIRED)
include_directories(src/cpp ${GTEST_INCLUDE_DIRS})

# Link GTest
target_link_libraries(lineartest ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

# Link libtorch
target_link_libraries(lineartest "${TORCH_LIBRARIES}")
set_property(TARGET lineartest PROPERTY CXX_STANDARD 17)
add_test(NAME LinearTest COMMAND ${CMAKE_BINARY_DIR}/bin/lineartest)

# Ensure the test executable is built when running 'make test'
add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target lineartest
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
)


