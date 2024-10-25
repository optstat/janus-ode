# Inherit include directories from the parent
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/matplotlib-cpp)

# Find libtorch
find_package(Torch REQUIRED)

# Specify the executable for the tests
set(SOURCE_FILE ode_solver_test.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(odetest ${SOURCE_FILE})

# Include GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Link GTest and Python
# Link libraries
# Instead of ${GTEST_LIBRARIES} and ${GTEST_MAIN_LIBRARIES}, use GTest::GTest and GTest::Main
target_link_libraries(odetest PRIVATE ${GTEST_BOTH_LIBRARIES} "${TORCH_LIBRARIES}" Python3::Python Python3::NumPy pthread)
set_property(TARGET odetest PROPERTY CXX_STANDARD 17)

# Set debug flags if CMAKE_BUILD_TYPE is Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug mode: Adding debugging flags")
    set_target_properties(odetest PROPERTIES COMPILE_FLAGS "-g")
endif()
# Add the test executable to CTest
enable_testing()  # Ensure testing is enabled
add_test(NAME OdeSolverTest COMMAND odetest)

# Ensure the test executable is built when running 'make test'
add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target odetest
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
)
