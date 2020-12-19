project(tomcat_tmm)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/tools/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")

cmake_policy(SET CMP0074 NEW)

# External libraries shared by src and test

# fmt is an open-source C++ formatting library
find_package(FMT REQUIRED)

# Eigen is a C++ library used for matrix and vector operations.
find_package(Eigen3 REQUIRED NO_MODULE)

# GSL is a C library used for sampling from a determined distribution.
find_package(GSL REQUIRED)

# Boost is used for several requirements in the project.
find_package(
        Boost 1.69
        COMPONENTS chrono
        date_time
        filesystem
        iostreams
        program_options
        regex
        system
        thread
        REQUIRED)

# Json library is being used for writing evaluations to an output stream
find_package(nlohmann_json REQUIRED)

# Finding the Mosquitto header and library
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
        "${CMAKE_SOURCE_DIR}/../../../tools/cmake")

find_package(Mosquitto 1.5 REQUIRED)

set(EXTERNAL_LIBS
        ${Boost_LIBRARIES}
        Eigen3::Eigen
        GSL::gsl
        GSL::gslcblas
        fmt::fmt
        nlohmann_json::nlohmann_json
        ${Mosquitto_LIBRARIES})

# Include these directories in all the targets of this project
include_directories(${Boost_INCLUDE_DIR} ${GSL_INCLUDE_DIRS})

# Todo make code coverage work on Linux as well.
if(APPLE
        AND CODE_COVERAGE
        AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Add required flags (GCC & LLVM/Clang)
    message("Building with code coverage")
    target_compile_options(
            tomcat -O0 # no optimization
            -g # generate debug info
            --coverage # sets all required flags
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
        target_link_options(tomcat PUBLIC --coverage)
    else()
        target_link_libraries(tomcat PUBLIC --coverage)
    endif()
endif()

add_subdirectory(src)
add_subdirectory(test)
