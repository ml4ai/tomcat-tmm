# Model sources
file(GLOB_RECURSE tomcat_model_sources CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Model library
add_library(tomcat_model_lib STATIC ${tomcat_model_sources})
target_include_directories(tomcat_model_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set(ALL_LIBS tomcat_model_lib ${EXTERNAL_LIBS})

# Target for online ToMCATß
add_executable(TomcatModelOnline run_online_tomcat.cpp)
target_include_directories(TomcatModelOnline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TomcatModelOnline PRIVATE ${ALL_LIBS})

# Target for experiments with model v1.0
add_executable(TomcatModelV1 run_experiments_v1.cpp)
target_include_directories(TomcatModelV1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TomcatModelV1 PRIVATE ${ALL_LIBS})

# Target for experiments with model v2.0
add_executable(TomcatModelV2 run_experiments_v2.cpp)
target_include_directories(TomcatModelV2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TomcatModelV2 PRIVATE ${ALL_LIBS})

# Target for data conversion
add_executable(TomcatConverter convert_messages.cpp)
target_include_directories(TomcatConverter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TomcatConverter PRIVATE ${ALL_LIBS})

# Targets for experiments under the ASIST evaluation reports
add_executable(TomcatASISTFall20 asist/run_experiments_fall20.cpp)
target_include_directories(TomcatASISTFall20 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TomcatASISTFall20 PRIVATE ${ALL_LIBS})

option(CODE_COVERAGE "Enable coverage reporting" OFF)

# Todo make code coverage work on Linux as well.
if(APPLE AND CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Add required flags (GCC & LLVM/Clang)
    message("Building with code coverage")
    target_compile_options(
            runMission
            -O0 # no optimization
            -g # generate debug info
            --coverage # sets all required flags
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
        target_link_options(TomcatModelV2 PUBLIC --coverage)
    else()
        target_link_libraries(TomcatModelV2 PUBLIC --coverage)
    endif()
endif()