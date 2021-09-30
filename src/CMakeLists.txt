# Model sources
file(GLOB_RECURSE tomcat_model_sources CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Model library
add_library(tomcat_model_lib STATIC ${tomcat_model_sources})
target_include_directories(tomcat_model_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIR} ${GSL_INCLUDE_DIRS})

set(ALL_LIBS tomcat_model_lib ${EXTERNAL_LIBS})

# Target for model training and evaluation
add_executable(train training.cpp)
target_include_directories(train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(train PRIVATE ${ALL_LIBS})

add_executable(evaluate evaluation.cpp)
target_include_directories(evaluate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(evaluate PRIVATE ${ALL_LIBS})

add_executable(generate data_generation.cpp)
target_include_directories(generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(generate PRIVATE ${ALL_LIBS})

add_executable(convert message_conversion.cpp)
target_include_directories(convert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(convert PRIVATE ${ALL_LIBS})

add_executable(start_agent online_estimation.cpp)
target_include_directories(start_agent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(start_agent PRIVATE ${ALL_LIBS})

add_executable(view model_visulization.cpp)
target_include_directories(view PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(view PRIVATE ${ALL_LIBS})

add_executable(convert_study2_hsr asist_study-2_2021.06.cpp)
target_include_directories(convert_study2_hsr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(convert_study2_hsr PRIVATE ${ALL_LIBS})

add_executable(random_gen random_gen.cpp)
target_include_directories(random_gen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(random_gen PRIVATE ${ALL_LIBS})

add_executable(multithread_check multithread_check.cpp)
target_include_directories(multithread_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(multithread_check PRIVATE ${ALL_LIBS})

add_executable(merge evidence_merge.cpp)
target_include_directories(merge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(merge PRIVATE ${ALL_LIBS})

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
