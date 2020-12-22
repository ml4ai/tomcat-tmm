set(TEST_LIBS
        tomcat_model_lib
        ${EXTERNAL_LIBS})

message(${TEST_LIBS})

add_executable(TomcatTest run_tests.cpp)

target_include_directories (TomcatTest PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(TomcatTest PRIVATE ${TEST_LIBS})

add_test(NAME all_tests COMMAND TomcatTest)