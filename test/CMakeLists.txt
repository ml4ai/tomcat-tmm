set(TEST_LIBS
        tomcat_model_lib
        ${EXTERNAL_LIBS})

message(${TEST_LIBS})

add_executable(TomcatTest run_tests.cpp)
add_executable(test_graph test_graph.cpp gen_graph.h)


target_include_directories (TomcatTest PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(TomcatTest PRIVATE ${TEST_LIBS})

target_include_directories (test_graph PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_graph PRIVATE Boost::graph)

add_test(NAME all_tests COMMAND TomcatTest)
