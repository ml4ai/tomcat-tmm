#define BOOST_TEST_MODULE TomcatModelTest

#include "boost/test/included/unit_test.hpp"
#include "pgm/ConstantNode.h"

using namespace tomcat::model;
using namespace std;

BOOST_AUTO_TEST_SUITE(node_test)

BOOST_AUTO_TEST_CASE(constant_node) {
    ConstantNode node(1);
    BOOST_TEST(node.get_assignment()(0, 0) == 1);
}

BOOST_AUTO_TEST_CASE(constant_node2) {
    ConstantNode node(2);
    BOOST_TEST(node.get_assignment()(0, 0) == 2);
}

BOOST_AUTO_TEST_SUITE_END()
