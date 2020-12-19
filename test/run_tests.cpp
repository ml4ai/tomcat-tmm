#include "gtest/gtest.h"
#include "pgm/ConstantNode.h"

using namespace tomcat::model;
using namespace std;

TEST(DistributionTest, SamplingFromDirichlet) {
    ConstantNode node(1);
    ASSERT_EQ(node.get_assignment()(0, 0), 1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
