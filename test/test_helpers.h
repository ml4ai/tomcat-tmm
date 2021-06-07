#pragma once

#include <string>

#include "eigen3/Eigen/Dense"
#include <boost/filesystem.hpp>

#include "utils/Definitions.h"
#include "utils/Tensor3.h"

struct Fixture {
    Fixture()   { boost::filesystem::current_path("../../test"); }
    ~Fixture()  { /* Run on tear down */ }
};

bool is_equal(const Eigen::MatrixXd& m1,
              const Eigen::MatrixXd& m2,
              double tolerance = 0.00001) {
    /**
     * This function checks if the elements of two matrices are equal within a
     * tolerance value.
     */

    for (int i = 0; i < m1.rows(); i++) {
        for (int j = 0; j < m1.cols(); j++) {
            if (abs(m1(i, j) - m2(i, j)) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

std::string get_matrix_check_msg(const Eigen::MatrixXd& estimated,
                            const Eigen::MatrixXd& expected) {
    std::stringstream msg;
    msg << "Estimated: [" << estimated << "]; Expected: [" << expected << "]";

    return msg.str();
}

std::pair<bool, std::string> check_matrix_eq(const Eigen::MatrixXd& estimated,
                                   const Eigen::MatrixXd& expected,
                                   double tolerance = 0.00001) {

    std::string msg = get_matrix_check_msg(estimated, expected);
    bool equal = is_equal(estimated, expected, tolerance);

    return make_pair(equal, msg);
}

bool check_tensor_eq(tomcat::model::Tensor3& estimated,
                     tomcat::model::Tensor3& expected,
                     double tolerance = 0.00001) {
    return estimated.equals(expected, tolerance);
}

Eigen::MatrixXd
get_cpd_table(const tomcat::model::DBNPtr& oracle, std::string node_label, bool prior) {
    const auto& metadata = oracle->get_metadata_of(node_label);
    int time_step;

    if (!metadata->is_replicable() || prior) {
        time_step = metadata->get_initial_time_step();
    }
    else {
        time_step = metadata->get_initial_time_step() + 1;
    }

    return oracle->get_node(node_label, time_step)->get_cpd()->get_table(0);
}