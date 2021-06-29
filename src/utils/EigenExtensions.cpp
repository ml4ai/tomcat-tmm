#include "EigenExtensions.h"

#include <sstream>

namespace tomcat {
    namespace model {

        using namespace std;

        Eigen::MatrixXd cum_sum(const Eigen::MatrixXd& matrix, int axis) {
            Eigen::MatrixXd new_matrix(matrix.rows(), matrix.cols());

            switch (axis) {
            case 0: {
                Eigen::VectorXd cum_sum = Eigen::VectorXd::Zero(matrix.cols());
                for (int i = 0; i < matrix.rows(); i++) {
                    new_matrix.row(i) = matrix.row(i) + cum_sum;
                    cum_sum = new_matrix.row(i);
                }
                break;
            }
            case 1: {
                Eigen::VectorXd cum_sum = Eigen::VectorXd::Zero(matrix.rows());
                for (int j = 0; j < matrix.cols(); j++) {
                    new_matrix.col(j) = matrix.col(j) + cum_sum;
                    cum_sum = new_matrix.col(j);
                }
                break;
            }
            default: {
                throw invalid_argument(
                    "Invalid axis. Valid axes are 0, 1 or 2.");
            }
            }

            return new_matrix;
        }

        Eigen::MatrixXd mean(const vector<Eigen::MatrixXd>& matrices) {
            Eigen::MatrixXd mean_matrix;

            if (!matrices.empty()) {
                mean_matrix = matrices[0];

                for (int i = 1; i < matrices.size(); i++) {
                    mean_matrix =
                        (mean_matrix.array() + matrices[i].array()).matrix();
                }

                mean_matrix = (mean_matrix.array() / matrices.size()).matrix();
            }

            return mean_matrix;
        }

        Eigen::MatrixXd
        standard_error(const vector<Eigen::MatrixXd>& matrices) {
            Eigen::MatrixXd mean_matrix = mean(matrices);
            Eigen::MatrixXd se_matrix =
                (matrices[0].array() - mean_matrix.array()).pow(2).matrix();

            for (int i = 1; i < matrices.size(); i++) {
                se_matrix =
                    se_matrix +
                    (matrices[i].array() - mean_matrix.array()).pow(2).matrix();
            }

            se_matrix = (se_matrix.array().sqrt() / matrices.size()).matrix();

            return se_matrix;
        }

        string to_string(const Eigen::VectorXd& vector) {
            stringstream ss;
            ss << vector;
            return ss.str();
        }

        string to_string(const Eigen::MatrixXd& matrix) {
            stringstream ss;
            ss << matrix;
            return ss.str();
        }

        void matrix_vstack(Eigen::MatrixXd& original_matrix,
                           const Eigen::MatrixXd& other_matrix) {

            if (original_matrix.size() == 0) {
                original_matrix = other_matrix;
            }
            else {
                int cols = original_matrix.cols();
                int old_rows = original_matrix.rows();
                int new_rows = old_rows + other_matrix.rows();

                original_matrix.conservativeResize(new_rows, Eigen::NoChange);
                original_matrix.block(old_rows, 0, other_matrix.rows(), cols) =
                    other_matrix;
            }
        }

        void matrix_hstack(Eigen::MatrixXd& original_matrix,
                           const Eigen::MatrixXd& other_matrix) {

            if (original_matrix.size() == 0) {
                original_matrix = other_matrix;
            }
            else {
                int rows = original_matrix.rows();
                int old_cols = original_matrix.cols();
                int new_cols = old_cols + other_matrix.cols();

                original_matrix.conservativeResize(Eigen::NoChange, new_cols);
                original_matrix.block(0, old_cols, rows, other_matrix.cols()) =
                    other_matrix;
            }
        }

        Eigen::MatrixXi to_categorical(const Eigen::VectorXi& integers,
                                       int num_bits) {
            int rows = integers.size();
            Eigen::MatrixXi cat = Eigen::MatrixXi::Zero(rows, num_bits);

            for (int i = 0; i < rows; i++) {
                cat(i, integers(i)) = 1;
            }

            return cat;
        }

        Eigen::VectorXd flatten_rowwise(const Eigen::MatrixXd& matrix) {
            Eigen::MatrixXd copy = matrix;
            return Eigen::Map<Eigen::VectorXd>(copy.data(), copy.size());
        }

    } // namespace model
} // namespace tomcat
