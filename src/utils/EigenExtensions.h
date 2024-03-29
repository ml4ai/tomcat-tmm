#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

namespace tomcat {
    namespace model {

        /**
         * Compute the cumulative sum of an eigen matrix.
         *
         * @param matrix: matrix
         * @param axis: axis to perform the operation on
         *
         * @return Cumulative sum over an axis.
         */
        Eigen::MatrixXd cum_sum(const Eigen::MatrixXd& matrix, int axis);

        /**
         * Compute the element-wise mean for a list of matrices.
         *
         * @param matrices: list of matrices to compute the mean
         *
         * @return Matrix formed by the average of the elements from the
         * matrices in the list.
         */
        Eigen::MatrixXd mean(const std::vector<Eigen::MatrixXd>& matrices);

        /**
         * Compute the element-wise standard error for a list of matrices.
         *
         * @param matrices: list of matrices to compute the standard error
         *
         * @return Matrix formed by the standard error of the elements from the
         * matrices in the list.
         */
        Eigen::MatrixXd
        standard_error(const std::vector<Eigen::MatrixXd>& matrices);

        /**
         * Returns a string representation of the vector.
         *
         * @param vector: vector
         *
         * @return String representation of the vector.
         */
        std::string to_string(const Eigen::VectorXd& vector);

        /**
         * Returns a string representation of the matrix.
         *
         * @param matrix: matrix
         *
         * @return String representation of the matrix.
         */
        std::string to_string(const Eigen::MatrixXd& matrix);


        /**
         * Returns a string representation of the matrix.
         *
         * @param matrix: matrix
         *
         * @return String representation of the matrix.
         */
        std::string to_string(const Eigen::MatrixXi& matrix);


        /**
         * Appends a matrix in the original one row wise.
         *
         * @param original_matrix: matrix to be appended
         * @param other_matrix: matrix to append in the original one
         */
        void matrix_vstack(Eigen::MatrixXd& original_matrix,
                    const Eigen::MatrixXd& other_matrix);

        /**
         * Appends a matrix in the original one column wise.
         *
         * @param original_matrix: matrix to be appended
         * @param other_matrix: matrix to append in the original one
         */
        void matrix_hstack(Eigen::MatrixXd& original_matrix,
                    const Eigen::MatrixXd& other_matrix);


        /**
         * Transform a vector of numbers into a matrix where each row
         * contains the binary representation of each number in the vector.
         *
         * @param integers: list of integers
         */
        Eigen::MatrixXi to_categorical(const Eigen::VectorXi& integers, int
        num_bits);

        /**
         * Copies the elements of a matrix to a vector in a row wise manner.
         *
         * @param matrix: matrix
         *
         * @return Flat matrix
         */
        Eigen::VectorXd flatten_rowwise(const Eigen::MatrixXd& matrix);

    } // namespace model
} // namespace tomcat
