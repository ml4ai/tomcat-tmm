#pragma once

#include <array>
#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        //------------------------------------------------------------------
        // Forward declarations
        //------------------------------------------------------------------

        //------------------------------------------------------------------
        // Structs
        //------------------------------------------------------------------

        /**
         * Class description here
         */
        class Tensor3 {
          public:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------
            static const int ALL = -1;

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an empty tensor.
             */
            Tensor3();

            /**
             * Creates a tensor with one value.
             */
            Tensor3(double value);

            /**
             * Creates tensor with one matrix.
             */
            Tensor3(const Eigen::MatrixXd& matrix);

            /**
             * Creates a tensor comprised of several matrices.
             */
            Tensor3(const std::vector<Eigen::MatrixXd> matrices);

            /**
             * Creates a tensor filled with data.
             *
             * @param buffer: values to be stored in the tensor. There must be
             * d1*d2*d3 elements in the buffer array.
             * @param d1: dimension of the first axis
             * @param d2: dimension of the second axis
             * @param d3: dimension of the third axis
             */
            Tensor3(double* buffer, int d1, int d2, int d3);

            ~Tensor3();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Tensor3(const Tensor3&) = default;

            Tensor3& operator=(const Tensor3&) = default;

            Tensor3(Tensor3&&) = default;

            Tensor3& operator=(Tensor3&&) = default;

            //------------------------------------------------------------------
            // Operator overload
            //------------------------------------------------------------------
            friend std::ostream& operator<<(std::ostream& os,
                                            const Tensor3& tensor);

            /**
             * Returns the reference to the matrix in a given index of the
             * first dimension of the tensor.
             *
             * @param i: index in the first dimension
             *
             * @return Matrix
             */
            Eigen::MatrixXd& operator[](int i);

            /**
             * Returns assignable matrix for a given index of the first axis.
             *
             * @param i: index to select in the chosen axis
             * @param axis: axis
             *
             * @return Assignable matrix.
             */
            Eigen::MatrixXd operator()(int i, int axis) const;

            /**
             * Returns an assignable number given indices of all axes.
             *
             * @param i: first axis' index
             * @param j: second axis' index
             * @param k: third axis' index
             *
             * @return Assignable number.
             */
            double& operator()(int i, int j, int k);

            /**
             * Returns the tensor formed by the element-wise sum of two tensors.
             *
             * @param tensor: another tensor
             *
             * @return Element-wise sum of two tensors.
             */
            Tensor3 operator+(const Tensor3& tensor) const;

            /**
             * Returns the tensor formed by the element-wise subtraction of two
             * tensors.
             *
             * @param tensor: another tensor
             *
             * @return Element-wise subtraction of two tensors.
             */
            Tensor3 operator-(const Tensor3& tensor) const;

            /**
             * Returns the tensor formed by the element-wise scalar division of
             * a tensor.
             *
             * @param value: value to divide the elements of a tensor
             *
             * @return Element-wise scalar division of a tensor.
             */
            Tensor3 operator/(double value) const;

            /**
             * Returns the tensor formed by the element-wise scalar
             * multiplication of a tensor.
             *
             * @param value: value to multiply the elements of a tensor
             *
             * @return Element-wise scalar multiplication of a tensor.
             */
            Tensor3 operator*(double value) const;

            /**
             * Returns a 1 x m x n tensor comprised by ones where the
             * coefficients along axis 0 are equal to value, and zero otherwise.
             *
             * @param value: value to compare the coefficients of axis 0 with
             *
             * @return Binary tensor
             */
            Eigen::MatrixXd operator==(const Eigen::VectorXd& value) const;

            /**
             * Performs element-wise division between the matrices of the
             * 0-axis of a tensor and another matrix.
             *
             * @param matrix: matrix
             *
             * @return Resultant tensor
             */
            Tensor3 operator/(const Eigen::MatrixXd& matrix) const;

            /**
             * Performs element-wise division between the matrices of the
             * 0-axis of a tensor and the matrices in the 0-axis of another
             * tensor.
             *
             * @param tensor: tensor
             *
             * @return Resultant tensor
             */
            Tensor3 operator/(const Tensor3& tensor) const;

            /**
             * Performs element-wise multiplication between the matrices of the
             * 0-axis of a tensor and another matrix.
             *
             * @param matrix: matrix
             *
             * @return Resultant tensor
             */
            Tensor3 operator*(const Eigen::MatrixXd& matrix) const;

            /**
             * Performs element-wise multiplication between the matrices of the
             * 0-axis of a tensor and the matrices in the 0-axis of another
             * tensor.
             *
             * @param tensor: tensor
             *
             * @return Resultant tensor
             */
            Tensor3 operator*(const Tensor3& tensor) const;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------
            /**
             * Creates a tensor filled with a constant value.
             *
             * @param d1: dimension of the first axis
             * @param d2: dimension of the second axis
             * @param d3: dimension of the third axis
             * @param value: constant value
             * @return Tensor of constant values.
             */
            static Tensor3 constant(int d1, int d2, int d3, double value);

            /**
             * Creates a tensor filled with 0s.
             *
             * @param d1: dimension of the first axis
             * @param d2: dimension of the second axis
             * @param d3: dimension of the third axis
             * @return Tensor of 0s.
             */
            static Tensor3 zeros(int d1, int d2, int d3);

            /**
             * Creates a tensor filled with 1s.
             *
             * @param d1: dimension of the first axis
             * @param d2: dimension of the second axis
             * @param d3: dimension of the third axis
             * @return Tensor of 1s.
             */
            static Tensor3 ones(int d1, int d2, int d3);

            /**
             * Returns a string representation for a matrix;
             *
             * @param matrix: matrix
             * @return Matrix's string representation.
             */
            static std::string matrix_to_string(const Eigen::MatrixXd& matrix);

            /**
             * Computes the element-wise sum of a list of tensors.
             *
             * @param tensors: tensors
             *
             * @return: Element-wise sum of a list of tensors.
             */
            static Tensor3 sum(const std::vector<Tensor3>& tensors);

            /**
             * Computes the element-wise mean of a list of tensors.
             *
             * @param tensors: tensors
             *
             * @return: Element-wise mean of a list of tensors.
             */
            static Tensor3 mean(const std::vector<Tensor3>& tensors);

            /**
             * Computes the element-wise standard deviation of a list of
             * tensors.
             *
             * @param tensors: tensors
             *
             * @return: Element-wise standard deviation of a list of tensors.
             */
            static Tensor3 std(const std::vector<Tensor3>& tensors);

            /**
             * Multiply a matrix across the 0-axis of a tensor.
             *
             * @param matrix: matrix
             * @param tensor: tensor
             *
             * @return Resultant tensor
             */
            static Tensor3 dot(const Eigen::MatrixXd& matrix,
                               const Tensor3& tensor);

            /**
             * Multiply every matrix in the 0-axis of a tensor by a given
             * matrix and stores the result in a new tensor.
             *
             * @param tensor: tensor
             * @param matrix: matrix
             *
             * @return Resultant tensor
             */
            static Tensor3 dot(const Tensor3& tensor,
                               const Eigen::MatrixXd& matrix);

            /**
             * Multiply a the matrices in the 0-axis of a tensor by another.
             *
             * @param tensor_left: first operand
             * @param tensor_right: second operand
             *
             * @return Resultant tensor
             */
            static Tensor3 dot(const Tensor3& left_tensor,
                               const Tensor3& right_tensor);

            /**
             * Creates a tensor formed by a series of identity matrices.
             *
             * @param depth: size of the 0 axis
             * @param size: size of the identity matrix
             *
             * @return Resultant tensor
             */
            static Tensor3 eye(int depth, int size);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Clears the tensor;
             */
            void clear();

            /**
             * Checks whether a tensor is empty.
             */
            bool is_empty() const;

            /**
             * Returns the content of the tensor as a string;
             *
             * @return String with the tensor's content.
             */
            std::string to_string() const;

            /**
             * Returns the number of elements in the tensor.
             *
             * @return The number of elements in the tensor.
             */
            int get_size() const;

            /**
             * Returns 3D array containing the dimensions of the tensor.
             *
             * @return Tensor's dimensions.
             */
            std::array<int, 3> get_shape() const;

            /**
             * Returns a non-assignable vector for given indices of the second
             * and third axes.
             *
             * @param j: second axis' index
             * @param k: third axis' index
             * @return Non-assignable vector.
             */
            Eigen::VectorXd at(int j, int k) const;

            /**
             * Returns a non-assignable coefficient from a specific tensor
             * index.
             *
             * @param i: first axis' index
             * @param j: second axis' index
             * @param k: third axis' index
             *
             * @return Non-assignable tensor coefficient.
             */
            double at(int i, int j, int k) const;

            /**
             * Returns a sliced copy of the tensor in a certain range.
             *
             * @param initial_index: first index in the range (inclusive). The
             * element in this index is kept.
             * @param final_index: last index in the range (exclusive). The
             * element in this index is not kept.
             * @param axis: axis where the slicing must be done
             *
             * @return Sliced copy of the original tensor.
             */
            Tensor3 slice(int initial_index, int final_index, int axis) const;

            /**
             * Returns a sliced copy of the tensor.
             *
             * @param indices: indices to keep from the original tensor
             * @param axis: axis where the slicing must be done
             *
             * @return Sliced copy of the original tensor.
             */
            Tensor3 slice(const std::vector<int>& indices, int axis) const;

            /**
             * Returns the mean of all the values in the tensor.
             *
             * @return Mean.
             */
            double mean() const;

            /**
             * Returns the mean of all the values in the tensor in a given
             * axis. This operation shrinks the number of elements in the axis
             * where the operation is performed but it preserves the number of
             * axes.
             *
             * @param axis: axis where the mean must be computed along
             *
             * @return Mean.
             */
            Tensor3 mean(int axis) const;

            /**
             * Reshapes the tensor if new dimensions are compatible with the
             * number of elements in the tensor.
             *
             * @param d1: dimension of the first axis
             * @param d2: dimension of the second axis
             * @param d3: dimension of the third axis
             *
             * @return Reshaped tensor.
             */
            Tensor3 reshape(int d1, int d2, int d3) const;

            /**
             * Repeats the elements of the tensor in a given axis by a certain
             * amount of time.
             *
             * @param num_repetitions: number of repetitions
             * @param axis: axis where the repetitions must occur
             *
             * @return Repeated tensor.
             */
            Tensor3 repeat(int num_repetitions, int axis) const;

            /**
             * Returns a tensor formed by the elements of the original tensor to
             * a given power.
             *
             * @param power: power
             *
             * @return Tensor formed by the elements of the original tensor to a
             * given power.
             */
            Tensor3 pow(int power) const;

            /**
             * Returns a tensor formed by the squared root of the elements of
             * the original tensor.
             *
             * @return Tensor formed by the squared root of the elements of the
             * original tensor.
             */
            Tensor3 sqrt() const;

            /**
             * Returns a tensor formed by the absolute values of the elements of
             * the original tensor.
             *
             * @return Tensor formed by the absolute values of the elements of the
             * original tensor.
             */
            Tensor3 abs() const;

            /**
             * Returns the bitwise-and of all the coefficients in a tensor in a
             * given axis. All numbers greater greater 0 are considered as true
             * for the purpose of this logical operation. Negative numbers are
             * neutral and preserved in the final tensor. This operation shrinks
             * the number of coefficients in the axis where the operation is
             * performed but it preserves the number of axes.
             *
             * @param axis: axis where the bitwise-and must be computed along
             *
             * @return Mean.
             */
            Tensor3 coeff_wise_and(int axis) const;

            /**
             * Appends the content of another tensor into this tensor along the
             * second dimension of the tensors.
             *
             * @param other: tensor to append
             */
            void vstack(const Tensor3& other);

            /**
             * Appends the content of another tensor into this tensor along the
             * third dimension of the tensors.
             *
             * @param other: tensor to append
             */
            void hstack(const Tensor3& other);

            /**
             * Sum across columns each matrix in the 0-axis
             *
             * @return Reduced tensor
             */
            Tensor3 sum_cols() const;

            /**
             * Sum across rows each matrix in the 0-axis
             *
             * @return Reduced tensor
             */
            Tensor3 sum_rows() const;

            /**
             * Transpose each matrix of the 0-axis of the tensor.
             */
            void transpose_matrices();

            /**
             * Multiply the matrices of a given tensor with the matrices of a
             * tensor in a column-wise manner. The number of rows of the matrix
             * must coincide with the number of rows of the matrices of the
             * tensor and the number of columns of the matrix must be 1 or equal
             * to the number of columns of the matrices of the tensor.
             *
             * @param matrix: matrix
             *
             * @return Resultant tensor
             */
            Tensor3 mult_colwise_broadcasting(const Tensor3& tensor) const;

            /**
             * Multiply the matrices of a given tensor with the matrices of a
             * tensor in a row-wise manner. The number of columns of the matrix
             * must coincide with the number of columns of the matrices of the
             * tensor and the number of rows of the matrix must be 1 or equal to
             * the number of rows of the matrices of the tensor.
             *
             * @param matrix: matrix
             *
             * @return Resultant tensor
             */
            Tensor3 mult_rowwise_broadcasting(const Tensor3& tensor) const;

            /**
             * Multiply a column vector with the matrices of a tensor in a
             * column-wise manner. The size of the vector must coincide with
             * the number of rows of the matrices of the tensor.
             *
             * @param v: vector
             *
             * @return Resultant tensor
             */
            Tensor3 mult_colwise_broadcasting(const Eigen::VectorXd& v) const;

            /**
             * Multiply a column vector with the matrices of a tensor in a
             * row-wise manner. The size of the vector must coincide with the
             * number of columns of the matrices of the tensor.
             *
             * @param v: vector
             *
             * @return Resultant tensor
             */
            Tensor3 mult_rowwise_broadcasting(const Eigen::VectorXd& v) const;

            /**
             * Divide the matrices of a given tensor with the matrices of a
             * tensor in a column-wise manner. The number of rows of the matrix
             * must coincide with the number of rows of the matrices of the
             * tensor and the number of columns of the matrix must be 1 or equal
             * to the number of columns of the matrices of the tensor.
             *
             * @param matrix: matrix
             *
             * @return Resultant tensor
             */
            Tensor3 div_colwise_broadcasting(const Tensor3& tensor) const;

            /**
             * Divide the matrices of a given tensor with the matrices of a
             * tensor in a row-wise manner. The number of columns of the matrix
             * must coincide with the number of columns of the matrices of the
             * tensor and the number of rows of the matrix must be 1 or equal to
             * the number of rows of the matrices of the tensor.
             *
             * @param matrix: matrix
             *
             * @return Resultant tensor
             */
            Tensor3 div_rowwise_broadcasting(const Tensor3& tensor) const;

            /**
             * Divide a column vector with the matrices of a tensor in a
             * column-wise manner. The size of the vector must coincide with
             * the number of rows of the matrices of the tensor.
             *
             * @param v: vector
             *
             * @return Resultant tensor
             */
            Tensor3 div_colwise_broadcasting(const Eigen::VectorXd& v) const;

            /**
             * Divide a column vector with the matrices of a tensor in a
             * row-wise manner. The size of the vector must coincide with the
             * number of columns of the matrices of the tensor.
             *
             * @param v: vector
             *
             * @return Resultant tensor
             */
            Tensor3 div_rowwise_broadcasting(const Eigen::VectorXd& v) const;

            /**
             * Normalize each column of each matrix of the tensor to sum up
             * to 1.
             */
            void normalize_columns();

            /**
             * Normalize each row of each matrix of the tensor to sum up
             * to 1.
             */
            void normalize_rows();

            /**
             * Gets an assignable reference to a specific row in one of the
             * matrices in the first axis of the tensor.
             *
             * @param depth: index of the first axis
             * @param row_idx: index of the row
             *
             * @return Assignable row
             */
            Eigen::MatrixXd::RowXpr row(int depth, int row_idx);

            /**
             * Gets a new tensor formed only by one of the rows of its matrices.
             *
             * @param row_idx: index of the row
             *
             * @return Assignable row
             */
            Tensor3 row(int row_idx) const;

            /**
             * Gets an assignable reference to a specific column in one of the
             * matrices in the first axis of the tensor.
             *
             * @param depth: index of the first axis
             * @param col_idx: index of the column
             *
             * @return Tensor
             */
            Eigen::MatrixXd::ColXpr col(int depth, int col_idx);

            /**
             * Gets a new tensor formed only by one of the columns of its
             * matrices.
             *
             * @param col_idx: index of the row
             *
             * @return Tensor
             */
            Tensor3 col(int col_idx) const;

            /**
             * Returns the vector formed by the values along the depth axis.
             *
             * @param row_idx: index of the row
             * @param col_idx: index of the column
             *
             * @return Values along the first axis
             */
            Eigen::VectorXd depth(int row_idx, int col_idx) const;

            /**
             * Checks whether the elements of a tensor are equals to the
             * elements of another tensor.
             *
             * @param other: other tensor
             * @param tolerance: precision in the comparison between the values
             * of the elements
             *
             * @return
             */
            bool equals(const Tensor3& other, double tolerance = EPSILON) const;

          private:
            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            std::vector<Eigen::MatrixXd> tensor;
        };

    } // namespace model
} // namespace tomcat
