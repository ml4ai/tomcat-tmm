#include "Tensor3.h"

#include <iomanip>
#include <sstream>

#include <eigen3/Eigen/Core>

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Tensor3::Tensor3() {}

        Tensor3::Tensor3(double value)
            : tensor({Eigen::MatrixXd::Constant(1, 1, value)}) {}

        Tensor3::Tensor3(const Eigen::MatrixXd& matrix) : tensor({matrix}) {}

        /**
         * Creates a tensor comprised of several matrices.
         */
        Tensor3::Tensor3(const vector<Eigen::MatrixXd> matrices)
            : tensor(matrices) {}

        Tensor3::Tensor3(double* buffer, int d1, int d2, int d3) {
            this->tensor.reserve(d1);
            for (int i = 0; i < d1; i++) {
                Eigen::MatrixXd matrix =
                    Eigen::Map<Eigen::Matrix<double,
                                             Eigen::Dynamic,
                                             Eigen::Dynamic,
                                             Eigen::RowMajor>>(buffer, d2, d3);
                this->tensor.push_back(move(matrix));
                if (i < d1 - 1) {
                    buffer += d2 * d3;
                }
            }
        }

        Tensor3::~Tensor3() {}

        //----------------------------------------------------------------------
        // Operator overload
        //----------------------------------------------------------------------
        ostream& operator<<(ostream& os, const Tensor3& tensor) {
            for (int m = 0; m < tensor.tensor.size(); m++) {
                const Eigen::MatrixXd& matrix = tensor.tensor[m];

                os << Tensor3::matrix_to_string(matrix);
                os << "\n";
            }

            return os;
        }

        Eigen::MatrixXd& Tensor3::operator[](int i) { return this->tensor[i]; }

        Eigen::MatrixXd Tensor3::operator()(int i, int axis) const {
            array<int, 3> shape = this->get_shape();
            Eigen::MatrixXd matrix;

            switch (axis) {
            case 0: {
                matrix = this->tensor[i];
                break;
            }
            case 1: {
                int j = i;
                matrix = Eigen::MatrixXd(shape[0], shape[2]);
                for (int i = 0; i < this->tensor.size(); i++) {
                    matrix.row(i) = this->tensor[i].row(j);
                }
                break;
            }
            case 2: {
                int k = i;
                matrix = Eigen::MatrixXd(shape[0], shape[1]);
                for (int i = 0; i < this->tensor.size(); i++) {
                    matrix.row(i) = this->tensor[i].col(k).transpose();
                }
                break;
            }
            default: {
                throw TomcatModelException(
                    "Invalid axis. Valid axes are 0, 1 or 2.");
            }
            }

            return matrix;
        }

        Eigen::VectorXd Tensor3::at(int j, int k) const {
            size_t vector_size = this->tensor.size();
            double* buffer = new double[vector_size];

            for (int i = 0; i < vector_size; i++) {
                buffer[i] = this->tensor[i](j, k); //(*this)(i, j, k);
            }

            Eigen::Map<Eigen::VectorXd> vector(buffer, vector_size);
            delete[] buffer;

            return vector;
        }

        double& Tensor3::operator()(int i, int j, int k) {
            return this->tensor[i](j, k);
        }

        Tensor3 Tensor3::operator+(const Tensor3& tensor) const {
            if (this->get_shape()[0] != tensor.get_shape()[0] ||
                this->get_shape()[1] != tensor.get_shape()[1] ||
                this->get_shape()[2] != tensor.get_shape()[2]) {
                throw TomcatModelException(
                    "It's not possible to sum tensors of different shapes.");
            }

            Tensor3 new_tensor;
            for (int i = 0; i < this->tensor.size(); i++) {
                new_tensor.tensor.push_back(
                    (this->tensor[i].array() + tensor.tensor[i].array())
                        .matrix());
            }

            return new_tensor;
        }

        Tensor3 Tensor3::operator-(const Tensor3& tensor) const {
            if (this->get_shape()[0] != tensor.get_shape()[0] ||
                this->get_shape()[1] != tensor.get_shape()[1] ||
                this->get_shape()[2] != tensor.get_shape()[2]) {
                throw TomcatModelException(
                    "It's not possible to sum tensors of different shapes.");
            }

            Tensor3 new_tensor;
            for (int i = 0; i < this->tensor.size(); i++) {
                new_tensor.tensor.push_back(
                    (this->tensor[i].array() - tensor.tensor[i].array())
                        .matrix());
            }

            return new_tensor;
        }

        Tensor3 Tensor3::operator/(double value) const {
            if (value == 0) {
                throw domain_error(
                    "It's not possible to divide a tensor by 0.");
            }

            Tensor3 new_tensor = *this;
            for (auto& matrix : new_tensor.tensor) {
                matrix = matrix.array() / value;
            }

            return new_tensor;
        }

        Tensor3 Tensor3::operator*(double value) const {
            Tensor3 new_tensor = *this;
            for (auto& matrix : new_tensor.tensor) {
                matrix = matrix.array() * value;
            }

            return new_tensor;
        }

        Eigen::MatrixXd
        Tensor3::operator==(const Eigen::VectorXd& value) const {
            Tensor3 new_tensor;

            int i = 0;
            for (const auto& matrix : this->tensor) {
                Eigen::MatrixXd new_matrix =
                    (matrix.array() == value(i))
                        .select(
                            Eigen::MatrixXd::Ones(matrix.rows(), matrix.cols()),
                            Eigen::MatrixXd::Zero(matrix.rows(),
                                                  matrix.cols()));
                new_tensor.tensor.push_back(move(new_matrix));
                i++;
            }

            return new_tensor.coeff_wise_and(0)(0, 0);
        }

        Tensor3 Tensor3::operator/(const Eigen::MatrixXd& matrix) const {
            vector<Eigen::MatrixXd> new_tensor;
            new_tensor.reserve(this->tensor.size());

            for (const auto& tensor_matrix : this->tensor) {
                new_tensor.push_back(tensor_matrix.array() / matrix.array());
            }

            return Tensor3(new_tensor);
        }

        Tensor3 Tensor3::operator/(const Tensor3& tensor) const {
            vector<Eigen::MatrixXd> new_tensor(tensor.tensor.size());

            for (int i = 0; i < new_tensor.size(); i++) {
                new_tensor[i] =
                    this->tensor[i].array() / tensor.tensor[i].array();
            }

            return Tensor3(new_tensor);
        }

        Tensor3 Tensor3::operator*(const Eigen::MatrixXd& matrix) const {
            vector<Eigen::MatrixXd> new_tensor;
            new_tensor.reserve(this->tensor.size());

            for (const auto& tensor_matrix : this->tensor) {
                new_tensor.push_back(tensor_matrix.array() * matrix.array());
            }

            return Tensor3(new_tensor);
        }

        Tensor3 Tensor3::operator*(const Tensor3& tensor) const {
            vector<Eigen::MatrixXd> new_tensor(tensor.tensor.size());

            for (int i = 0; i < new_tensor.size(); i++) {
                new_tensor[i] =
                    this->tensor[i].array() * tensor.tensor[i].array();
            }

            return Tensor3(new_tensor);
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        Tensor3 Tensor3::constant(int d1, int d2, int d3, double value) {
            double* buffer = new double[d1 * d2 * d3];

            for (int i = 0; i < d1 * d2 * d3; i++) {
                buffer[i] = value;
            }

            Tensor3 tensor(buffer, d1, d2, d3);

            return tensor;
        }

        Tensor3 Tensor3::zeros(int d1, int d2, int d3) {
            return Tensor3::constant(d1, d2, d3, 0);
        }

        Tensor3 Tensor3::ones(int d1, int d2, int d3) {
            return Tensor3::constant(d1, d2, d3, 1);
        }

        string Tensor3::matrix_to_string(const Eigen::MatrixXd& matrix) {
            stringstream ss;
            for (int i = 0; i < matrix.rows(); i++) {
                for (int j = 0; j < matrix.cols(); j++) {
                    double value = matrix(i, j);

                    if (int(value) == value) {
                        ss << fixed << setprecision(0) << value;
                    }
                    else {
                        ss << fixed << setprecision(20) << value;
                    }

                    if (j < matrix.cols() - 1) {
                        ss << " ";
                    }
                }
                ss << "\n";
            }

            return ss.str();
        }

        Tensor3 Tensor3::sum(const vector<Tensor3>& tensors) {
            Tensor3 new_tensor;

            if (!tensors.empty()) {
                new_tensor = tensors[0];

                for (int i = 1; i < tensors.size(); i++) {
                    new_tensor = new_tensor + tensors[i];
                }
            }

            return new_tensor;
        }

        Tensor3 Tensor3::mean(const vector<Tensor3>& tensors) {
            Tensor3 new_tensor = Tensor3::sum(tensors);
            new_tensor = new_tensor / tensors.size();

            return new_tensor;
        }

        Tensor3 Tensor3::std(const vector<Tensor3>& tensors) {
            Tensor3 new_tensor;

            if (!tensors.empty()) {
                Tensor3 mean_tensor = Tensor3::mean(tensors);
                new_tensor = (tensors[0] - mean_tensor).pow(2);

                for (int i = 1; i < tensors.size(); i++) {
                    new_tensor = new_tensor + (tensors[i] - mean_tensor).pow(2);
                }

                new_tensor = (new_tensor / tensors.size()).sqrt();
            }

            return new_tensor;
        }

        Tensor3 Tensor3::dot(const Eigen::MatrixXd& matrix,
                             const Tensor3& tensor) {
            vector<Eigen::MatrixXd> new_tensor;
            new_tensor.reserve(tensor.tensor.size());

            for (const auto& tensor_matrix : tensor.tensor) {
                new_tensor.push_back(matrix * tensor_matrix);
            }

            return Tensor3(new_tensor);
        }

        Tensor3 Tensor3::dot(const Tensor3& tensor,
                             const Eigen::MatrixXd& matrix) {
            vector<Eigen::MatrixXd> new_tensor;
            new_tensor.reserve(tensor.tensor.size());

            for (const auto& tensor_matrix : tensor.tensor) {
                new_tensor.push_back(tensor_matrix * matrix);
            }

            return Tensor3(new_tensor);
        }

        Tensor3 Tensor3::dot(const Tensor3& left_tensor,
                             const Tensor3& right_tensor) {
            vector<Eigen::MatrixXd> new_tensor(left_tensor.tensor.size());

            for (int i = 0; i < new_tensor.size(); i++) {
                new_tensor[i] = left_tensor.tensor[i] * right_tensor.tensor[i];
            }

            return Tensor3(new_tensor);
        }

        Tensor3 Tensor3::eye(int depth, int size) {
            vector<Eigen::MatrixXd> eye_matrices(depth);

            for (int i = 0; i < depth; i++) {
                eye_matrices[i] = Eigen::MatrixXd::Identity(size, size);
            }

            return Tensor3(eye_matrices);
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Tensor3::clear() { this->tensor.clear(); }

        bool Tensor3::is_empty() const { return this->tensor.empty(); }

        string Tensor3::to_string() const {
            stringstream ss;
            ss << *this;
            return ss.str();
        }

        int Tensor3::get_size() const {
            return this->get_shape()[0] * this->get_shape()[1] *
                   this->get_shape()[2];
        }

        array<int, 3> Tensor3::get_shape() const {
            int i = this->tensor.size();
            int j = 0;
            int k = 0;
            if (i > 0) {
                j = this->tensor[0].rows();
                k = this->tensor[0].cols();
            }

            return array<int, 3>({i, j, k});
        }

        double Tensor3::at(int i, int j, int k) const {
            return this->tensor[i](j, k);
        }

        Tensor3
        Tensor3::slice(int initial_index, int final_index, int axis) const {
            if (final_index == ALL) {
                final_index = this->get_shape()[axis];
            }

            vector<int> indices;
            indices.reserve(final_index - initial_index);
            for (int idx = initial_index; idx < final_index; idx++) {
                indices.push_back(idx);
            }

            return slice(indices, axis);
        }

        Tensor3 Tensor3::slice(const vector<int>& indices, int axis) const {
            Tensor3 sliced_tensor;

            switch (axis) {
            case 0: {
                sliced_tensor = *this;
                for (int i = 0; i < indices.size(); i++) {
                    sliced_tensor.tensor.erase(sliced_tensor.tensor.begin() +
                                               indices[i]);
                }

                break;
            }
            case 1: {
                sliced_tensor = *this;
                for (auto& matrix : sliced_tensor.tensor) {
                    Eigen::MatrixXd sliced_matrix(indices.size(),
                                                  matrix.cols());

                    for (int j = 0; j < indices.size(); j++) {
                        sliced_matrix.row(j) = matrix.row(indices[j]);
                    }

                    matrix = sliced_matrix;
                }
                break;
            }
            case 2: {
                sliced_tensor = *this;
                for (auto& matrix : sliced_tensor.tensor) {
                    Eigen::MatrixXd sliced_matrix(matrix.rows(),
                                                  indices.size());

                    for (int k = 0; k < indices.size(); k++) {
                        sliced_matrix.col(k) = matrix.col(indices[k]);
                    }

                    matrix = sliced_matrix;
                }
                break;
            }
            default: {
                throw TomcatModelException(
                    "Invalid axis. Valid axes are 0, 1 or 2.");
            }
            }

            return sliced_tensor;
        }

        double Tensor3::mean() const {
            double mean = 0;
            for (const auto& matrix : this->tensor) {
                mean += matrix.mean();
            }

            return mean / this->tensor.size();
        }

        Tensor3 Tensor3::mean(int axis) const {
            Tensor3 new_tensor;

            switch (axis) {
            case 0: {

                new_tensor.tensor = vector<Eigen::MatrixXd>();
                for (const auto& matrix : this->tensor) {
                    if (new_tensor.tensor.empty()) {
                        new_tensor.tensor.push_back(matrix);
                    }
                    else {
                        new_tensor.tensor[0] += matrix;
                    }
                }

                new_tensor.tensor[0] =
                    new_tensor.tensor[0].array() / this->tensor.size();
                break;
            }
            case 1: {
                new_tensor = *this;

                for (auto& matrix : new_tensor.tensor) {
                    Eigen::MatrixXd new_matrix(1, matrix.cols());
                    new_matrix.row(0) = matrix.colwise().mean();
                    matrix = new_matrix;
                }
                break;
            }
            case 2: {
                new_tensor = *this;

                for (auto& matrix : new_tensor.tensor) {
                    Eigen::MatrixXd new_matrix(matrix.rows(), 1);
                    new_matrix.col(0) = matrix.rowwise().mean();
                    matrix = new_matrix;
                }
                break;
            }
            default: {
                throw TomcatModelException(
                    "Invalid axis. Valid axes are 0, 1 or 2.");
            }
            }

            return new_tensor;
        }

        Tensor3 Tensor3::reshape(int d1, int d2, int d3) const {
            if (d1 * d2 * d3 != this->get_size()) {
                throw TomcatModelException(
                    "New dimensions must result in a tensor of the same size.");
            }

            double* buffer = new double[d1 * d2 * d3];
            for (auto& matrix : this->tensor) {
                for (int i = 0; i < matrix.rows(); i++) {
                    for (int j = 0; j < matrix.cols(); j++) {
                        *buffer = matrix(i, j);
                        buffer++;
                    }
                }
            }

            buffer -= d1 * d2 * d3;
            return Tensor3(buffer, d1, d2, d3);
        }

        Tensor3 Tensor3::repeat(int num_repetitions, int axis) const {
            Tensor3 new_tensor;

            switch (axis) {
            case 0: {
                new_tensor = *this;

                for (int r = 0; r < num_repetitions - 1; r++) {
                    for (const auto& matrix : this->tensor) {
                        new_tensor.tensor.push_back(matrix);
                    }
                }
                break;
            }
            case 1: {
                new_tensor = *this;

                for (auto& matrix : new_tensor.tensor) {
                    Eigen::MatrixXd new_matrix(num_repetitions, matrix.cols());
                    new_matrix = matrix.colwise().replicate(num_repetitions);
                    matrix = new_matrix;
                }
                break;
            }
            case 2: {
                new_tensor = *this;

                for (auto& matrix : new_tensor.tensor) {
                    Eigen::MatrixXd new_matrix(matrix.rows(), num_repetitions);
                    new_matrix = matrix.rowwise().replicate(num_repetitions);
                    matrix = new_matrix;
                }
                break;
            }
            default: {
                throw TomcatModelException(
                    "Invalid axis. Valid axes are 0, 1 or 2.");
            }
            }

            return new_tensor;
        }

        Tensor3 Tensor3::pow(int power) const {
            Tensor3 new_tensor = *this;

            for (auto& matrix : new_tensor.tensor) {
                matrix = matrix.array().pow(power).matrix();
            }

            return new_tensor;
        }

        Tensor3 Tensor3::sqrt() const {
            Tensor3 new_tensor = *this;

            for (auto& matrix : new_tensor.tensor) {
                matrix = matrix.array().sqrt().matrix();
            }

            return new_tensor;
        }

        Tensor3 Tensor3::abs() const {
            Tensor3 new_tensor = *this;

            for (auto& matrix : new_tensor.tensor) {
                matrix = matrix.array().abs().matrix();
            }

            return new_tensor;
        }

        Tensor3 Tensor3::coeff_wise_and(int axis) const {
            auto [d1, d2, d3] = this->get_shape();
            Tensor3 new_tensor;

            switch (axis) {
            case 0: {
                new_tensor = Tensor3::constant(1, d2, d3, 1);
                for (int j = 0; j < d2; j++) {
                    for (int k = 0; k < d3; k++) {
                        for (int i = 0; i < d1; i++) {
                            if (this->tensor[i](j, k) <= 0) {
                                new_tensor.tensor[0](j, k) = 0;
                                break;
                            }
                        }
                    }
                }
                break;
            }
            case 1: {
                new_tensor = Tensor3::constant(d1, 1, d3, 1);
                for (int i = 0; i < d1; i++) {
                    for (int k = 0; k < d3; k++) {
                        for (int j = 0; j < d2; j++) {
                            if (this->tensor[i](j, k) <= 0) {
                                new_tensor.tensor[i](0, k) = 0;
                                break;
                            }
                        }
                    }
                }
                break;
            }
            case 2: {
                new_tensor = Tensor3::constant(d1, d2, 1, 1);
                for (int i = 0; i < d1; i++) {
                    for (int j = 0; j < d2; j++) {
                        for (int k = 0; k < d3; k++) {
                            if (this->tensor[i](j, k) <= 0) {
                                new_tensor.tensor[i](j, 0) = 0;
                                break;
                            }
                        }
                    }
                }
                break;
            }
            default: {
                throw TomcatModelException(
                    "Invalid axis. Valid axes are 0, 1 or 2.");
            }
            }

            return new_tensor;
        }

        void Tensor3::vstack(const Tensor3& other) {
            if (this->is_empty()) {
                this->tensor = other.tensor;
            }

            for (int i = 0; i < this->tensor.size(); i++) {
                matrix_vstack(this->tensor[i], other.tensor.at(i));
            }
        }

        void Tensor3::hstack(const Tensor3& other) {
            if (this->is_empty()) {
                this->tensor = other.tensor;
            }

            for (int i = 0; i < this->tensor.size(); i++) {
                matrix_hstack(this->tensor[i], other.tensor.at(i));
            }
        }

        Tensor3 Tensor3::sum_cols() const {
            vector<Eigen::MatrixXd> new_tensor;
            new_tensor.reserve(this->tensor.size());

            for (const auto& matrix : this->tensor) {
                new_tensor.push_back(matrix.rowwise().sum());
            }

            return Tensor3(new_tensor);
        }

        Tensor3 Tensor3::sum_rows() const {
            vector<Eigen::MatrixXd> new_tensor;
            new_tensor.reserve(this->tensor.size());

            for (const auto& matrix : this->tensor) {
                new_tensor.push_back(matrix.colwise().sum());
            }

            return Tensor3(new_tensor);
        }

        void Tensor3::transpose_matrices() {
            for (auto& matrix : this->tensor) {
                matrix.transposeInPlace();
            }
        }

        Tensor3
        Tensor3::mult_colwise_broadcasting(const Tensor3& tensor) const {
            vector<Eigen::MatrixXd> new_tensor(this->tensor.size());

            for (int i = 0; i < new_tensor.size(); i++) {
                new_tensor[i] = this->tensor[i].array().colwise() *
                                tensor.tensor[i].col(0).array();
            }

            return Tensor3(new_tensor);
        }

        Tensor3
        Tensor3::mult_rowwise_broadcasting(const Tensor3& tensor) const {
            vector<Eigen::MatrixXd> new_tensor(this->tensor.size());

            for (int i = 0; i < new_tensor.size(); i++) {
                new_tensor[i] = this->tensor[i].array().rowwise() *
                                tensor.tensor[i].row(0).array();
            }

            return Tensor3(new_tensor);
        }

        Tensor3
        Tensor3::mult_colwise_broadcasting(const Eigen::VectorXd& v) const {
            vector<Eigen::MatrixXd> new_tensor;
            new_tensor.reserve(this->tensor.size());

            for (const auto& tensor_matrix : this->tensor) {
                new_tensor.push_back(tensor_matrix.array().colwise() *
                                     v.array());
            }

            return Tensor3(new_tensor);
        }

        Tensor3
        Tensor3::mult_rowwise_broadcasting(const Eigen::VectorXd& v) const {
            vector<Eigen::MatrixXd> new_tensor;
            new_tensor.reserve(this->tensor.size());

            for (const auto& tensor_matrix : this->tensor) {
                new_tensor.push_back(tensor_matrix.array().rowwise() *
                                     v.transpose().array());
            }

            return Tensor3(new_tensor);
        }

        Tensor3 Tensor3::div_colwise_broadcasting(const Tensor3& tensor) const {
            vector<Eigen::MatrixXd> new_tensor(this->tensor.size());

            for (int i = 0; i < new_tensor.size(); i++) {
                new_tensor[i] = this->tensor[i].array().colwise() /
                                tensor.tensor[i].col(0).array();
            }

            return Tensor3(new_tensor);
        }

        Tensor3 Tensor3::div_rowwise_broadcasting(const Tensor3& tensor) const {
            vector<Eigen::MatrixXd> new_tensor(this->tensor.size());

            for (int i = 0; i < new_tensor.size(); i++) {
                new_tensor[i] = this->tensor[i].array().rowwise() /
                                tensor.tensor[i].row(0).array();
            }

            return Tensor3(new_tensor);
        }

        Tensor3
        Tensor3::div_colwise_broadcasting(const Eigen::VectorXd& v) const {
            vector<Eigen::MatrixXd> new_tensor;
            new_tensor.reserve(this->tensor.size());

            for (const auto& tensor_matrix : this->tensor) {
                new_tensor.push_back(tensor_matrix.array().colwise() /
                                     v.array());
            }

            return Tensor3(new_tensor);
        }

        Tensor3
        Tensor3::div_rowwise_broadcasting(const Eigen::VectorXd& v) const {
            vector<Eigen::MatrixXd> new_tensor;
            new_tensor.reserve(this->tensor.size());

            for (const auto& tensor_matrix : this->tensor) {
                new_tensor.push_back(tensor_matrix.array().rowwise() /
                                     v.transpose().array());
            }

            return Tensor3(new_tensor);
        }

        void Tensor3::normalize_columns() {
            for (auto& matrix : this->tensor) {
                Eigen::VectorXd sum_per_column = matrix.colwise().sum();
                for (int row = 0; row < matrix.rows(); row++) {
                    for (int col = 0; col < matrix.cols(); col++) {
                        if (sum_per_column[col] == 0) {
                            matrix(row, col) = 0;
                        }
                        else {
                            matrix(row, col) =
                                matrix(row, col) / sum_per_column[col];
                        }
                    }
                }
            }
        }

        void Tensor3::normalize_rows() {
            for (auto& matrix : this->tensor) {
                Eigen::VectorXd sum_per_row = matrix.rowwise().sum();
                for (int row = 0; row < matrix.rows(); row++) {
                    for (int col = 0; col < matrix.cols(); col++) {
                        if (sum_per_row[row] == 0) {
                            matrix(row, col) = 0;
                        }
                        else {
                            matrix(row, col) =
                                matrix(row, col) / sum_per_row[row];
                        }
                    }
                }
            }
        }

        Eigen::MatrixXd::RowXpr Tensor3::row(int depth, int row_idx) {
            return this->tensor[depth].row(row_idx);
        }

        Tensor3 Tensor3::row(int row_idx) const {
            return this->slice(row_idx, row_idx + 1, 1);
        }

        Eigen::MatrixXd::ColXpr Tensor3::col(int depth, int col_idx) {
            return this->tensor[depth].col(col_idx);
        }

        Tensor3 Tensor3::col(int col_idx) const {
            return this->slice(col_idx, col_idx + 1, 2);
        }

        Eigen::VectorXd Tensor3::depth(int row_idx, int col_idx) const {
            Eigen::VectorXd values(this->tensor.size());

            for (int i = 0; i < this->tensor.size(); i++) {
                values(i) = this->tensor[i](row_idx, col_idx);
            }

            return values;
        }

        bool Tensor3::equals(const Tensor3& other, double tolerance) const {
            if (this->get_shape() != other.get_shape()) {
                return false;
            }

            for (int i = 0; i < this->get_shape().at(0); i++) {
                for (int j = 0; j < this->get_shape().at(1); j++) {
                    for (int k = 0; k < this->get_shape().at(2); k++) {
                        if (std::abs(this->tensor[i](j, k) - other.at(i, j, k)) >
                            tolerance) {
                            return false;
                        }
                    }
                }
            }

            return true;
        }

    } // namespace model
} // namespace tomcat
