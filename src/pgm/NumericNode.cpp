#include "NumericNode.h"

#include <fmt/format.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        NumericNode::NumericNode(double value, const string& label) {
            this->assignment = Eigen::MatrixXd(1, 1);
            this->assignment(0, 0) = value;
            this->create_default_metadata(label, 1);
        }

        NumericNode::NumericNode(const Eigen::VectorXd& values,
                                   const string& label) {
            this->assignment = Eigen::MatrixXd(1, values.size());
            this->assignment.row(0) = values;
            this->create_default_metadata(label, values.size());
        }

        NumericNode::NumericNode(const Eigen::VectorXd&& values,
                                   const string& label) {
            this->assignment = Eigen::MatrixXd(1, values.size());
            this->assignment.row(0) = move(values);
            this->create_default_metadata(label, this->assignment.size());
        }

        NumericNode::~NumericNode() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        NumericNode::NumericNode(const NumericNode& node) {
            this->metadata = node.metadata;
            this->assignment = node.assignment;
        }

        NumericNode& NumericNode::operator=(const NumericNode& node) {
            this->metadata = node.metadata;
            this->assignment = node.assignment;
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void NumericNode::create_default_metadata(const string& label,
                                                   int sample_size) {
            NodeMetadata metadata =
                NodeMetadata::create_single_time_link_metadata(
                    label, true, false, 0, sample_size, 1);
            this->metadata = make_shared<NodeMetadata>(move(metadata));
        }

        unique_ptr<Node> NumericNode::clone() const {
            unique_ptr<NumericNode> new_node =
                make_unique<NumericNode>(*this);
            new_node->metadata = make_shared<NodeMetadata>(*this->metadata);
            return new_node;
        }

        string NumericNode::get_timed_name() const {
            return this->metadata->get_label();
        }

        string NumericNode::get_description() const {
            if (this->assignment.size() == 1) {
                stringstream assignment_string;
                assignment_string << this->assignment;

                return fmt::format("Constant({}, {})",
                                   this->metadata->get_label(),
                                   assignment_string.str());
            }
            else {
                stringstream assignment_string;
                assignment_string << this->assignment.transpose();

                return fmt::format("Constant({}, [{}])",
                                   this->metadata->get_label(),
                                   assignment_string.str());
            }
        }

    } // namespace model
} // namespace tomcat
