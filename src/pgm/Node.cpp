#include "Node.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Node::Node() {}

        Node::Node(const shared_ptr<NodeMetadata>& metadata)
            : metadata(move(metadata)) {
            this->stacked_assignment = Eigen::MatrixXd(0, 0);
        }

        Node::~Node() {}

        //----------------------------------------------------------------------
        // Operator overload
        //----------------------------------------------------------------------
        ostream& operator<<(ostream& os, const Node& node) {
            node.print(os);
            return os;
        };

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Node::print(ostream& os) const { os << this->get_description(); }

        int Node::get_size() const { return this->assignment.rows(); }

        void Node::stack_assignment() {
            this->stacked_assignment = this->assignment;
        }

        void Node::increment_assignment(int increment) {
            this->assignment.array() += increment;
        }

        void Node::pop_assignment() {
            if (this->stacked_assignment.size() > 0) {
                this->assignment = this->stacked_assignment;
            }
            this->stacked_assignment = Eigen::MatrixXd(0, 0);
        }

        void Node::invert_assignment() {
            this->assignment.array() = 1 / this->assignment.array();
        }

        bool Node::is_random_variable() const { return false; }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        const shared_ptr<NodeMetadata>& Node::get_metadata() const {
            return metadata;
        }

        const Eigen::MatrixXd& Node::get_assignment() const {
            return assignment;
        }

        void Node::set_assignment(const Eigen::MatrixXd& assignment) {
            this->assignment = assignment;
        }

    } // namespace model
} // namespace tomcat
