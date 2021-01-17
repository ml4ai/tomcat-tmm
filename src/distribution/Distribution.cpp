#include "distribution/Distribution.h"

#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Distribution::Distribution() {}

        Distribution::Distribution(const vector<shared_ptr<Node>>& parameters)
            : parameters(parameters) {}

        Distribution::Distribution(vector<shared_ptr<Node>>&& parameters)
            : parameters(move(parameters)) {}

        Distribution::~Distribution() {}

        //----------------------------------------------------------------------
        // Operator overload
        //----------------------------------------------------------------------
        ostream& operator<<(ostream& os, const Distribution& distribution) {
            distribution.print(os);
            return os;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Distribution::update_dependencies(
            const Node::NodeMap& parameter_nodes_map, int time_step) {

            for (auto& parameter : this->parameters) {
                string parameter_timed_name;
                const shared_ptr<NodeMetadata>& metadata =
                    parameter->get_metadata();
                if (metadata->is_replicable()) {
                    parameter_timed_name = metadata->get_timed_name(time_step);
                }
                else {
                    // If not replicable, there'll be only one instance of
                    // the parameter at its initial time step.
                    parameter_timed_name = metadata->get_timed_name(
                        metadata->get_initial_time_step());
                }

                if (parameter_nodes_map.count(parameter_timed_name) > 0) {
                    parameter = parameter_nodes_map.at(parameter_timed_name);
                }
            }
        }

        void Distribution::update_sufficient_statistics(
            const vector<double>& values) {
            for (auto& parameter : this->parameters) {
                if (parameter->get_metadata()->is_parameter()) {
                    if (RandomVariableNode* rv_node =
                            dynamic_cast<RandomVariableNode*>(
                                parameter.get())) {
                        rv_node->add_to_sufficient_statistics(values);
                    }
                }
            }
        }

        Eigen::VectorXd Distribution::get_values(int parameter_idx) const {
            Eigen::VectorXd parameter_vector;
            if (this->parameters.size() == 1) {
                parameter_vector =
                    this->parameters[0]->get_assignment().row(parameter_idx);
            }
            else {
                parameter_vector = Eigen::VectorXd(this->parameters.size());

                int col = 0;
                for (const auto& parameter_node : this->parameters) {
                    // Each parameter is in a separate node.
                    parameter_vector(parameter_idx) =
                        parameter_node->get_assignment()(parameter_idx, col);
                    col++;
                }
            }

            return parameter_vector;
        }

        void Distribution::print(ostream& os) const {
            os << this->get_description();
        }

        void Distribution::copy(const Distribution& distribution) {
            this->parameters = distribution.parameters;
        }

    } // namespace model
} // namespace tomcat
