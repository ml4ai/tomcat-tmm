#include "DBNTrainer.h"

#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        DBNTrainer::DBNTrainer() {}

        DBNTrainer::~DBNTrainer() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        unordered_map<string, Tensor3> DBNTrainer::get_partials() const {
            return this->param_label_to_samples;
        }

        int DBNTrainer::get_num_partials() const {
            for (const auto& [_, param_samples] :
                 this->param_label_to_samples) {
                return param_samples.get_shape()[1];
            }
        }

        void DBNTrainer::update_model_from_partial(int sample_idx) {
            this->update_model(make_unique<int>(sample_idx));
        }

        void DBNTrainer::update_model(unique_ptr<int> sample_idx) {
            shared_ptr<DynamicBayesNet> model = this->get_model();

            for (const auto& node : model->get_parameter_nodes()) {
                shared_ptr<RandomVariableNode> rv_node =
                    dynamic_pointer_cast<RandomVariableNode>(node);
                if (!rv_node->is_frozen()) {
                    string node_label = node->get_metadata()->get_label();
                    if (EXISTS(node_label, this->param_label_to_samples)) {
                        int time_step = rv_node->get_time_step();

                        Eigen::MatrixXd samples =
                            this->param_label_to_samples[node_label](time_step,
                                                                     2)
                                .transpose();
                        Eigen::MatrixXd param_value(1, samples.cols());

                        if (sample_idx) {
                            param_value = samples.row(*sample_idx);
                        }
                        else {
                            param_value = samples.colwise().mean();
                        }

                        rv_node->set_assignment(param_value);
                    }
                }
            }
        }

        void DBNTrainer::update_model_from_partials() {
            this->update_model(nullptr);
        }

    } // namespace model
} // namespace tomcat
