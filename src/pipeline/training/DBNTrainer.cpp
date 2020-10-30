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
        void DBNTrainer::prepare() {
            this->param_label_to_samples.clear();
        }


        unordered_map<string, Tensor3>
        DBNTrainer::get_partials(int split_idx) const {
            return this->param_label_to_samples[split_idx];
        }

        int DBNTrainer::get_num_partials() const {
            if (!this->param_label_to_samples.empty()) {
                // The number of samples for any node and split are the same,
                // so it suffices to check one of the splits and nodes.
                for (const auto& [_, param_samples] :
                     this->param_label_to_samples[0]) {
                    return param_samples.get_shape()[1];
                }
            }

            return 0;
        }

        void DBNTrainer::update_model_from_partial(int sample_idx,
                                                   int split_idx,
                                                   bool force) {
            this->update_model(make_unique<int>(sample_idx), split_idx, force);
        }

        void DBNTrainer::update_model(unique_ptr<int> sample_idx,
                                      int split_idx,
                                      bool force) {
            shared_ptr<DynamicBayesNet> model = this->get_model();

            for (const auto& node : model->get_parameter_nodes()) {
                shared_ptr<RandomVariableNode> rv_node =
                    dynamic_pointer_cast<RandomVariableNode>(node);
                if (!rv_node->is_frozen() || force) {
                    string node_label = node->get_metadata()->get_label();
                    if (EXISTS(node_label,
                               this->param_label_to_samples[split_idx])) {
                        int time_step = rv_node->get_time_step();

                        Eigen::MatrixXd samples =
                            this->param_label_to_samples[split_idx][node_label](
                                    time_step, 2)
                                .transpose();
                        Eigen::MatrixXd param_value(1, samples.cols());

                        if (sample_idx) {
                            param_value = samples.row(*sample_idx);
                        }
                        else {
                            param_value = samples.colwise().mean();
                        }

                        if (rv_node->is_frozen()) {
                            rv_node->unfreeze();
                            rv_node->set_assignment(param_value);
                            rv_node->freeze();
                        }
                        else {
                            rv_node->set_assignment(param_value);
                        }
                    }
                }
            }
        }

        void DBNTrainer::update_model_from_partials(int split_idx, bool force) {
            this->update_model(nullptr, split_idx, force);
        }

    } // namespace model
} // namespace tomcat
