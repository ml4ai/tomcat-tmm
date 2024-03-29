#include "AncestralSampler.h"

#include "utils/Multithreading.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        AncestralSampler::AncestralSampler(
            const shared_ptr<DynamicBayesNet>& model, int num_jobs)
            : Sampler(model, num_jobs) {}

        AncestralSampler::~AncestralSampler() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        AncestralSampler&
        AncestralSampler::operator=(const AncestralSampler& sampler) {
            this->copy_sampler(sampler);
            return *this;
        }

        AncestralSampler::AncestralSampler(const AncestralSampler& sampler) {
            this->copy_sampler(sampler);
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void AncestralSampler::sample_latent(
            const shared_ptr<gsl_rng>& random_generator) {

            NodeSet node_set = this->get_node_set();

            vector<shared_ptr<gsl_rng>> random_generators_per_job =
                split_random_generator(random_generator, this->num_jobs);

            for (auto& node : node_set.nodes_to_sample) {
                const string& node_label = node->get_metadata()->get_label();
                this->sampled_node_labels.insert(node_label);

                shared_ptr<RandomVariableNode> rv_node =
                    dynamic_pointer_cast<RandomVariableNode>(node);

                // If data was provided for in-plate nodes. We have to
                // respect the number of data points given since the nodes
                // will be frozen for more sampling. The maximum number of
                // samples we'll be able to generate for other nodes will be
                // the number of in-plate nodes.
                int max_num_samples = this->num_samples;
                if (node->get_metadata()->is_in_plate()) {
                    if (this->num_in_plate_samples > 0) {
                        max_num_samples = this->num_in_plate_samples;
                    }
                }

                Eigen::MatrixXd sample;
                if (rv_node->get_time_step() <=
                    this->equal_samples_time_step_limit) {
                    // If samples up to a time t are required to have the same
                    // values, we generate a single sample following the
                    // ancestral sampling procedure and replicate it.

                    sample = rv_node->sample(random_generators_per_job,
                                             1,
                                             this->time_steps_per_sample);
                    sample = sample.replicate(max_num_samples, 1);
                }
                else {
                    sample = rv_node->sample(random_generators_per_job,
                                             max_num_samples,
                                             this->time_steps_per_sample);
                }

                // A sample is stored as an assignment of the node.
                rv_node->set_assignment(sample);
            }
        }

        AncestralSampler::NodeSet AncestralSampler::get_node_set() const {
            NodeSet node_set;

            // We start by sampling nodes in the root and keep moving forward
            // until we reach the leaves. Sampling a node depends on the values
            // sampled from its parents. Therefore, a top-down topological order
            // of the nodes is used here.
            for (int t = 0; t < this->model->get_time_steps(); t++) {
                for (auto& node :
                     this->model->get_nodes_in_topological_order_at(t)) {

                    shared_ptr<RandomVariableNode> rv_node =
                        dynamic_pointer_cast<RandomVariableNode>(node);

                    // If a node is frozen, we don't generate samples for it as
                    // it already contains pre-defined samples.
                    if (rv_node->is_frozen()) {
                        continue;
                    }

                    node_set.nodes_to_sample.push_back(node);
                }
            }

            return node_set;
        }

        void AncestralSampler::get_info(nlohmann::json& json) const {
            json["name"] = "ancestral";
        }

        unique_ptr<Sampler> AncestralSampler::clone(bool unroll_model) const {
            unique_ptr<Sampler> new_sampler =
                make_unique<AncestralSampler>(this->model, this->num_jobs);
            // Clone the model and the nodes in it
            DynamicBayesNet model_copy = *dynamic_cast<DynamicBayesNet*>(this->model->clone().get());
            if (unroll_model) {
                model_copy.unroll(this->model->get_time_steps(), true);
            }
            new_sampler->set_model(
                make_shared<DynamicBayesNet>(model_copy));

            return new_sampler;
        }

        unordered_set<string>
        AncestralSampler::get_sampled_node_labels() const {
            unordered_set<string> labels;

            for (auto node : this->get_node_set().nodes_to_sample) {
                labels.insert(node->get_metadata()->get_label());
            }

            return labels;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void AncestralSampler::set_equal_samples_time_step_limit(
            int equal_samples_time_step_limit) {
            this->equal_samples_time_step_limit = equal_samples_time_step_limit;
        }

    } // namespace model
} // namespace tomcat
