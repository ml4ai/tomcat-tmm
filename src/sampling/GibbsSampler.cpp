#include "GibbsSampler.h"

#include <unordered_set>

// This is deprecated. The new version is in boost/timer/progress_display.hpp
// but only available for boost 1.72
#include <boost/progress.hpp>

#include "AncestralSampler.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        GibbsSampler::GibbsSampler(shared_ptr<DynamicBayesNet> model,
                                   int burn_in_period)
            : Sampler(model), burn_in_period(burn_in_period) {}

        GibbsSampler::~GibbsSampler() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        GibbsSampler& GibbsSampler::operator=(const GibbsSampler& sampler) {
            this->copy_sampler(sampler);
            return *this;
        }

        GibbsSampler::GibbsSampler(const GibbsSampler& sampler) {
            this->copy_sampler(sampler);
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void GibbsSampler::copy_sampler(const GibbsSampler& sampler) {
            Sampler::copy_sampler(sampler);
            this->burn_in_period = sampler.burn_in_period;
            this->node_label_to_samples = sampler.node_label_to_samples;
            this->iteration = sampler.iteration;
        }

        void GibbsSampler::sample_latent(shared_ptr<gsl_rng> random_generator,
                                         int num_samples) {
            // We will proceed from the root to the leaves so that child nodes
            // can update the sufficient statistics of parameter nodes correctly
            // given that parent nodes were already sampled.
            this->reset();
            vector<shared_ptr<Node>> data_nodes;
            vector<shared_ptr<Node>> parameter_nodes;
            vector<shared_ptr<Node>> latent_nodes;
            for (auto& node : this->model->get_nodes_topological_order()) {
                if (node->get_metadata()->is_parameter()) {
                    shared_ptr<RandomVariableNode> rv_node =
                        dynamic_pointer_cast<RandomVariableNode>(node);
                    if (!rv_node->is_frozen()) {
                        if (this->max_time_step_to_sample < 0 ||
                            rv_node->get_time_step() <=
                                this->max_time_step_to_sample) {
                            // TODO - change this if we need to have a parameter
                            //  that depends on other parameters.
                            parameter_nodes.push_back(node);
                            latent_nodes.push_back(node);
                        }
                    }
                }
                else {
                    // Observable nodes (frozen) need to be processed to update
                    // the sufficient statistics of the distribution they depend
                    // on.
                    data_nodes.push_back(node);
                    latent_nodes.push_back(node);
                }
            }

            this->fill_initial_samples(random_generator);
            this->init_samples_storage(num_samples, latent_nodes);
            bool discard = true;
            LOG("Burn-in");
            boost::progress_display progress(this->burn_in_period);

            // O(s*max{nct^2*min{kd, k^p(p-1) + d}, npk})
            for (int i = 0; i < this->burn_in_period + num_samples; i++) {
                if (i >= burn_in_period && discard) {
                    discard = false;
                    LOG("Sampling");
                    progress.restart(num_samples);
                }

                // O(min{nct^2kd, nct^2(k^p(p-1) + d)})
                for (auto& node : data_nodes) {
                    // O(min{ctkd, ct(k^p(p-1) + d)})
                    this->sample_data_node(random_generator, node, discard);
                }

                for (auto& node : parameter_nodes) { // O(npk)
                    this->sample_parameter_node(
                        random_generator, node, discard); // O(1)
                }

                ++progress;
                if (!discard) {
                    this->iteration++;
                }
            }
        }

        void GibbsSampler::reset() {
            this->iteration = 0;
            this->node_label_to_samples.clear();
        }

        void GibbsSampler::fill_initial_samples(
            const shared_ptr<gsl_rng>& random_generator) {
            // Observable nodes are already frozen by the Gibbs sampler thus
            // there's no need to add them as data in the ancestral sampler.
            AncestralSampler initial_sampler(this->model);
            initial_sampler.set_num_in_plate_samples(
                this->num_in_plate_samples);
            initial_sampler.sample(random_generator, 1);
        }

        void GibbsSampler::init_samples_storage(
            int num_samples, const vector<shared_ptr<Node>>& latent_nodes) {
            // If there's no observation for a node in a specific time step,
            // this might be inferred by the value in the column that
            // represent such time step in the matrix of samples as it's
            // going to be filled with the original value = -1. Therefore,
            // all the matrices of samples have the same size, regardless of
            // the node's initial time step.
            for (const auto& node : latent_nodes) {
                string node_label = node->get_metadata()->get_label();
                if (!EXISTS(node_label, this->node_label_to_samples)) {
                    int sample_size = node->get_metadata()->get_sample_size();
                    this->node_label_to_samples[node_label] =
                        Tensor3::constant(sample_size,
                                          num_samples,
                                          this->model->get_time_steps(),
                                          -1);
                }
            }
        }

        void GibbsSampler::sample_data_node(
            const shared_ptr<gsl_rng>& random_generator,
            const shared_ptr<Node>& node,
            bool discard) {

            shared_ptr<RandomVariableNode> rv_node =
                dynamic_pointer_cast<RandomVariableNode>(node);

            if (!rv_node->is_frozen()) {
                Eigen::MatrixXd sample =
                    rv_node->sample_from_posterior(random_generator);

                rv_node->set_assignment(sample);
                if (!discard) {
                    this->keep_sample(rv_node, sample);
                }
            }

            rv_node->update_parents_sufficient_statistics();
        }

        void
        GibbsSampler::keep_sample(const shared_ptr<RandomVariableNode>& node,
                                  const Eigen::MatrixXd& sample) {
            if (sample.rows() == 1) {
                string node_label = node->get_metadata()->get_label();
                this->sampled_node_labels.insert(node_label);
                int time_step = node->get_time_step();
                for (int i = 0; i < sample.cols(); i++) {
                    this->node_label_to_samples.at(node_label)(
                        i, this->iteration, time_step) = sample(0, i);
                }
            }
        }

        void GibbsSampler::sample_parameter_node(
            const shared_ptr<gsl_rng>& random_generator,
            const shared_ptr<Node>& node,
            bool discard) {

            vector<shared_ptr<Node>> parent_nodes =
                this->model->get_parent_nodes_of(node, true);
            shared_ptr<RandomVariableNode> rv_node =
                dynamic_pointer_cast<RandomVariableNode>(node);
            Eigen::MatrixXd sample = rv_node->sample_from_conjugacy(
                random_generator, parent_nodes, rv_node->get_size());
            rv_node->set_assignment(sample);

            // As nodes are processed, the sufficient statistic
            // table of their dependent parent parameter nodes are
            // updated accordingly. So at this point we already have all
            // the information needed to sample the parameter from its
            // posterior.
            rv_node->reset_sufficient_statistics();

            if (!discard) {
                this->keep_sample(rv_node, sample);
            }
        }

        Tensor3 GibbsSampler::get_samples(const string& node_label) const {
            return this->node_label_to_samples.at(node_label);
        }

        void GibbsSampler::get_info(nlohmann::json& json) const {
            json["name"] = "gibbs";
            json["burn_in"] = this->burn_in_period;
        }

    } // namespace model
} // namespace tomcat
