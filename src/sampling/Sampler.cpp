#include "Sampler.h"

#include <array>

#include <boost/filesystem.hpp>
#include <fmt/format.h>

#include "utils/FileHandler.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Sampler::Sampler() {}

        Sampler::Sampler(const shared_ptr<DynamicBayesNet>& model, int num_jobs)
            : model(model), num_jobs(num_jobs) {

            if (num_jobs < 1) {
                throw invalid_argument("The number of jobs has to be at least "
                                       "one.");
            }
        }

        Sampler::~Sampler() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Sampler::copy_sampler(const Sampler& sampler) {
            this->model = sampler.model;
            this->num_in_plate_samples = sampler.num_in_plate_samples;
            this->sampled_node_labels = sampler.sampled_node_labels;
        }

        void Sampler::sample(const shared_ptr<gsl_rng>& random_generator,
                             int num_samples) {
            this->freeze_observable_nodes();
            this->sample_latent(random_generator, num_samples);
            this->unfreeze_observable_nodes();
        }

        void Sampler::add_data(const EvidenceSet& data) {
            if (data.get_num_data_points() == this->num_in_plate_samples) {
                this->data = data;
                for (const auto& node_label : this->data.get_node_labels()) {
                    for (auto& node :
                         this->model->get_nodes_by_label(node_label)) {
                        shared_ptr<RandomVariableNode> rv_node =
                            dynamic_pointer_cast<RandomVariableNode>(node);

                        Tensor3 node_data = data[node_label];
                        int t = rv_node->get_time_step();
                        if (t < node_data.get_shape().at(2)) {
                            rv_node->set_assignment(
                                node_data(t, 2).transpose());
                        }
                    }
                }
            }
            else {
                throw invalid_argument(
                    "The number of data points must be equal to the number of "
                    "samples of in-plate nodes defined for the sampler.");
            }
        }

        Tensor3 Sampler::get_samples(const string& node_label) const {
            vector<shared_ptr<Node>> nodes =
                this->model->get_nodes_by_label(node_label);

            // Get the max time step for which samples were generated
            int max_time_step = this->max_time_step_to_sample > 0
                                    ? min(this->max_time_step_to_sample + 1,
                                          this->model->get_time_steps())
                                    : this->model->get_time_steps();

            return this->get_samples(node_label, 0, max_time_step);
        }

        Tensor3 Sampler::get_samples(const string& node_label,
                                     int low_time_step,
                                     int high_time_step) const {

            if (!this->model->has_node_with_label(node_label)) {
                stringstream ss;
                ss << "The node " << node_label
                   << " does not belong to the "
                      "model.";
                throw invalid_argument(ss.str());
            }

            // Get the max time step for which samples were generated
            int min_time_step = low_time_step;
            int max_time_step =
                min(high_time_step, this->model->get_time_steps());

            // Use the first node to get the size of the matrix
            const NodePtr first_node =
                this->model->get_nodes_by_label(node_label)[0];
            int d1 = first_node->get_metadata()->get_sample_size();
            int d2 = first_node->get_size();
            int d3 = max_time_step - min_time_step;
            double* buffer = new double[d1 * d2 * d3];
            fill_n(buffer, d1 * d2 * d3, NO_OBS);
            for (int t = min_time_step; t < max_time_step; t++) {
                const auto& metadata = this->model->get_metadata_of(node_label);
                // If a node is not replicable, it means there's
                // only one instance of it in the unrolled DBN. We repeat
                // the samples from that single time step for the time
                // range requested.
                int node_time_step =
                    metadata->is_replicable()
                        ? t
                        : min(t, metadata->get_initial_time_step());
                if (const auto& node =
                        this->model->get_node(node_label, node_time_step)) {
                    Eigen::Matrix assignment = node->get_assignment();
                    for (int i = 0; i < assignment.rows(); i++) {
                        for (int j = 0; j < assignment.cols(); j++) {
                            int index =
                                j * d2 * d3 + i * d3 + (t - min_time_step);
                            buffer[index] = assignment(i, j);
                        }
                    }
                }
            }

            return Tensor3(buffer, d1, d2, d3);
        }

        void Sampler::prepare() {}

        void Sampler::save_samples_to_folder(
            const string& output_folder,
            const unordered_set<string> excluding) const {

            boost::filesystem::create_directories(output_folder);

            for (const auto& node_label : this->sampled_node_labels) {
                if (!EXISTS(node_label, excluding)) {
                    string filename = node_label;
                    string filepath = get_filepath(output_folder, filename);
                    save_tensor_to_file(filepath,
                                        this->get_samples(node_label));
                }
            }
        }

        void Sampler::freeze_observable_nodes() {
            for (const auto& node_label : this->data.get_node_labels()) {
                for (auto& node : this->model->get_nodes_by_label(node_label)) {
                    dynamic_pointer_cast<RandomVariableNode>(node)->freeze();
                }
            }
        }

        void Sampler::unfreeze_observable_nodes() {
            for (const auto& node_label : this->data.get_node_labels()) {
                for (auto& node : this->model->get_nodes_by_label(node_label)) {
                    dynamic_pointer_cast<RandomVariableNode>(node)->unfreeze();
                }
            }
        }

        // ---------------------------------------------------------------------
        // Getters & Setters
        // ---------------------------------------------------------------------
        void Sampler::set_num_in_plate_samples(int num_in_plate_samples) {
            this->num_in_plate_samples = num_in_plate_samples;
        }

        const shared_ptr<DynamicBayesNet>& Sampler::get_model() const {
            return model;
        }

        void Sampler::set_model(const shared_ptr<DynamicBayesNet>& model) {
            this->model = model;
        }

        void Sampler::set_min_initialization_time_step(int time_step) {
            this->min_initialization_time_step = time_step;
        }

        void Sampler::set_min_time_step_to_sample(int time_step) {
            if (time_step < 0) {
                throw TomcatModelException("The minimum time step to sample "
                                           "must be at least 0.");
            }
            this->min_time_step_to_sample = time_step;
        }

        void Sampler::set_max_time_step_to_sample(int time_step) {
            this->max_time_step_to_sample = time_step;
        }

        void Sampler::set_trainable(bool trainable) {
            this->trainable = trainable;
        }

        int Sampler::get_num_jobs() const { return num_jobs; }

        void Sampler::set_sample_after_max_time_step(
            bool sample_after_max_time_step) {
            this->sample_after_max_time_step = sample_after_max_time_step;
        }

    } // namespace model
} // namespace tomcat
