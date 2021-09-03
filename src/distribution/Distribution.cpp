#include "distribution/Distribution.h"

#include <thread>

#include "pgm/RandomVariableNode.h"
#include "utils/Multithreading.h"

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
            const Node::NodeMap& parameter_nodes_map) {

            for (auto& parameter : this->parameters) {
                string param_name;
                const shared_ptr<NodeMetadata>& metadata =
                    parameter->get_metadata();
                int time_step = metadata->get_initial_time_step();
                param_name = metadata->get_timed_name(time_step);

                if (EXISTS(param_name, parameter_nodes_map)) {
                    parameter = parameter_nodes_map.at(param_name);
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
                    parameter_vector(col) =
                        parameter_node->get_assignment()(parameter_idx, 0);
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

        Eigen::MatrixXd Distribution::sample_many(
            std::vector<std::shared_ptr<gsl_rng>> random_generator_per_job,
            int num_samples,
            int parameter_idx) const {

            Eigen::MatrixXd samples(num_samples, 1);
            mutex samples_mutex;

            int num_jobs = random_generator_per_job.size();
            if (num_jobs == 1) {
                // Run in the main thread
                this->run_sample_thread(random_generator_per_job.at(0),
                                        parameter_idx,
                                        samples,
                                        make_pair(0, num_samples),
                                        samples_mutex);
            }
            else {
                const auto processing_blocks =
                    get_parallel_processing_blocks(num_jobs, num_samples);
                vector<thread> threads;
                for (int i = 0; i < processing_blocks.size(); i++) {
                    thread samples_thread(&Distribution::run_sample_thread,
                                          this,
                                          ref(random_generator_per_job.at(i)),
                                          parameter_idx,
                                          ref(samples),
                                          ref(processing_blocks.at(i)),
                                          ref(samples_mutex));
                    threads.push_back(move(samples_thread));
                }

                for (auto& samples_thread : threads) {
                    samples_thread.join();
                }
            }

            return samples;
        }

        void Distribution::run_sample_thread(
            shared_ptr<gsl_rng> random_generator,
            int parameter_idx,
            Eigen::MatrixXd& full_samples,
            const pair<int, int>& processing_block,
            mutex& samples_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;

            Eigen::MatrixXd samples(num_rows, 1);

            for (int i = initial_row; i < initial_row + num_rows; i++) {
                Eigen::VectorXd sample;

                sample = this->sample(random_generator, parameter_idx);

                if (samples.cols() < sample.size()) {
                    samples.resize(Eigen::NoChange, sample.size());
                }

                samples.row(i - initial_row) = move(sample);
            }

            scoped_lock lock(samples_mutex);

            if (full_samples.cols() < samples.cols()) {
                full_samples.resize(Eigen::NoChange, samples.cols());
            }
            full_samples.block(initial_row, 0, num_rows, full_samples.cols()) =
                samples;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        const vector<std::shared_ptr<Node>>&
        Distribution::get_parameters() const {
            return parameters;
        }

    } // namespace model
} // namespace tomcat
