#include "NodeProbability.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        NodeProbability::NodeProbability(std::string node_label,
                                         Eigen::VectorXd assignment,
                                         int inference_horizon)
            : CustomSamplingMetric(inference_horizon), node_label(node_label),
              assignment(assignment) {}

        NodeProbability::~NodeProbability() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        NodeProbability::NodeProbability(
            const NodeProbability& node_probability)
            : CustomSamplingMetric(node_probability.inference_horizon) {
            this->copy(node_probability);
        }

        NodeProbability&
        NodeProbability::operator=(const NodeProbability& node_probability) {
            this->copy(node_probability);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void NodeProbability::copy(const NodeProbability& node_probability) {
            CustomSamplingMetric::copy(node_probability);
            this->node_label = node_probability.node_label;
            this->assignment = node_probability.assignment;
        }

        std::vector<double>
        NodeProbability::calculate(const std::shared_ptr<Sampler>& sampler,
                                   int time_step) const {
            const auto& node_metadata =
                sampler->get_model()->get_metadata_of(this->node_label);
            int node_initial_time_step = node_metadata->get_initial_time_step();
            Eigen::MatrixXd samples(0, 0);
            if (this->inference_horizon > 0) {
                samples = sampler->get_samples(this->node_label)(0, 0);
                samples = samples.block(
                    0, time_step + 1, samples.rows(), this->inference_horizon);
            }
            else {
                if (time_step >= node_initial_time_step) {
                    samples = sampler->get_samples(this->node_label)(0, 0).col(
                        node_initial_time_step);
                }
            }

            int k = 1;
            double low;
            double high;
            if (this->assignment.size() == 0) {
                // For each possible discrete value the node can take, we
                // compute the probability estimate for the node.
                k = sampler->get_model()->get_cardinality_of(this->node_label);
            }
            else {
                low = this->assignment(0);
                high = this->assignment(0);
            }

            vector<double> probs_per_assignment(k);
            for (int i = 0; i < k; i++) {
                double probability;
                if (samples.size() == 0) {
                    probability = NO_OBS;
                }
                else {
                    if (k > 1) {
                        low = i;
                        high = i;
                    }

                    probs_per_assignment[i] =
                        this->get_probability_in_range(samples, low, high);
                }
            }

            return probs_per_assignment;
        }

        double NodeProbability::get_probability_in_range(
            const Eigen::MatrixXd& samples, double low, double high) const {

            Eigen::VectorXd matches;
            double probability;
            if (low == high) {
                matches =
                    ((samples.array() == low).cast<int>().rowwise().sum() > 0)
                        .cast<double>();
            }
            else {
                Eigen::VectorXi matches_low =
                    ((samples.array() >= low).cast<int>().rowwise().sum() > 0)
                        .cast<int>();

                Eigen::VectorXi matches_high =
                    ((samples.array() <= high).cast<int>().rowwise().sum() > 0)
                        .cast<int>();
                matches = (matches_low.array() == matches_high.array())
                              .cast<double>();
            }

            probability = matches.mean();
            return probability;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        const string& NodeProbability::get_node_label() const {
            return node_label;
        }
        const Eigen::VectorXd& NodeProbability::get_assignment() const {
            return assignment;
        }

    } // namespace model
} // namespace tomcat
