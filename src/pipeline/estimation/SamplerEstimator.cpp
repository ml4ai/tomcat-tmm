#include "SamplerEstimator.h"

#include <iostream>
#include <thread>

#include "pipeline/estimation/custom_metrics/FinalTeamScoreEstimator.h"
#include "pipeline/estimation/custom_metrics/IndependentMapVersionAssignmentEstimator.h"
#include "pipeline/estimation/custom_metrics/IndependentMarkerLegendVersionAssignmentEstimator.h"
#include "pipeline/estimation/custom_metrics/MapVersionAssignmentEstimator.h"
#include "pipeline/estimation/custom_metrics/MarkerLegendVersionAssignmentEstimator.h"
#include "pipeline/estimation/custom_metrics/NextAreaOnNearbyMarkerEstimator.h"
#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        SamplerEstimator::SamplerEstimator() {}

        SamplerEstimator::SamplerEstimator(
            const shared_ptr<DynamicBayesNet>& model,
            int inference_horizon,
            const std::string& node_label,
            const Eigen::VectorXd& low,
            const Eigen::VectorXd& high,
            FREQUENCY_TYPE frequency_type)
            : Estimator(model, inference_horizon, node_label, low),
              frequency_type(frequency_type) {

            if (low.size() == 0) {
                int cardinality = model->get_cardinality_of(node_label);
                this->estimates.estimates =
                    vector<Eigen::MatrixXd>(cardinality);
            }
            else {
                this->estimates.estimates = vector<Eigen::MatrixXd>(1);
            }
        }

        SamplerEstimator::~SamplerEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        SamplerEstimator::SamplerEstimator(const SamplerEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->copy_estimator(estimator);
        }

        SamplerEstimator&
        SamplerEstimator::operator=(const SamplerEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->copy_estimator(estimator);
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        Eigen::VectorXd SamplerEstimator::get_prior(const RVNodePtr& node) {
            Eigen::VectorXd prior;

            if (node->get_parents().empty()) {
                prior = node->get_cpd()->get_distributions()[0]->get_values(0);
            }
            else {
                const auto& indexing__map =
                    node->get_cpd()->get_parent_label_to_indexing();

                vector<Eigen::VectorXd> parent_priors(indexing__map.size());
                for (const auto& parent_node : node->get_parents()) {
                    const string& parent_label =
                        parent_node->get_metadata()->get_label();

                    parent_priors[indexing__map.at(parent_label).order] =
                        get_prior(dynamic_pointer_cast<RandomVariableNode>(
                            parent_node));
                }

                Eigen::VectorXd cartesian_product = parent_priors[0];
                for (int i = 1; i < parent_priors.size(); i++) {
                    cartesian_product = flatten_rowwise(
                        cartesian_product * parent_priors[i].transpose());
                }

                prior = cartesian_product.transpose() *
                        node->get_cpd()->get_table(0);
            }

            return prior;
        }

        SamplerEstimatorPtr SamplerEstimator::create_custom_estimator(
            const std::string& name,
            const DBNPtr& model,
            const nlohmann::json& json_config,
            FREQUENCY_TYPE frequency_type) {

            SamplerEstimatorPtr estimator;
            if (name == FinalTeamScoreEstimator::NAME) {
                estimator =
                    make_shared<FinalTeamScoreEstimator>(model, frequency_type);
            }
            else if (name == MapVersionAssignmentEstimator::NAME) {
                estimator = make_shared<MapVersionAssignmentEstimator>(
                    model, frequency_type);
            }
            else if (name == IndependentMapVersionAssignmentEstimator::NAME) {
                estimator =
                    make_shared<IndependentMapVersionAssignmentEstimator>(
                        model, frequency_type);
            }
            else if (name == MarkerLegendVersionAssignmentEstimator::NAME) {
                estimator = make_shared<MarkerLegendVersionAssignmentEstimator>(
                    model, frequency_type);
            }
            else if (name ==
                     IndependentMarkerLegendVersionAssignmentEstimator::NAME) {
                estimator = make_shared<
                    IndependentMarkerLegendVersionAssignmentEstimator>(
                    model, frequency_type);
            }
            else if (name == NextAreaOnNearbyMarkerEstimator::NAME) {
                estimator = make_shared<NextAreaOnNearbyMarkerEstimator>(
                    model, json_config);
            }

            return estimator;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void SamplerEstimator::copy(const SamplerEstimator& estimator) {
            Estimator::copy_estimator(estimator);
        }

        void SamplerEstimator::estimate(const EvidenceSet& new_data) {
            new TomcatModelException(
                "Whenever working with sampler estimators, "
                "please use the alternative estimate "
                "function.");
        }

        void SamplerEstimator::get_info(nlohmann::json& json) const {
            json["name"] = this->get_name();
            json["inference_horizon"] = this->inference_horizon;
        }

        string SamplerEstimator::get_name() const { return "sampler"; }

        bool SamplerEstimator::does_estimation_at(
            int data_point, int time_step, const EvidenceSet& new_data) const {
            bool temp = false;
            if (this->frequency_type == all ||
                EXISTS(time_step, this->fixed_steps)) {
                temp = true;
            }
            else if (this->frequency_type == dynamic) {
                temp = this->is_event_triggered_at(
                    data_point, time_step, new_data);
            }
            return temp;
        }

        bool SamplerEstimator::is_event_triggered_at(
            int data_point, int time_step, const EvidenceSet& new_data) const {
            // It works like frequency == all for a non custom estimator
            return true;
        }

        void SamplerEstimator::estimate(const EvidenceSet& new_data,
                                        const EvidenceSet& particles,
                                        const EvidenceSet& projected_particles,
                                        const EvidenceSet& marginals,
                                        int data_point_idx,
                                        int time_step) {

            if (this->inference_horizon == 0) {
                for (int t = 0; t < particles.get_time_steps(); t++) {
                    // Each dimensionality is computed individually
                    const auto& metadata = this->get_model()->get_metadata_of(
                        this->estimates.label);
                    const auto& node = this->model->get_node(
                        this->estimates.label,
                        metadata->get_initial_time_step());

                    if (this->estimates.assignment.size() == 0) {
                        int k = this->get_model()->get_cardinality_of(
                            this->estimates.label);
                        Eigen::VectorXd probs =
                            Eigen::VectorXd::Constant(k, NO_OBS);

                        if (marginals.has_data_for(this->estimates.label)) {
                            probs =
                                marginals[this->estimates.label](0, 0).col(t);
                        }
                        else {
                            if (time_step < metadata->get_initial_time_step()) {
                                probs = get_prior(node);
                            }
                            else {
                                const Tensor3 samples_tensor =
                                    particles[this->estimates.label];
                                // Compute estimates for each value the node can
                                // take (assuming the node's distribution is
                                // discrete)
                                Eigen::MatrixXd samples = samples_tensor(0, 0);

                                for (int i = 0; i < samples.rows(); i++) {
                                    int value = samples(i, t);
                                    probs[value] += 1;
                                }

                                probs /= samples.rows();
                            }
                        }

                        for (int i = 0; i < k; i++) {
                            this->update_estimates(
                                i, data_point_idx, time_step, probs[i]);
                        }
                    }
                    else {
                        double prob;
                        double low = this->estimates.assignment[0];
                        // TODO - change this when range is implemented
                        double high = low;

                        if (marginals.has_data_for(this->estimates.label)) {
                            prob = marginals[this->estimates.label](0, 0).col(
                                t)((int) low);
                        }
                        else {
                            if (time_step < metadata->get_initial_time_step()) {
                                prob = get_prior(node)(low);
                            }
                            else {
                                const Tensor3 samples_tensor =
                                    particles[this->estimates.label];
                                prob = this->get_probability_in_range(
                                    samples_tensor.col(t), low, high);
                            }
                        }
                        this->update_estimates(
                            0, data_point_idx, time_step, prob);
                    }

                    time_step++;
                }
            }
            else {
                const Tensor3 samples_tensor =
                    projected_particles[this->estimates.label];

                if (this->estimates.assignment.size() == 0) {
                    // Compute estimates for each value the node can take
                    // (assuming the node's distribution is discrete)
                    int k = this->get_model()->get_cardinality_of(
                        this->estimates.label);

                    for (int i = 0; i < k; i++) {
                        double prob = this->get_probability_in_range(
                            samples_tensor, i, i);
                        this->update_estimates(
                            i, data_point_idx, time_step, prob);
                    }
                }
                else {
                    double low = this->estimates.assignment[0];
                    // TODO - change this when range is implemented
                    double high = low;
                    double prob = this->get_probability_in_range(
                        samples_tensor, low, high);
                    this->update_estimates(0, data_point_idx, time_step, prob);
                }
            }
        }

        double SamplerEstimator::get_probability_in_range(
            const Tensor3& samples, double low, double high) const {

            Eigen::VectorXi matches =
                Eigen::VectorXi::Ones(samples.get_shape()[1]);
            double probability;
            for (int i = 0; i < samples.get_shape()[0]; i++) {
                Eigen::MatrixXd matrix = samples(i, 0);

                if (low == high) {
                    matches =
                        (matches.array() ==
                         ((matrix.array() == low).cast<int>().rowwise().sum() >
                          0)
                             .cast<int>())
                            .cast<int>();
                }
                else {
                    Eigen::VectorXi matches_low =
                        (matches.array() ==
                         ((matrix.array() >= low).cast<int>().rowwise().sum() >
                          0)
                             .cast<int>())
                            .cast<int>();

                    Eigen::VectorXi matches_high =
                        (matches.array() ==
                         ((matrix.array() <= high).cast<int>().rowwise().sum() >
                          0)
                             .cast<int>())
                            .cast<int>();
                    matches = (matches_low.array() == matches_high.array())
                                  .cast<int>();
                }
            }

            probability = matches.cast<double>().mean();
            return probability;
        }

        void SamplerEstimator::update_estimates(int estimates_idx,
                                                int data_point_idx,
                                                int time_step,
                                                double probability) {
            auto& estimates_matrix =
                this->estimates.estimates.at(estimates_idx);
            int new_rows =
                max(data_point_idx + 1, (int)estimates_matrix.rows());
            int new_cols = max(time_step + 1, (int)estimates_matrix.cols());
            estimates_matrix.conservativeResizeLike(
                Eigen::MatrixXd::Constant(new_rows, new_cols, NO_OBS));
            estimates_matrix(data_point_idx, time_step) = probability;
        }

        void SamplerEstimator::update_custom_data(int estimates_idx,
                                                  int data_point_idx,
                                                  int time_step,
                                                  double probability) {
            auto& estimates_matrix =
                this->estimates.custom_data.at(estimates_idx);
            int new_rows =
                max(data_point_idx + 1, (int)estimates_matrix.rows());
            int new_cols = max(time_step + 1, (int)estimates_matrix.cols());
            estimates_matrix.conservativeResizeLike(
                Eigen::MatrixXd::Constant(new_rows, new_cols, NO_OBS));
            estimates_matrix(data_point_idx, time_step) = probability;
        }

        void SamplerEstimator::prepare_for_the_next_data_point() const {
            // No default preparation needed
        }

        bool SamplerEstimator::is_binary_on_prediction() const {
            // A positive horizon is used in the estimates but we are not
            // constrained to a binary problem.
            return false;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void SamplerEstimator::set_fixed_steps(
            const unordered_set<int>& fixed_steps) {
            SamplerEstimator::fixed_steps = fixed_steps;
        }

    } // namespace model
} // namespace tomcat
