#include "Estimator.h"

#include <sstream>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Estimator::Estimator() {}

        Estimator::Estimator(const shared_ptr<DynamicBayesNet>& model,
                             int inference_horizon,
                             const std::string& node_label,
                             const Eigen::VectorXd& assignment)
            : model(model), inference_horizon(inference_horizon),
              compound(false) {

            const auto& metadata = model->get_metadata_of(node_label);
            if (!metadata->is_replicable() && inference_horizon > 0) {
                throw TomcatModelException("Inference horizon for "
                                           "non-replicable nodes can only be "
                                           "0.");
            }

            if (assignment.size() == 0 && metadata->get_cardinality() <= 1) {
                throw TomcatModelException(
                    "An assignment has to be provided for nodes with "
                    "continuous distributions.");
            }

            this->estimates.label = node_label;
            this->estimates.assignment = assignment;
            this->cumulative_estimates.label = node_label;
            this->cumulative_estimates.assignment = assignment;
        }

        Estimator::Estimator(const shared_ptr<DynamicBayesNet>& model)
            : model(model), compound(true) {}

        Estimator::~Estimator() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Estimator::copy_estimator(const Estimator& estimator) {
            this->model = estimator.model;
            this->training_data = estimator.training_data;
            this->estimates = estimator.estimates;
            this->cumulative_estimates = estimator.cumulative_estimates;
            this->inference_horizon = estimator.inference_horizon;
        }

        NodeEstimates Estimator::get_estimates_at(int time_step) const {
            if (this->estimates.estimates[0].cols() <= time_step) {
                stringstream ss;
                ss << "The chosen estimator can only calculate estimates "
                      "up to time step "
                   << time_step;
                throw out_of_range(ss.str());
            }

            NodeEstimates sliced_estimates;
            sliced_estimates.label = this->estimates.label;
            sliced_estimates.assignment = this->estimates.assignment;
            for (const auto& estimates_per_assignment :
                 this->estimates.estimates) {
                sliced_estimates.estimates.push_back(
                    estimates_per_assignment.col(time_step));
            }

            return sliced_estimates;
        }

        void Estimator::prepare() {
            // Clear estimates so they can be recalculated over the new
            // training data in the next call to the function estimate.
            this->estimates.estimates.clear();

            int k = 1;
            if(this->estimates.assignment.size() == 0) {
                k = this->model->get_cardinality_of(this->estimates.label);
            }
            this->estimates.estimates.resize(k);
        }

        void Estimator::keep_estimates() {
            this->cumulative_estimates.estimates.push_back({});
            int i = this->cumulative_estimates.estimates.size() - 1;
            for (const auto& estimate : this->estimates.estimates) {
                this->cumulative_estimates.estimates[i].push_back(estimate);
            }
        }

        void Estimator::cleanup() {
            this->estimates.estimates.clear();
            this->cumulative_estimates.estimates.clear();
        }

        bool Estimator::is_computing_estimates_for(
            const std::string& node_label) const {
            return this->estimates.label == node_label;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        NodeEstimates Estimator::get_estimates() const {
            return this->estimates;
        }

        CumulativeNodeEstimates Estimator::get_cumulative_estimates() const {
            return this->cumulative_estimates;
        }

        void Estimator::set_training_data(const EvidenceSet& training_data) {
            this->training_data = training_data;
        }

        int Estimator::get_inference_horizon() const {
            return inference_horizon;
        }

        const shared_ptr<DynamicBayesNet>& Estimator::get_model() const {
            return model;
        }

        void Estimator::set_show_progress(bool show_progress) {
            this->show_progress = show_progress;
        }

        bool Estimator::is_compound() const { return compound; }

        vector<shared_ptr<const Estimator>>
        Estimator::get_base_estimators() const {
            vector<shared_ptr<const Estimator>> base_estimators;
            base_estimators.push_back(shared_from_this());
            return base_estimators;
        }
    } // namespace model
} // namespace tomcat
