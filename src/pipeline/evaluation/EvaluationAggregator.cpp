#include "EvaluationAggregator.h"

#include <sstream>

#include <boost/progress.hpp>

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        EvaluationAggregator::EvaluationAggregator(METHOD method)
            : method(method) {}

        EvaluationAggregator::~EvaluationAggregator() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void EvaluationAggregator::clear_evaluations() {
            for (auto& evaluations : this->evaluations_per_measure) {
                evaluations.clear();
            }
        }

        void
        EvaluationAggregator::add_measure(const shared_ptr<Measure>& measure) {
            this->measures.push_back(measure);
            this->evaluations_per_measure.push_back({});
            this->aggregations_per_measure.push_back({});
        }

        void EvaluationAggregator::evaluate(const EvidenceSet& test_data) {
            int i = 0;

            cout << "\nEvaluating...\n";
            boost::progress_display progress(this->measures.size());
            for (auto& measure : this->measures) {
                this->evaluations_per_measure[i].push_back(
                    measure->evaluate(test_data));
                i++;
                ++progress;
            }
        }

        void EvaluationAggregator::aggregate() {
            int m = 0;
            for (auto& evaluations_per_execution :
                 this->evaluations_per_measure) {
                if (!evaluations_per_execution.empty()) {
                    // The evaluations are stored in the rows of a single
                    // column. So we loop in a row-wise fashion to aggregate the
                    // values.
                    NodeEvaluationAggregation node_aggregation;
                    node_aggregation.label = evaluations_per_execution[0].label;
                    node_aggregation.assignment =
                        evaluations_per_execution[0].assignment;

                    node_aggregation.aggregated_evaluation =
                        this->compute_aggregation(evaluations_per_execution);
                    this->aggregations_per_measure[m] = move(node_aggregation);
                }
                m++;
            }
        }

        Aggregation EvaluationAggregator::compute_aggregation(
            const vector<NodeEvaluation>& evaluations) const {

            Aggregation aggregation;

            switch (this->method) {
            case METHOD::no_aggregation: {
                for (const auto& evaluation : evaluations) {
                    aggregation.aggregated_values.push_back(
                        evaluation.evaluation);

                    if (evaluation.confusion_matrix.size() > 0) {
                        aggregation.aggregated_confusion_matrices.push_back(
                            evaluation.confusion_matrix);
                    }
                }
                break;
            }
            case METHOD::average: {
                vector<Eigen::MatrixXd> values;
                for (const auto& evaluation : evaluations) {
                    values.push_back(evaluation.evaluation);

                    // Confusion matrices ate never aggregated
                    if (evaluation.confusion_matrix.size() > 0) {
                        aggregation.aggregated_confusion_matrices.push_back(
                            evaluation.confusion_matrix);
                    }
                }

                aggregation.aggregated_values =
                    vector<Eigen::MatrixXd>(1, mean(values));
                aggregation.errors = standard_error(values);
                break;
            }
            default: {
                throw TomcatModelException(
                    "The aggregation method assigned is not a valid one.");
            }
            }

            return aggregation;
        }

        void EvaluationAggregator::get_info(nlohmann::json& json) const {
            json = nlohmann::json::array();
            for (int m = 0; m < this->measures.size(); m++) {
                nlohmann::json json_evaluation;
                this->measures[m]->get_info(json_evaluation["measure"]);
                json_evaluation["method"] = this->get_method_name();

                NodeEvaluationAggregation aggregations =
                    this->aggregations_per_measure[m];

                json_evaluation["node_label"] = aggregations.label;

                json_evaluation["node_assignment"] =
                    to_string(aggregations.assignment);

                json_evaluation["error"] =
                    to_string(aggregations.aggregated_evaluation.errors);

                json_evaluation["aggregated_value"] = nlohmann::json::array();
                for (const auto& aggregated_value :
                     aggregations.aggregated_evaluation.aggregated_values) {
                    json_evaluation["aggregated_value"].push_back(
                        to_string(aggregated_value));
                }

                json_evaluation["confusion_matrices"] = nlohmann::json::array();
                for (const auto& confusion_matrix :
                     aggregations.aggregated_evaluation
                         .aggregated_confusion_matrices) {
                    json_evaluation["confusion_matrices"].push_back(
                        to_string(confusion_matrix));
                }

                json.push_back(json_evaluation);
            }
        }

        string EvaluationAggregator::get_method_name() const {
            string method_name = "";

            switch (this->method) {
            case METHOD::no_aggregation: {
                method_name = "no aggregation";
                break;
            }
            case METHOD::average: {
                method_name = "average";
                break;
            }
            }

            return method_name;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------\

        bool EvaluationAggregator::has_measures() const {
            return !this->measures.empty();
        }

    } // namespace model
} // namespace tomcat
