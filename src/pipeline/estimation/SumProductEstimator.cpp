#include "SumProductEstimator.h"

#include <iostream>

#include <boost/progress.hpp>

#include "pgm/inference/FactorGraph.h"
#include "pgm/inference/SegmentExpansionFactorNode.h"
#include "pgm/inference/VariableNode.h"
#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        SumProductEstimator::SumProductEstimator(
            const shared_ptr<DynamicBayesNet>& model,
            int inference_horizon,
            const std::string& node_label,
            const Eigen::VectorXd& assignment)
            : Estimator(model, inference_horizon, node_label, assignment) {

            if (inference_horizon > 2) {
                if (assignment.size() == 0) {
                    throw TomcatModelException(
                        "An assignment must be given for estimations with "
                        "inference horizon greater than 0.");
                }
                else if (this->model->get_metadata_of(node_label)
                             ->get_cardinality() != 2) {
                    throw TomcatModelException(
                        "Prediction within a window larger than 1 time step "
                        "ahead is only implemented for nodes with a binary "
                        "distribution.");
                }
            }
        }

        SumProductEstimator::~SumProductEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        SumProductEstimator::SumProductEstimator(
            const SumProductEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->next_time_step = estimator.next_time_step;
        }

        SumProductEstimator&
        SumProductEstimator::operator=(const SumProductEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->next_time_step = estimator.next_time_step;
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void SumProductEstimator::prepare() {
            Estimator::prepare();
            this->next_time_step = 0;
            this->factor_graph =
                FactorGraph::create_from_unrolled_dbn(*this->model);
            this->factor_graph.store_topological_traversal_per_time_step();
        }

        void SumProductEstimator::estimate(const EvidenceSet& new_data) {
            this->estimate_forward_in_time(new_data);
        }

        void SumProductEstimator::estimate_forward_in_time(
            const EvidenceSet& new_data) {

            int total_time = this->next_time_step + new_data.get_time_steps();
            unique_ptr<boost::progress_display> progress;

            if (this->show_progress) {
                cout << this->estimates.label
                     << " (h = " << this->inference_horizon << ")";
                progress = make_unique<boost::progress_display>(total_time);
            }

            for (int t = this->next_time_step; t < total_time; t++) {
                // We expand one time step at a time to store how estimates
                // update at every time step.
                int initial_time_step_in_window =
                    this->variable_subgraph_window
                        ? 0
                        : max(t - this->subgraph_window_size + 1, 0);

                bool any_change;
                do {

                    // Propagate messages forward within a window
                    for (int w = initial_time_step_in_window; w <= t; w++) {
                        // We pass messages in the time slice until they
                        // converge
                        any_change = false;
                        any_change |= this->compute_forward_messages(
                            this->factor_graph, w, new_data);
                    }

                    // Propagate messages backwards within a window
                    for (int w = t; w >= initial_time_step_in_window; w--) {
                        any_change |= this->compute_backward_messages(
                            this->factor_graph, w, new_data, t);
                    }
                } while (any_change);

                Eigen::MatrixXd marginal;

                if (this->inference_horizon > 0) {
                    marginal = this->get_predictions(
                        t, new_data.get_num_data_points());
                }
                else {
                    marginal = this->factor_graph.get_marginal_for(
                        this->estimates.label, t, true);
                }

                Eigen::VectorXd no_obs = Eigen::MatrixXd::Constant(
                    new_data.get_num_data_points(), 1, NO_OBS);

                if (this->estimates.assignment.size() == 0) {
                    // For each possible discrete value the node can
                    // take, we compute the probability estimate.
                    // marginal.cols() is the cardinality of the node.
                    int cardinality =
                        this->model->get_cardinality_of(this->estimates.label);

                    for (int col = 0; col < cardinality; col++) {
                        Eigen::VectorXd estimates_in_time_step = no_obs;
                        if (marginal.size() > 0) {
                            estimates_in_time_step = marginal.col(col);
                        }
                        this->add_column_to_estimates(estimates_in_time_step,
                                                      col);
                    }
                }
                else {
                    int discrete_assignment = this->estimates.assignment[0];
                    Eigen::VectorXd estimates_in_time_step = no_obs;
                    if (marginal.size() > 0) {
                        if(marginal.cols() == 1) {
                            estimates_in_time_step =
                                marginal.col(0);
                        } else {
                            estimates_in_time_step =
                                marginal.col(discrete_assignment);
                        }
                    }

                    this->add_column_to_estimates(estimates_in_time_step);
                }

                if (this->show_progress) {
                    ++(*progress);
                }
            }

            this->next_time_step += new_data.get_time_steps();
        }

        bool SumProductEstimator::compute_forward_messages(
            const FactorGraph& factor_graph,
            int time_step,
            const EvidenceSet& new_data) {

            bool any_change = false;
            for (auto& node :
                 factor_graph.get_vertices_topological_order_in(time_step)) {

                auto parent_nodes =
                    factor_graph.get_parents_of(node, time_step);

                if (!node->is_factor() && time_step >= this->next_time_step) {
                    // Do not temper with data set in previous time steps.
                    shared_ptr<VariableNode> variable_node =
                        dynamic_pointer_cast<VariableNode>(node);
                    if (new_data.has_data_for(node->get_label())) {
                        // Column of the data matrix that contains data for the
                        // time step being processed.
                        Eigen::MatrixXd node_data =
                            new_data[node->get_label()](0, 0);
                        Eigen::MatrixXd data_in_time_step =
                            node_data.col(time_step - this->next_time_step);

                        variable_node->set_data_at(time_step,
                                                   data_in_time_step);
                    }
                    else {
                        variable_node->erase_data_at(time_step);
                    }
                }

                if (parent_nodes.empty() && node->is_factor()) {
                    // This vertex is a factor that represents the prior
                    // probability of the factor's child node.
                    int num_data_points =
                        max(1, new_data.get_num_data_points());
                    if (node->is_segment()) {
                        VariableNode segment_prior(
                            MessageNode::PRIOR_NODE_LABEL, time_step);
                        int cardinality =
                            dynamic_pointer_cast<SegmentExpansionFactorNode>(
                                node)
                                ->get_timed_node_cardinality();
                        node->set_incoming_message_from(
                            make_shared<VariableNode>(segment_prior),
                            time_step,
                            time_step,
                            Tensor3::ones(num_data_points, 1, cardinality),
                            MessageNode::Direction::forward);
                    }
                    else {
                        node->set_incoming_message_from(
                            MessageNode::PRIOR_NODE_LABEL,
                            time_step,
                            time_step,
                            Tensor3::ones(1, num_data_points, 1),
                            MessageNode::Direction::forward);
                    }
                }
                else {
                    for (const auto& [parent_node, transition] : parent_nodes) {
                        int parent_incoming_messages_time_step = time_step;
                        if (transition) {
                            // If it's a node that links nodes in different time
                            // steps, the messages that arrive to this parent
                            // node comes from the table of messages of the
                            // previous time step.
                            parent_incoming_messages_time_step = time_step - 1;
                        }

                        //                        LOG("Forward");
                        //                        stringstream ss;
                        //                        ss << MessageNode::get_name(
                        //                                  parent_node->get_label(),
                        //                                  parent_incoming_messages_time_step)
                        //                           << " -> "
                        //                           <<
                        //                           MessageNode::get_name(node->get_label(),
                        //                                                    time_step);
                        //                        LOG(ss.str());

                        Tensor3 message = parent_node->get_outward_message_to(
                            node,
                            parent_incoming_messages_time_step,
                            time_step,
                            MessageNode::Direction::forward);

                        //                        LOG(message);

                        any_change |= node->set_incoming_message_from(
                            parent_node,
                            parent_incoming_messages_time_step,
                            time_step,
                            message,
                            MessageNode::Direction::forward);
                    }
                }
            }

            return any_change;
        }

        bool SumProductEstimator::compute_backward_messages(
            const FactorGraph& factor_graph,
            int time_step,
            const EvidenceSet& new_data,
            int last_time_step) {

            bool any_change = false;
            for (auto& node : factor_graph.get_vertices_topological_order_in(
                     time_step, false)) {

                auto child_nodes =
                    factor_graph.get_children_of(node, time_step);

                int num_data_points = max(1, new_data.get_num_data_points());

                if (child_nodes.empty()) {
                    // This vertex is a leaf in the factor graph for that time
                    // step (it can have a child in the next time step, which
                    // should not be considered at this point). It's single
                    // incoming bottom up message is a vector of ones.
                    // Implemented as a matrix so that messages for multiple
                    // data sets can be processes at once.
                    int cardinality = dynamic_pointer_cast<VariableNode>(node)
                                          ->get_cardinality();

                    node->set_incoming_message_from(
                        MessageNode::END_NODE_LABEL,
                        time_step,
                        time_step,
                        Tensor3::ones(1, num_data_points, cardinality),
                        MessageNode::Direction::backwards);
                }
                else {
                    for (const auto& [child_node, transition] : child_nodes) {
                        int child_incoming_messages_time_step = time_step;
                        if (transition) {
                            // If it's a node that links nodes in different time
                            // steps, the messages that arrive to this parent
                            // node comes from the table of messages of the
                            // previous time step.
                            child_incoming_messages_time_step = time_step + 1;
                        }

                        //                        LOG("Backward");
                        //                        stringstream ss;
                        //                        ss <<
                        //                        MessageNode::get_name(node->get_label(),
                        //                                                    time_step)
                        //                           << " <- "
                        //                           << MessageNode::get_name(
                        //                                  child_node->get_label(),
                        //                                  child_incoming_messages_time_step);
                        //                        LOG(ss.str());

                        Tensor3 message;
                        if (child_incoming_messages_time_step >
                            last_time_step) {
                            if (node->is_segment()) {
                                continue;
                            }

                            // Boundary nodes are never factor nodes. Therefore,
                            // the cast will work.
                            int cardinality =
                                dynamic_pointer_cast<VariableNode>(node)
                                    ->get_cardinality();

                            message =
                                Tensor3::ones(1, num_data_points, cardinality);
                        }
                        else {
                            message = child_node->get_outward_message_to(
                                node,
                                child_incoming_messages_time_step,
                                time_step,
                                MessageNode::Direction::backwards);
                        }

                        //                        LOG(message);

                        any_change |= node->set_incoming_message_from(
                            child_node,
                            child_incoming_messages_time_step,
                            time_step,
                            message,
                            MessageNode::Direction::backwards);
                    }
                }
            }

            return any_change;
        }

        Eigen::MatrixXd
        SumProductEstimator::get_predictions(int time_step,
                                             int num_data_points) {

            const string& node_label = this->estimates.label;

            Eigen::MatrixXd estimated_probabilities;
            if (this->inference_horizon == 1) {
                EvidenceSet empty_set;
                this->compute_forward_messages(
                    this->factor_graph, time_step + 1, empty_set);

                estimated_probabilities = this->factor_graph.get_marginal_for(
                    this->estimates.label, time_step + 1, true);
            }
            else {
                // To compute the probability of observing at least one
                // occurence of an assignment in the inference_horizon, we
                // compute the complement of not observing the assignment in
                // any of the time steps in the inference horizon.
                // This implementation only supports prediction in horizons
                // larger than 1 for nodes with binary distribution. The
                // following codes, assumes that to work properly.
                int discrete_assignment = this->estimates.assignment[0];
                Eigen::MatrixXd opposite_assignment =
                    Eigen::MatrixXd::Zero(num_data_points, 1);
                opposite_assignment.col(discrete_assignment) =
                    Eigen::VectorXd::Constant(num_data_points, 1 - discrete_assignment);

                EvidenceSet horizon_data;
                horizon_data.add_data(node_label, Tensor3(opposite_assignment));

                int curr_next_time_step = this->next_time_step;
                for (int t = time_step + 1;
                     t < time_step + this->inference_horizon;
                     t++) {
                    // Simulate new data coming and compute estimates in a
                    // regular way.
                    this->next_time_step = t;

                    int initial_time_step_in_window =
                        this->variable_subgraph_window
                            ? 0
                            : max(t - this->subgraph_window_size + 1, 0);
                    bool any_change;
                    do {
                        any_change = false;
                        any_change |= this->compute_forward_messages(
                            this->factor_graph, t, horizon_data);

                        any_change |= this->compute_backward_messages(
                            this->factor_graph, t, horizon_data, t);

                    } while (any_change);

                    Eigen::MatrixXd marginal =
                        factor_graph.get_marginal_for(node_label, t, true);

                    if (estimated_probabilities.size() == 0) {
                        estimated_probabilities =
                            marginal.col(1 - discrete_assignment).array();
                    }
                    else {
                        estimated_probabilities =
                            estimated_probabilities.array() *
                            marginal.col(1 - discrete_assignment).array();
                    }
                }

                // We only need to pass the message forward once in the last
                // time step to collect the final probability.
                EvidenceSet empty_set;
                this->compute_forward_messages(this->factor_graph,
                                               time_step +
                                                   this->inference_horizon,
                                               empty_set);

                Eigen::MatrixXd marginal = this->factor_graph.get_marginal_for(
                    this->estimates.label,
                    time_step + this->inference_horizon,
                    true);

                estimated_probabilities = estimated_probabilities.array() *
                                          marginal.col(1 - discrete_assignment).array();

                estimated_probabilities = 1 - estimated_probabilities.array();

                // Adjust the time counter back to it's original position.
                this->next_time_step = curr_next_time_step;
                this->factor_graph.erase_incoming_messages_beyond(time_step);
            }

            return estimated_probabilities;
        }

        void SumProductEstimator::add_column_to_estimates(
            const Eigen::VectorXd new_column, int index) {

            if (this->estimates.estimates.size() < index + 1) {
                this->estimates.estimates.push_back(Eigen::MatrixXd(0, 0));
            }
            matrix_hstack(this->estimates.estimates[index], new_column);
        }

        void SumProductEstimator::get_info(nlohmann::json& json) const {
            json["name"] = this->get_name();
            json["inference_horizon"] = this->inference_horizon;
        }

        string SumProductEstimator::get_name() const { return "sum-product"; }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        void SumProductEstimator::set_subgraph_window_size(
            int subgraph_window_size) {
            this->subgraph_window_size = subgraph_window_size;
        }

        void SumProductEstimator::set_variable_window(bool variable_window) {
            this->variable_subgraph_window = variable_window;
        }

    } // namespace model
} // namespace tomcat
