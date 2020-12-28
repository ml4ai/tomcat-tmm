#include "RandomVariableNode.h"
#include <algorithm>
#include <fmt/format.h>
#include <iterator>
#include <stdexcept>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        RandomVariableNode::RandomVariableNode(
            shared_ptr<NodeMetadata>& metadata, int time_step)
            : Node(metadata), time_step(time_step) {}

        RandomVariableNode::RandomVariableNode(
            shared_ptr<NodeMetadata>&& metadata, int time_step)
            : Node(move(metadata)), time_step(time_step) {}

        RandomVariableNode::~RandomVariableNode() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        RandomVariableNode::RandomVariableNode(const RandomVariableNode& node) {
            this->copy_node(node);
        }

        RandomVariableNode&
        RandomVariableNode::operator=(const RandomVariableNode& node) {
            this->copy_node(node);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void RandomVariableNode::copy_node(const RandomVariableNode& node) {
            this->metadata = node.metadata;
            this->time_step = node.time_step;
            this->assignment = node.assignment;
            this->cpd_templates = node.cpd_templates;
            this->cpd = node.cpd;
            this->frozen = node.frozen;
            this->parents = node.parents;
            this->children = node.children;
        }

        string RandomVariableNode::get_description() const {

            if (this->assignment.size() == 1) {
                stringstream assignment_string;
                assignment_string << this->assignment;

                return fmt::format("RV({}, {}, {})",
                                   this->metadata->get_label(),
                                   this->time_step,
                                   assignment_string.str());
            }
            else {
                stringstream assignment_string;
                assignment_string << this->assignment.transpose();

                return fmt::format("RV({}, {}, [{}])",
                                   this->metadata->get_label(),
                                   this->time_step,
                                   assignment_string.str());
            }
        }

        unique_ptr<Node> RandomVariableNode::clone() const {
            unique_ptr<RandomVariableNode> new_node =
                make_unique<RandomVariableNode>(*this);
            new_node->metadata = make_shared<NodeMetadata>(*this->metadata);
            new_node->clone_cpd_templates();
            new_node->clone_cpd();
            return new_node;
        }

        void RandomVariableNode::clone_cpd_templates() {
            for (auto& mapping : this->cpd_templates) {
                mapping.second = mapping.second->clone();
            }
        }

        void RandomVariableNode::clone_cpd() {
            if (this->cpd != nullptr) {
                this->cpd = this->cpd->clone();
            }
        }

        string RandomVariableNode::get_timed_name() const {
            return this->metadata->get_timed_name(this->time_step);
        }

        void RandomVariableNode::reset_cpd_updated_status() {
            this->cpd->reset_updated_status();
        }

        void RandomVariableNode::update_cpd_templates_dependencies(
            NodeMap& parameter_nodes_map, int time_step) {
            for (auto& mapping : this->cpd_templates) {
                if (!mapping.second->is_updated()) {
                    mapping.second->update_dependencies(parameter_nodes_map,
                                                        time_step);
                }
            }
        }

        Eigen::MatrixXd
        RandomVariableNode::sample(shared_ptr<gsl_rng> random_generator,
                                   const vector<shared_ptr<Node>>& parent_nodes,
                                   int num_samples,
                                   bool equal_samples) const {

            return this->cpd->sample(
                random_generator, parent_nodes, num_samples, equal_samples);
        }

        Eigen::MatrixXd RandomVariableNode::sample_from_posterior(
            shared_ptr<gsl_rng> random_generator) {

            // Get p(children(node)|node)
            int rows = this->get_size();
            int cols = this->get_metadata()->get_sample_size();
            Eigen::MatrixXd sample(rows, cols);
            // O(min{ctkd, ct(k^p(p-1) + d)})
            Eigen::MatrixXd weights = this->get_posterior_weights();
            //            Eigen::MatrixXd weights = Eigen::MatrixXd::Ones(rows,
            //                this->get_metadata()->get_cardinality());
            if (this->get_metadata()->is_in_plate()) {
                // This assumes this node was previously initialized  and
                // therefore, the number of instances in-plate can be
                // determined by its size (the number of rows in its
                // assignment matrix).
                for (int i = 0; i < rows; i++) { // O(d)
                    sample.row(i) = this->cpd->sample_from_posterior(
                        random_generator, this->parents, i, weights.row(i));
                }
            }
            else {
                sample.row(0) = this->cpd->sample_from_posterior(
                    random_generator, this->parents, 0, weights.row(0));
            }

            return sample;
        }

        Eigen::MatrixXd RandomVariableNode::get_posterior_weights() {
            int rows = this->get_size();
            int cols = this->get_metadata()->get_cardinality();
            Eigen::MatrixXd log_weights = Eigen::MatrixXd::Ones(rows, cols);

            // O(min{ctkd, ct(k^p(p-1) + d)})
            // In the worst case scenario, one node will have ct children. if
            // the node off plate and is parent of in-plate nodes.
            for (auto& child : this->children) {
                shared_ptr<RandomVariableNode> rv_child =
                    dynamic_pointer_cast<RandomVariableNode>(child);

                Eigen::MatrixXd child_weights =
                    rv_child->get_cpd()->get_posterior_weights(
                        rv_child->get_parents(),
                        shared_from_this(),
                        child->get_assignment()); // O(min{kd, k^p(p-1) + d})
                if (this->get_metadata()->is_in_plate()) {
                    // Multiply weights of each one of the assignments
                    // separately.
                    log_weights = (log_weights.array() +
                                   (child_weights.array() + EPSILON).log());
                }
                else { // O(d)
                    // Multiply the weights of each one of the assignments of
                    // the child node. When an off-plate node is
                    // parent of an in-plate node, all of the in-plate
                    // instances of the child node are also considered
                    // children of the off-plate node, and, therefore, their
                    // weights must be multiplied together.
                    Eigen::MatrixXd agg_child_weights =
                        (child_weights.array() + EPSILON).log().rowwise().sum();
                    log_weights =
                        (log_weights.array() + agg_child_weights.array());
                }
            }

            // Unlog and normalize the weights
            log_weights.colwise() -= log_weights.rowwise().maxCoeff();
            log_weights = log_weights.array().exp();
            Eigen::VectorXd sum_per_row = log_weights.rowwise().sum();
            return (log_weights.array().colwise() / sum_per_row.array());
        }

        Eigen::MatrixXd RandomVariableNode::sample_from_conjugacy(
            shared_ptr<gsl_rng> random_generator,
            const vector<shared_ptr<Node>>& parent_nodes,
            int num_samples) const {
            return this->cpd->sample_from_conjugacy(
                random_generator, parent_nodes, num_samples);
        }

        void RandomVariableNode::update_parents_sufficient_statistics() {
            this->cpd->update_sufficient_statistics(this->parents,
                                                    this->assignment);
        }

        void RandomVariableNode::add_to_sufficient_statistics(
            const vector<double>& values) {
            this->cpd->add_to_sufficient_statistics(values);
        }

        void RandomVariableNode::reset_sufficient_statistics() {
            this->cpd->reset_sufficient_statistics();
        }

        void RandomVariableNode::freeze() { RandomVariableNode::frozen = true; }

        void RandomVariableNode::unfreeze() {
            RandomVariableNode::frozen = false;
        }

        void RandomVariableNode::add_cpd_template(shared_ptr<CPD>& cpd) {
            this->cpd_templates[cpd->get_id()] = cpd;
        }

        void RandomVariableNode::add_cpd_template(shared_ptr<CPD>&& cpd) {
            this->cpd_templates[cpd->get_id()] = move(cpd);
        }

        shared_ptr<CPD> RandomVariableNode::get_cpd_for(
            const vector<string>& parent_labels) const {
            string key = this->get_unique_key_from_labels(parent_labels);
            shared_ptr<CPD> cpd;
            if (EXISTS(key, this->cpd_templates)) {
                cpd = this->cpd_templates.at(key);
            }
            else {
                throw invalid_argument(
                    "No CPD found associated with the parents informed.");
            }

            return cpd;
        }

        string RandomVariableNode::get_unique_key_from_labels(
            vector<string> labels) const {
            stringstream ss;
            sort(labels.begin(), labels.end());
            copy(labels.begin(),
                 labels.end(),
                 ostream_iterator<string>(ss, ","));
            return ss.str();
        }

        // ---------------------------------------------------------------------
        // Getters & Setters
        // ---------------------------------------------------------------------
        int RandomVariableNode::get_time_step() const { return time_step; }

        void RandomVariableNode::set_time_step(int time_step) {
            this->time_step = time_step;
        }

        void RandomVariableNode::set_assignment(Eigen::MatrixXd assignment) {
            if (!this->frozen) {
                this->assignment = assignment;
            }
        }

        bool RandomVariableNode::is_frozen() const { return frozen; }

        void RandomVariableNode::set_cpd(const shared_ptr<CPD>& cpd) {
            this->cpd = cpd;
        }
        const shared_ptr<CPD>& RandomVariableNode::get_cpd() const {
            return cpd;
        }

        const vector<std::shared_ptr<Node>>&
        RandomVariableNode::get_parents() const {
            return parents;
        }

        void RandomVariableNode::set_parents(
            const vector<std::shared_ptr<Node>>& parents) {
            this->parents = parents;
        }

        const vector<std::shared_ptr<Node>>&
        RandomVariableNode::get_children() const {
            return children;
        }

        void RandomVariableNode::set_children(
            const vector<std::shared_ptr<Node>>& children) {
            this->children = children;
        }

    } // namespace model
} // namespace tomcat
