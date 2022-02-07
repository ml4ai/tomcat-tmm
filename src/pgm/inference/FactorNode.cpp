#include "FactorNode.h"

#include "distribution/Categorical.h"
#include "pgm/inference/VariableNode.h"

using namespace std;

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        FactorNode::FactorNode() : dynamic(false) {}

        FactorNode::FactorNode(const string& label, int time_step)
            : MessageNode(compose_label(label), time_step), dynamic(false) {}

        FactorNode::FactorNode(const string& label,
                               int time_step,
                               const Eigen::MatrixXd& probability_table,
                               const CPD::TableOrderingMap& ordering_map,
                               const string& cpd_main_node_label)
            : MessageNode(compose_label(label), time_step), dynamic(false) {

            this->original_potential.potential = PotentialFunction(
                ordering_map, probability_table, cpd_main_node_label);
            this->create_potential_function_rotations();
            this->working_potential = this->original_potential;
        }

        FactorNode::FactorNode(const string& label,
                               int time_step,
                               const DistributionPtrVec& distributions,
                               const CPD::TableOrderingMap& ordering_map,
                               const string& cpd_main_node_label)
            : MessageNode(compose_label(label), time_step), dynamic(true) {

            if (instanceof <Categorical>(distributions[0].get())) {
                throw TomcatModelException(
                    "This implementation of the SumProduct only supports "
                    "static categorical distributions. Use the alternative "
                    "constructor for this purpose.");
            }

            this->original_potential.potential = PotentialFunction(
                ordering_map, distributions, cpd_main_node_label);
            // The CPD table in this case will store indices of the
            // distributions to be evaluated at runtime.
            Eigen::MatrixXd index_table(distributions.size(), 1);
            for (int i = 0; i < distributions.size(); i++) {
                index_table(i, 0) = i;
            }
            this->original_potential.potential.probability_table = index_table;
            this->create_potential_function_rotations();
            this->working_potential = this->original_potential;
        }

        FactorNode::~FactorNode() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        FactorNode::FactorNode(const FactorNode& node) {
            this->copy_node(node);
        }

        FactorNode& FactorNode::operator=(const FactorNode& node) {
            this->copy_node(node);
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        string FactorNode::compose_label(const string& variable_node_label) {
            return "f:" + variable_node_label;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void FactorNode::create_potential_function_rotations() {

            for (const auto& [node_label, ordering] :
                 this->original_potential.potential.ordering_map) {

                if (node_label ==
                    this->original_potential.potential.main_node_label) {
                    // This is the original version.
                    continue;
                }

                Eigen::MatrixXd new_matrix = this->rotate_table(
                    this->original_potential.potential.probability_table,
                    ordering.cardinality,
                    ordering.right_cumulative_cardinality);

                PotentialFunction new_function;
                new_function.probability_table = move(new_matrix);
                new_function.main_node_label = node_label;
                // Create a new ordering map and replace the parent node's label
                // by the main node name, that happens to be the child of this
                // factor node;
                CPD::TableOrderingMap new_map =
                    this->original_potential.potential.ordering_map;
                string prev_main_node_label =
                    this->original_potential.potential.main_node_label;
                if (EXISTS(prev_main_node_label, new_map)) {
                    new_function.duplicate_key = prev_main_node_label;
                    // Adds * to the key to differentiate it from the already
                    // existing key with the same label. E.g. transition matrix
                    // between one node with a label and another node with the
                    // same label but in the future.
                    prev_main_node_label =
                        PotentialFunction::get_alternative_key_label(
                            prev_main_node_label);
                }
                auto map_entry = new_map.extract(node_label);
                map_entry.key() = prev_main_node_label;
                new_map.insert(move(map_entry));

                new_function.ordering_map = move(new_map);

                this->original_potential
                    .node_label_to_rotated_potential[node_label] = new_function;
            }
        }

        Eigen::MatrixXd
        FactorNode::rotate_table(const Eigen::MatrixXd& table,
                                 int main_node_cardinality,
                                 int right_cumulative_cardinality) const {

            int num_rows = table.rows();
            int num_cols = table.cols();

            int block_rows = right_cumulative_cardinality;
            int block_size = block_rows * num_cols;
            int num_blocks = num_rows / block_rows;

            int new_num_rows = (num_rows * num_cols) / main_node_cardinality;
            int new_num_cols = main_node_cardinality;
            Eigen::MatrixXd rotated_table(new_num_rows, new_num_cols);
            int row = 0;
            int col = 0;
            for (int b = 0; b < num_blocks; b++) {
                Eigen::MatrixXd block =
                    table.block(b * block_rows, 0, block_rows, num_cols);
                Eigen::VectorXd vector =
                    Eigen::Map<Eigen::VectorXd>(block.data(), block.size());

                // Stack the vector column-wise in the new matrix.
                rotated_table.block(row * block_size, col, block_size, 1) =
                    vector;
                col = (col + 1) % new_num_cols;
                if (col == 0) {
                    row++;
                }
            }

            return rotated_table;
        }

        void FactorNode::copy_node(const FactorNode& node) {
            MessageNode::copy_node(node);
            this->original_potential = node.original_potential;
            this->working_potential = node.working_potential;
            this->block_forward_message = node.block_forward_message;
            this->dynamic = node.dynamic;
            this->incoming_continuous_messages_per_time_slice =
                node.incoming_continuous_messages_per_time_slice;
        }

        Tensor3 FactorNode::get_outward_message_to(
            const shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            int target_time_step,
            Direction direction) const {

            if ((direction == Direction::forward &&
                 this->block_forward_message) ||
                (direction == Direction::backwards &&
                 this->block_backward_message)) {
                return {};
            }

            PotentialFunction potential_function;
            if (direction == Direction::forward) {
                potential_function = this->working_potential.potential;
            }
            else {
                potential_function =
                    this->working_potential.node_label_to_rotated_potential.at(
                        template_target_node->get_label());
            }

            if (this->dynamic) {
                return this->get_dynamic_factor_outward_message_to(
                    potential_function,
                    template_target_node,
                    template_time_step,
                    target_time_step,
                    direction);
            }
            else {
                return this->get_static_factor_outward_message_to(
                    potential_function,
                    template_target_node,
                    template_time_step,
                    target_time_step,
                    direction);
            }
        }

        Tensor3 FactorNode::get_cartesian_tensor(
            const std::vector<Tensor3>& tensors) const {
            // Each row in the message matrix contains the message for one data
            // point. We need to perform the following operation individually
            // for each data point before multiplying by the potential function
            // matrix. For instance:
            // m1 = [a, b, c]
            // m2 = [d, e, f]
            // temp_vector = [ad, ae, af, bd, be, bf, cd, ce, cf]
            int depths = tensors[0].get_shape()[0];
            vector<Eigen::MatrixXd> new_tensor(depths);
            int rows = tensors[0].get_shape()[1];
            for (int depth = 0; depth < depths; depth++) {
                Eigen::MatrixXd temp_matrix = Eigen::MatrixXd::Zero(rows, 0);

                for (int row = 0; row < rows; row++) {
                    Eigen::MatrixXd message = tensors.at(0)(depth, 0);
                    Eigen::VectorXd temp_vector = message.row(row);

                    for (int i = 1; i < tensors.size(); i++) {
                        message = tensors.at(i)(depth, 0);
                        Eigen::MatrixXd cartesian_product =
                            temp_vector * message.row(row);
                        cartesian_product.transposeInPlace();
                        // Flatten the matrix
                        temp_vector = Eigen::Map<Eigen::VectorXd>(
                            cartesian_product.data(), cartesian_product.size());
                    }

                    if (temp_matrix.cols() < temp_vector.size()) {
                        temp_matrix.conservativeResize(rows,
                                                       temp_vector.size());
                    }
                    temp_matrix.row(row) = temp_vector;
                }

                new_tensor[depth] = temp_matrix;
            }

            if (new_tensor.empty()) {
                return {1};
            }

            return {new_tensor};
        }

        Tensor3 FactorNode::get_static_factor_outward_message_to(
            const PotentialFunction& potential_function,
            const shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            int target_time_step,
            Direction direction) const {

            // To achieve the correct indexing when multiplying incoming
            // messages by the potential function matrix, the messages have to
            // be multiplied following the CPD order defined for parent nodes.
            vector<Tensor3> messages_in_order =
                this->get_incoming_messages_in_order(
                    template_target_node->get_label(),
                    template_time_step,
                    target_time_step,
                    potential_function);

            Eigen::MatrixXd outward_message;

            if (!messages_in_order.empty()) {
                Eigen::MatrixXd indexing_probs =
                    this->get_cartesian_tensor(messages_in_order)(0, 0);

                // This will marginalize the incoming nodes by summing the rows.
                outward_message =
                    indexing_probs * potential_function.probability_table;

                // Normalize the message
                Eigen::VectorXd sum_per_row = outward_message.rowwise().sum();
                outward_message =
                    (outward_message.array().colwise() / sum_per_row.array())
                        .matrix();
            }

            return {outward_message};
        }

        Tensor3 FactorNode::get_dynamic_factor_outward_message_to(
            const PotentialFunction& potential_function,
            const shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            int target_time_step,
            Direction direction) const {

            if (direction == Direction::forward) {
                // This algorithm only supports models with continuous variables
                // as leaves. A forward message cannot be computed analytically.
                // But that is not a problem considering that leaf nodes to not
                // have children such that the messages produced in a forward
                // pass can reach to. Therefore, we can ignore them.
                return {};
            }
            // In the backward pass we can compute the message from a
            // continuous distribution because we have evidence for the leaf
            // node. Then we can compute p(evidence|parents) and replace the
            // continuous message with the pdfs.
            vector<Tensor3> messages_in_order =
                this->get_incoming_messages_in_order(
                    template_target_node->get_label(),
                    template_time_step,
                    target_time_step,
                    potential_function);
            Eigen::MatrixXd indexing_probs =
                this->get_cartesian_tensor(messages_in_order)(0, 0);

            auto& distributions =
                this->working_potential.potential.distributions;
            Eigen::MatrixXd pdf_table = potential_function.probability_table;

            if (EXISTS(template_time_step,
                       this->incoming_continuous_messages_per_time_slice)) {

                Eigen::MatrixXd continuous_values =
                    this->incoming_continuous_messages_per_time_slice.at(
                        template_time_step)(0, 0);

                Eigen::MatrixXd outward_message(continuous_values.rows(),
                                                pdf_table.cols());

                // For each data point we compute their pdf and populate a
                // CPD table using the index table associated to the
                // potential function in use.
                for (int data_point = 0; data_point < continuous_values.rows();
                     data_point++) {
                    for (int i = 0; i < pdf_table.rows(); i++) {
                        for (int j = 0; j < pdf_table.cols(); j++) {
                            int distribution_idx =
                                (int)potential_function.probability_table(i, j);
                            pdf_table(i, j) =
                                distributions[distribution_idx]->get_pdf(
                                    continuous_values.row(data_point));
                        }
                    }

                    outward_message.row(data_point) =
                        indexing_probs * pdf_table;
                    outward_message.row(data_point).array() /=
                        outward_message.row(data_point).sum();
                }

                return {outward_message};
            }
            else {
                PotentialFunction ignore_evidence = potential_function;
                ignore_evidence.probability_table.fill(1);
                return this->get_static_factor_outward_message_to(
                    potential_function,
                    template_target_node,
                    template_time_step,
                    target_time_step,
                    direction);
            }

            return {};
        }

        vector<Tensor3> FactorNode::get_incoming_messages_in_order(
            const string& ignore_label,
            int template_time_step,
            int target_time_step,
            const PotentialFunction& potential_function) const {

            int num_messages = 0;
            vector<Tensor3> messages_in_order;

            if (EXISTS(template_time_step,
                       this->incoming_messages_per_time_slice)) {

                num_messages = this->incoming_messages_per_time_slice
                                   .at(template_time_step)
                                   .size();
                messages_in_order.resize(num_messages);
                int added_duplicate_key_order = -1;
                int added_duplicate_key_time_step = -1;

                MessageContainer message_container =
                    this->incoming_messages_per_time_slice.at(
                        template_time_step);

                for (const auto& [incoming_node_name, incoming_message] :
                     message_container.node_name_to_messages) {

                    if (MessageNode::is_prior(incoming_node_name)) {
                        // No parents. There's only one incoming message.
                        messages_in_order.resize(1);
                        messages_in_order[0] = incoming_message;
                        break;
                    }
                    else {
                        // Ignore messages that come from a specific node.
                        // Because messages can go forward and backwards in the
                        // graph, when computing the messages that go towards a
                        // target node via this node, the messages that arrive
                        // in this node from that same target have to be
                        // ignored.
                        if (incoming_node_name ==
                            MessageNode::get_name(ignore_label,
                                                  target_time_step)) {
                            messages_in_order.pop_back();
                            continue;
                        }

                        int order = 0;
                        auto [incoming_node_label, incoming_node_time_step] =
                            MessageNode::strip(incoming_node_name);

                        if (potential_function.duplicate_key ==
                            incoming_node_label) {
                            // The potential function matrix is indexed by
                            // another node with the same label. We need to
                            // detect the correct order of this node somehow. We
                            // simply use the order defined for one of the
                            // labels and adjust later by swapping the orders
                            // when we process the second entry with the same
                            // label.

                            if (added_duplicate_key_order < 0 &&
                                EXISTS(incoming_node_label,
                                       potential_function.ordering_map)) {
                                order = potential_function.ordering_map
                                            .at(incoming_node_label)
                                            .order;
                                added_duplicate_key_order = order;
                                added_duplicate_key_time_step =
                                    incoming_node_time_step;
                            }
                            else {
                                string alternative_key_label =
                                    PotentialFunction::
                                        get_alternative_key_label(
                                            incoming_node_label);
                                order = potential_function.ordering_map
                                            .at(alternative_key_label)
                                            .order;
                                if (incoming_node_time_step <
                                    added_duplicate_key_time_step) {
                                    // This node is in the past with respect to
                                    // the previous node with the same key
                                    // processed. Therefore, they have to change
                                    // position as the node's alternative key
                                    // has to be in the future according to how
                                    // CPDs were defined in this implementation.
                                    // An CPD indexing node is never in the
                                    // future regarding the CPD's main node.
                                    // When CPDs were adjusted in this factor
                                    // node, the main node was swapped with one
                                    // of the indexing nodes and an alternative
                                    // key was created if there was a conflict
                                    // with an already existing label.
                                    // Therefore, nodes with alternative keys
                                    // can never be in the past and we must swap
                                    // the order with the previously processed
                                    // label of the same kind.
                                    messages_in_order[order] = messages_in_order
                                        [added_duplicate_key_order];
                                    order = added_duplicate_key_order;
                                }
                            }
                        }
                        else {
                            order = potential_function.ordering_map
                                        .at(incoming_node_label)
                                        .order;
                        }

                        messages_in_order[order] = incoming_message;
                    }
                }
            }

            return messages_in_order;
        }

        bool FactorNode::is_factor() const { return true; }

        bool FactorNode::is_segment() const { return false; }

        bool
        FactorNode::set_incoming_message_from(const string& source_node_label,
                                              int source_time_step,
                                              int target_time_step,
                                              const Tensor3& message,
                                              Direction direction) {

            if (message.is_empty())
                return false;

            bool changed;
            if (this->dynamic &&
                source_node_label ==
                    this->original_potential.potential.main_node_label) {
                // We store inputs from leaf continuous variables in a different
                // attribute for fast computation of the potential function when
                // passing the pdf backwards.
                this->max_time_step_stored =
                    max(this->max_time_step_stored, target_time_step);

                if (EXISTS(target_time_step,
                           this->incoming_continuous_messages_per_time_slice)) {
                    const Tensor3& curr_msg =
                        this->incoming_continuous_messages_per_time_slice.at(
                            target_time_step);
                    changed = !curr_msg.equals(message);
                }
                else {
                    changed = true;
                }

                this->incoming_continuous_messages_per_time_slice
                    [target_time_step] = message;
            }
            else {
                changed =
                    MessageNode::set_incoming_message_from(source_node_label,
                                                           source_time_step,
                                                           target_time_step,
                                                           message,
                                                           direction);
            }

            return changed;
        }

        void FactorNode::erase_incoming_messages_beyond(int time_step) {
            for (int t = time_step + 1; t <= this->max_time_step_stored; t++) {
                this->incoming_continuous_messages_per_time_slice.erase(t);
            }
            MessageNode::erase_incoming_messages_beyond(time_step);
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void FactorNode::set_block_forward_message(bool block_forward_message) {
            this->block_forward_message = block_forward_message;
        }

        void
        FactorNode::set_block_backward_message(bool block_backward_message) {
            this->block_backward_message = block_backward_message;
        }

    } // namespace model
} // namespace tomcat
