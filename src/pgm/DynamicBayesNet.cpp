#include "DynamicBayesNet.h"

#include <boost/filesystem.hpp>
#include <boost/graph/topological_sort.hpp>

#include "pgm/JSONModel.h"
#include "pgm/TimerNode.h"
#include "utils/FileHandler.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        DynamicBayesNet::DynamicBayesNet() {}

        DynamicBayesNet::DynamicBayesNet(int num_node_templates) {
            this->node_templates.reserve(num_node_templates);
        }

        DynamicBayesNet::~DynamicBayesNet() {}

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------

        DynamicBayesNet
        DynamicBayesNet::create_from_json(const string& filepath) {
            return create_model_from_json(filepath);
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void DynamicBayesNet::add_node_template(
            const shared_ptr<RandomVariableNode>& node) {
            shared_ptr<RandomVariableNode> node_ptr;

            if (node->get_metadata()->is_timer()) {
                node_ptr = make_shared<TimerNode>(
                    *dynamic_cast<TimerNode*>(node->clone().get()));
                this->exact_inference_allowed = false;
            }
            else {
                node_ptr = make_shared<RandomVariableNode>(
                    *dynamic_cast<RandomVariableNode*>(node->clone().get()));
            }

            this->node_templates.push_back(node_ptr);
        }

        void DynamicBayesNet::unroll(int time_steps, bool force) {
            if (time_steps != this->time_steps || force) {
                this->reset();
                this->expand(time_steps);
            }
        }

        void DynamicBayesNet::expand(int new_time_steps) {
            if (new_time_steps > 0) {
                this->create_vertices_from_nodes(new_time_steps);
                this->create_edges(new_time_steps);
                this->set_timers_to_nodes(new_time_steps);
                this->set_timed_copies_to_nodes(new_time_steps);
                this->update_cpd_templates_dependencies(new_time_steps);
                this->set_parents_children_and_cpd_to_nodes();
                this->time_steps += new_time_steps;
                this->save_topological_list();
            }
        }

        void DynamicBayesNet::reset() {
            this->time_steps = 0;
            this->graph.clear();
            this->name_to_id.clear();
            this->parameter_nodes_map.clear();
            this->label_to_nodes.clear();
            this->nodes.clear();
        }

        void DynamicBayesNet::create_vertices_from_nodes(int new_time_steps) {
            for (const auto& node_template : this->node_templates) {
                const shared_ptr<NodeMetadata>& metadata =
                    node_template->get_metadata();

                // Start from the next time step available because the DBN
                // may have already been unrolled by some time steps
                // previously
                int from_time =
                    max(metadata->get_initial_time_step(), this->time_steps);
                if (metadata->is_replicable()) {
                    int to_time = this->time_steps + new_time_steps - 1;

                    for (int t = from_time; t <= to_time; t++) {
                        this->add_vertex(node_template, t);
                    }
                }
                else {
                    // There's only one copy of this node in the unrolled
                    // DBN
                    int t0 = metadata->get_initial_time_step();
                    if (this->time_steps - 1 < t0 &&
                        t0 <= this->time_steps + new_time_steps - 1) {
                        this->add_vertex(node_template, t0);
                    }
                }
            }
        }

        VertexData DynamicBayesNet::add_vertex(
            const std::shared_ptr<RandomVariableNode>& node_template,
            int time_step) {
            int vertex_id = boost::add_vertex(this->graph);

            VertexData data;
            if (node_template->get_metadata()->is_timer()) {
                data.node = make_shared<TimerNode>(
                    *dynamic_pointer_cast<TimerNode>(node_template));
            }
            else {
                data.node = make_shared<RandomVariableNode>(*node_template);
            }
            data.node->set_time_step(time_step);
            data.label = data.node->get_timed_name();

            // Save mapping between the vertice id and it's name.
            string node_name = data.node->get_timed_name();
            this->name_to_id[node_name] = vertex_id;

            // Include node as a property of the vertex in the graph.
            this->graph[vertex_id] = data;

            // Include node in the list of created nodes
            this->label_to_nodes[data.node->get_metadata()->get_label()]
                .push_back(data.node);
            this->nodes.push_back(data.node);
            if (data.node->get_metadata()->is_parameter()) {
                this->parameter_nodes_map[node_name] = data.node;
            }

            return data;
        }

        void DynamicBayesNet::create_edges(int new_time_steps) {
            for (const auto& node_template : this->node_templates) {
                const shared_ptr<NodeMetadata>& metadata =
                    node_template->get_metadata();
                for (const auto& parent_link : metadata->get_parent_links()) {
                    if (metadata->is_replicable()) {
                        int from_time = max(metadata->get_initial_time_step(),
                                            this->time_steps);
                        int to_time = this->time_steps + new_time_steps - 1;

                        for (int t = from_time; t <= to_time; t++) {
                            this->add_edge(*parent_link.parent_node_metadata,
                                           *metadata,
                                           parent_link.time_crossing,
                                           t);

                            if (metadata->is_timer()) {
                                // Create link between instances of a timer
                                // node over time.
                                this->add_edge(*metadata, *metadata, true, t);
                            }
                        }
                    }
                    else {
                        // There's only one copy of this node in the
                        // unrolled DBN
                        int t0 = metadata->get_initial_time_step();
                        if (this->time_steps - 1 < t0 &&
                            t0 <= this->time_steps + new_time_steps - 1) {
                            this->add_edge(*parent_link.parent_node_metadata,
                                           *metadata,
                                           parent_link.time_crossing,
                                           t0);
                        }
                    }
                }
            }
        }

        void DynamicBayesNet::add_edge(const NodeMetadata& source_node_metadata,
                                       const NodeMetadata& target_node_metadata,
                                       bool time_crossing,
                                       int target_time_step) {

            int parent_time_step = -1;
            if (source_node_metadata.is_replicable()) {
                if (time_crossing) {
                    // A replicable node (source) that shows up at time step
                    // t-1 and is linked to another node (target) that shows
                    // up at time step t.
                    if (source_node_metadata.get_initial_time_step() <=
                        target_time_step - 1) {
                        parent_time_step = target_time_step - 1;
                    }
                }
                else {
                    // A replicable node (source) that shows up at time step
                    // t and is linked to another node (target) that also
                    // shows up at time step t.
                    if (source_node_metadata.get_initial_time_step() <=
                        target_time_step) {
                        parent_time_step = target_time_step;
                    }
                }
            }
            else {
                if (source_node_metadata.is_single_time_link()) {
                    if (time_crossing) {
                        // A non-replicable node (source) that shows up once
                        // at its predefined initial time step (t) and is
                        // linked once to another node (target) that shows
                        // up at time step t.
                        if (source_node_metadata.get_initial_time_step() ==
                            target_time_step - 1) {
                            parent_time_step = target_time_step - 1;
                        }
                    }
                    else {
                        // A non-replicable node (source) that shows up once
                        // at its predefined initial time step (t) and is
                        // linked once to another node (target) that shows
                        // up at time step t+1.
                        if (source_node_metadata.get_initial_time_step() ==
                            target_time_step) {
                            parent_time_step = target_time_step;
                        }
                    }
                }
                else {
                    // A non-replicable node (source) that shows up once at
                    // its predefined initial time step (t) and is linked
                    // to replicas of another node (target) over all time
                    // steps starting at t.
                    if (source_node_metadata.get_initial_time_step() <=
                        target_time_step) {
                        parent_time_step =
                            source_node_metadata.get_initial_time_step();
                    }
                }
            }

            if (parent_time_step >= 0) {
                int source_vertex_id = this->name_to_id.at(
                    source_node_metadata.get_timed_name(parent_time_step));
                int target_vertex_id = this->name_to_id.at(
                    target_node_metadata.get_timed_name(target_time_step));

                boost::add_edge(
                    source_vertex_id, target_vertex_id, this->graph);
            }
        }

        void DynamicBayesNet::set_parents_children_and_cpd_to_nodes() {
            for (auto& node : this->nodes) {
                shared_ptr<RandomVariableNode> rv_node =
                    dynamic_pointer_cast<RandomVariableNode>(node);

                vector<string> parent_labels;
                vector<shared_ptr<Node>> parent_nodes;

                for (const auto& parent_node :
                     this->get_parent_nodes_of(node, true)) {
                    string label = parent_node->get_metadata()->get_label();

                    // Timer nodes do not index CPD tables as other discrete
                    // parent nodes do. They have connections to other nodes
                    // in the DBN only for effects of topological order when
                    // sampling.
                    if (!parent_node->get_metadata()->is_timer()) {
                        parent_labels.push_back(label);
                        parent_nodes.push_back(parent_node);
                    }
                }
                rv_node->set_parents(parent_nodes);

                vector<shared_ptr<Node>> child_nodes =
                    this->get_child_nodes_of(node, false);
                rv_node->set_children(child_nodes);

                // Set node's actual CPD based on the concrete instance of
                // its parents.
                shared_ptr<CPD> cpd = rv_node->get_cpd_for(parent_labels);
                rv_node->set_cpd(cpd);

                // If the node's CPD does not follow a discrete distribution,
                // we cannot use the exact inference method implemented in
                // this library.
                // TODO - allow to use sum product when the continuous
                //  distribution is Gaussian
                if (rv_node->is_continuous()) {
                    this->exact_inference_allowed = false;
                }
            }
        }

        void DynamicBayesNet::set_timers_to_nodes(int num_timed_copies) {
            int to_time = this->time_steps + num_timed_copies - 1;

            for (const auto& node_template : this->node_templates) {
                if (const auto& timer_metadata =
                        node_template->get_metadata()->get_timer_metadata()) {
                    int t0 = this->time_steps;
                    for (int t = t0; t <= to_time; t++) {
                        string node_name =
                            node_template->get_metadata()->get_timed_name(t);
                        int node_id = this->name_to_id[node_name];
                        const auto& node = this->graph[node_id].node;

                        const string& timer_name =
                            timer_metadata->get_timed_name(t);
                        int timer_id = this->name_to_id[timer_name];
                        const auto& timer = dynamic_pointer_cast<TimerNode>(
                            this->graph[timer_id].node);

                        node->set_timer(timer);
                        timer->set_controlled_node(node);
                    }
                }
            }
        }

        void DynamicBayesNet::set_timed_copies_to_nodes(int num_timed_copies) {
            int to_time = this->time_steps + num_timed_copies - 1;

            for (const auto& node_template : this->node_templates) {
                if (node_template->get_metadata()->is_replicable()) {
                    int t0 =
                        node_template->get_metadata()->get_initial_time_step();

                    auto timed_copies =
                        make_shared<vector<shared_ptr<RandomVariableNode>>>();

                    for (int t = t0; t <= to_time; t++) {
                        string node_name =
                            node_template->get_metadata()->get_timed_name(t);
                        int node_id = this->name_to_id[node_name];
                        const auto& node = this->graph[node_id].node;
                        timed_copies->push_back(node);
                        node->set_timed_copies(timed_copies);
                    }
                }
            }
        }

        void
        DynamicBayesNet::update_cpd_templates_dependencies(int new_time_steps) {
            // The updates are made using the node templates and they will
            // reflect in the timed instances that share the same CPD. This
            // is to avoid calling the update multiple times for the same
            // CPD that has sharable distributions among the timed copies.

            for (const auto& node_template : this->node_templates) {
                const shared_ptr<NodeMetadata> metadata =
                    node_template->get_metadata();
                if (metadata->has_parameter_parents()) {
                    int from_time = max(metadata->get_initial_time_step(),
                                        this->time_steps);
                    int to_time;
                    if (metadata->is_replicable()) {
                        // 1 is the max time step where parameter nodes can show
                        // up.
                        // TODO - change this if parameters can show up in
                        //  nodes can show up in future positions. This is
                        //  just to make the update of the CPD's faster since
                        //  we only need to update one pointer to reflect in
                        //  all of the copies of the node.
                        to_time = min(this->time_steps + new_time_steps - 1, 1);
                    }
                    else {
                        to_time = min(this->time_steps + new_time_steps - 1,
                                      metadata->get_initial_time_step());
                    }

                    for (int t = from_time; t <= to_time; t++) {
                        // We only need to update once (when the node is
                        // created). Timed copies of future expansions of
                        // the DBN will inherit the correct dependencies set
                        // here.
                        string node_name = metadata->get_timed_name(t);
                        int vertex_id = this->name_to_id.at(node_name);
                        auto& vertex_data = this->graph[vertex_id];
                        vertex_data.node->update_cpd_templates_dependencies(
                            this->parameter_nodes_map);
                    }
                }
            }
        }

        void DynamicBayesNet::save_topological_list() {
            this->topological_nodes_per_time =
                vector<RVNodePtrVec>(this->time_steps);

            for (const auto& node : this->get_nodes_topological_order()) {
                const RVNodePtr& rv_node =
                    dynamic_pointer_cast<RandomVariableNode>(node);
                int t = rv_node->get_time_step();
                this->topological_nodes_per_time.at(t).push_back(rv_node);
            }
        }

        void DynamicBayesNet::check() {
            // TODO - Implement the verifications needed to make sure the
            // DBN is
            //  valid and prepared to be unrolled.
            //  Only allow conjugate priors
        }

        vector<shared_ptr<Node>> DynamicBayesNet::get_parameter_nodes() {
            vector<shared_ptr<Node>> parameter_nodes;
            parameter_nodes.reserve(this->parameter_nodes_map.size());

            for (const auto& mapping : this->parameter_nodes_map) {
                parameter_nodes.push_back(mapping.second);
            }

            return parameter_nodes;
        }

        vector<shared_ptr<Node>>
        DynamicBayesNet::get_nodes_by_label(const string& node_label) const {
            vector<shared_ptr<Node>> nodes;
            if (EXISTS(node_label, this->label_to_nodes)) {
                nodes = this->label_to_nodes.at(node_label);
            }

            return nodes;
        }

        vector<shared_ptr<Node>> DynamicBayesNet::get_nodes_topological_order(
            bool from_roots_to_leaves) const {
            vector<int> vertex_ids;
            boost::topological_sort(this->graph, back_inserter(vertex_ids));

            vector<shared_ptr<Node>> nodes(vertex_ids.size());
            int i = 0;
            if (from_roots_to_leaves) {
                // The default behavior of the boost topological sort is to
                // order from the leaves to the roots so if we want
                // otherwise, the elements have to be inserted in reversed
                // order in the final array.
                i = vertex_ids.size() - 1;
            }

            for (int vertex_id : vertex_ids) {
                if (from_roots_to_leaves) {
                    nodes[i--] = this->graph[vertex_id].node;
                }
                else {
                    nodes[i++] = this->graph[vertex_id].node;
                }
            }

            return nodes;
        }

        vector<shared_ptr<Node>>
        DynamicBayesNet::get_parent_nodes_of(const shared_ptr<Node>& node,
                                             bool exclude_parameters) const {

            int vertex_id = this->name_to_id.at(node->get_timed_name());
            vector<shared_ptr<Node>> parent_nodes;

            Graph::in_edge_iterator in_begin, in_end;
            boost::tie(in_begin, in_end) = in_edges(vertex_id, this->graph);
            while (in_begin != in_end) {
                int parent_vertex_id = source(*in_begin, graph);
                if (!this->graph[parent_vertex_id]
                         .node->get_metadata()
                         ->is_parameter() ||
                    !exclude_parameters) {
                    parent_nodes.push_back(this->graph[parent_vertex_id].node);
                }
                in_begin++;
            }

            return parent_nodes;
        }

        vector<shared_ptr<Node>>
        DynamicBayesNet::get_child_nodes_of(const shared_ptr<Node>& node,
                                            bool exclude_timers) const {

            int vertex_id = this->name_to_id.at(node->get_timed_name());
            vector<shared_ptr<Node>> child_nodes;

            Graph::out_edge_iterator out_begin, out_end;
            boost::tie(out_begin, out_end) = out_edges(vertex_id, this->graph);
            while (out_begin != out_end) {
                int child_vertex_id = target(*out_begin, graph);
                if (!this->graph[child_vertex_id]
                         .node->get_metadata()
                         ->is_timer() ||
                    !exclude_timers) {
                    child_nodes.push_back(this->graph[child_vertex_id].node);
                }
                out_begin++;
            }

            return child_nodes;
        }

        void DynamicBayesNet::save_to(const string& output_dir) const {

            boost::filesystem::create_directories(output_dir);

            for (const auto& mapping : this->parameter_nodes_map) {
                string filename = mapping.first;
                string filepath = get_filepath(output_dir, filename);
                save_matrix_to_file(filepath, mapping.second->get_assignment());
            }
        }

        void DynamicBayesNet::load_from(const string& input_dir,
                                        bool freeze_nodes) {
            // Parameters of the model should be in files with the same name
            // as the parameters timed names. (eg. (Theta_S0,0).txt)
            for (const auto& file :
                 boost::filesystem::directory_iterator(input_dir)) {
                // Ignore directories
                if (boost::filesystem::is_regular_file(file)) {
                    string filename = file.path().filename().string();
                    string parameter_timed_name = filename;

                    // Only process the file's content if a parameter with
                    // the same name exists in the model.
                    if (EXISTS(parameter_timed_name,
                               this->parameter_nodes_map)) {

                        RandomVariableNode* parameter_node =
                            dynamic_cast<RandomVariableNode*>(
                                this->parameter_nodes_map
                                    .at(parameter_timed_name)
                                    .get());

                        // We set the loaded matrix as the assignment of the
                        // corresponding parameter node and freeze it so it
                        // cannot be modified by sampling.
                        Eigen::MatrixXd assignment =
                            read_matrix_from_file(file.path().string());
                        parameter_node->set_assignment(assignment);

                        if (freeze_nodes) {
                            parameter_node->freeze();
                        }
                    }
                }
            }
        }

        vector<DynamicBayesNet::Edge> DynamicBayesNet::get_edges() const {
            Graph::edge_iterator begin, end;
            boost::tie(begin, end) = boost::edges(this->graph);
            vector<Edge> edges;
            while (begin != end) {
                int source_vertex_id = boost::source(*begin, graph);
                int target_vertex_id = boost::target(*begin, graph);

                Edge edge = make_pair(this->graph[source_vertex_id].node,
                                      this->graph[target_vertex_id].node);
                edges.push_back(move(edge));
                begin++;
            }

            return edges;
        }

        void
        DynamicBayesNet::write_graphviz(std::ostream& output_stream) const {
            boost::write_graphviz(output_stream,
                                  this->graph,
                                  boost::make_label_writer(boost::get(
                                      &VertexData::label, this->graph)));
        }

        int DynamicBayesNet::get_cardinality_of(
            const std::string& node_label) const {
            return this->get_metadata_of(node_label)->get_cardinality();
        }

        bool DynamicBayesNet::has_node_with_label(
            const std::string& node_label) const {
            return EXISTS(node_label, this->label_to_nodes);
        }

        bool DynamicBayesNet::has_parameter_node_with_label(
            const std::string& node_label) const {
            if (EXISTS(node_label, this->label_to_nodes)) {
                return this->label_to_nodes.at(node_label)[0]
                    ->get_metadata()
                    ->is_parameter();
            }

            return false;
        }

        vector<string> DynamicBayesNet::get_parameter_node_labels() const {
            vector<string> labels;

            for (const auto& node_template : this->node_templates) {
                if (node_template->get_metadata()->is_parameter()) {
                    labels.push_back(
                        node_template->get_metadata()->get_label());
                }
            }

            return labels;
        }

        RVNodePtr DynamicBayesNet::get_node(const std::string& label,
                                            int time_step) {
            RVNodePtr node;
            string name = NodeMetadata::get_timed_name(label, time_step);
            if (EXISTS(name, this->name_to_id)) {
                int id = this->name_to_id.at(name);
                node = this->graph[id].node;
            }

            return node;
        }

        shared_ptr<NodeMetadata>
        DynamicBayesNet::get_metadata_of(const std::string& node_label) const {
            return this->label_to_nodes.at(node_label)[0]->get_metadata();
        }

        DynamicBayesNet DynamicBayesNet::clone(bool unroll) {
            DynamicBayesNet new_dbn(this->node_templates.size());
            for (const auto& node_template : this->node_templates) {
                new_dbn.add_node_template(node_template);
            }

            if (unroll) {
                new_dbn.unroll(this->time_steps, true);
            }

            return new_dbn;
        }

        void DynamicBayesNet::mirror_parameter_nodes_from(
            const DynamicBayesNet& dbn) {

            for (const auto& [parameter_name, parameter_node] :
                 dbn.parameter_nodes_map) {
                if (EXISTS(parameter_name, this->parameter_nodes_map)) {
                    RVNodePtr rv_parameter_node =
                        dynamic_pointer_cast<RandomVariableNode>(
                            parameter_node);
                    RVNodePtr original_node =
                        dynamic_pointer_cast<RandomVariableNode>(
                            this->parameter_nodes_map.at(parameter_name));

                    original_node->set_assignment(
                        parameter_node->get_assignment());
                    if (rv_parameter_node->is_frozen()) {
                        original_node->freeze();
                    }
                }
            }
        }

        RVNodePtrVec DynamicBayesNet::get_nodes_in_topological_order() {
            RVNodePtrVec nodes;

            for (int t = 0; t < this->time_steps; t++) {
                const auto& nodes_at_time_step =
                    this->get_nodes_in_topological_order_at(t);
                nodes.insert(nodes.end(),
                             nodes_at_time_step.begin(),
                             nodes_at_time_step.end());
            }

            return nodes;
        }

        RVNodePtrVec
        DynamicBayesNet::get_nodes_in_topological_order_at(int time_step) {
            return this->topological_nodes_per_time.at(time_step);
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        int DynamicBayesNet::get_time_steps() const { return time_steps; }

        bool DynamicBayesNet::is_exact_inference_allowed() const {
            return exact_inference_allowed;
        }

    } // namespace model
} // namespace tomcat
