#include "pgm/JSONModel.h"

#include <fstream>

#include <nlohmann/json.hpp>

#include "distribution/Categorical.h"
#include "distribution/Dirichlet.h"
#include "distribution/Distribution.h"
#include "distribution/Gamma.h"
#include "distribution/Gaussian.h"
#include "distribution/Geometric.h"
#include "distribution/Poisson.h"
#include "pgm/ConstantNode.h"
#include "pgm/DynamicBayesNet.h"
#include "pgm/NodeMetadata.h"
#include "pgm/RandomVariableNode.h"
#include "pgm/TimerNode.h"
#include "pgm/cpd/CPD.h"
#include "pgm/cpd/CategoricalCPD.h"
#include "pgm/cpd/DirichletCPD.h"
#include "pgm/cpd/GammaCPD.h"
#include "pgm/cpd/GaussianCPD.h"
#include "pgm/cpd/GeometricCPD.h"
#include "pgm/cpd/PoissonCPD.h"
#include "utils/Definitions.h"

/**
 * This file creates a DBN from a the model's specifications in a JSON file.
 */

using namespace std;

namespace tomcat {
    namespace model {

        typedef unordered_map<string, vector<MetadataPtr>> MetadataMap;
        typedef unordered_map<string, RVNodePtrVec> RVMap;
        typedef unordered_set<string> NodeSet;
        typedef unordered_map<string, pair<int, int>> ParamMapConfig;

        /**
         * Creates metadatas for timer nodes and adds to the list of metadatas.
         *
         * @param json_model: json object containing the model's specification
         * @param metadatas: metadatas created so far
         */
        static void create_timer_metadatas(const nlohmann::json& json_model,
                                           MetadataMap& metadatas);

        /**
         * Creates metadatas for data nodes and adds to the list of metadatas.
         *
         * @param json_model: json object containing the model's specification
         * @param metadatas: metadatas created so far
         */
        static void create_data_metadatas(const nlohmann::json& json_model,
                                          MetadataMap& metadatas);

        /**
         * Creates metadatas for parameter nodes and adds to the list of
         * metadatas.
         *
         * @param json_model: json object containing the model's specification
         * @param metadatas: metadatas created so far
         */
        static void create_parameter_metadatas(const nlohmann::json& json_model,
                                               MetadataMap& metadatas);

        /**
         * Gets the set of nodes that have at least one child that is
         * replicable over time.
         *
         * @param json_model: json object containing the model's specification
         *
         * @return Set of nodes
         */
        static NodeSet
        get_nodes_with_replicable_child(const nlohmann::json& json_model);

        /**
         * Gets the number of copies we need to create of parameter nodes
         * based on the number of rows in the CPDs they show up, and their time
         * step based on the minimum initial time step of the nodes that own
         * these CPDs.
         *
         * @param json_model: json object containing the model's specification
         * @param metadatas: metadatas created so far
         *
         * @return Map between parameter node labels and their number of copies
         * and time steps.
         */
        static ParamMapConfig
        get_num_param_copies_and_time_steps(const nlohmann::json& json_model,
                                            const MetadataMap& metadatas);

        /**
         * Updates the number of copies we need to create of parameter nodes
         * based on the number of rows in the CPDs they show up, and their time
         * step based on the minimum initial time step of the nodes that own
         * these CPDs.
         *
         * @param json_cpds: json object containing a list of CPDs
         * @param cpd_owner_label: label of the node that owns the CPD
         * @param metadatas: metadatas created so far
         * @param param_config: map between parameter node labels and their
         * number of copies and time steps.
         *
         * @return number of copies and time steps
         */
        static void
        update_num_param_copies_and_time_steps(const nlohmann::json& json_cpds,
                                               const string& cpd_owner_label,
                                               const MetadataMap& metadatas,
                                               ParamMapConfig& param_config);

        /**
         * Creates a series of CPD templates and associates them to the
         * corresponding node.
         *
         * @param cpd: json with the CPD specifications
         * @param cpd_owner_label: label of the node that owns the CPD
         * @param rv_nodes: list of all the nodes in the model
         */
        static void create_cpds(const nlohmann::json& json_cpds,
                                const string& cpd_owner_label,
                                const RVMap& rv_nodes);

        /**
         * Creates a constant CPD and associates it to the corresponding node.
         *
         * @tparam C: Class of the CPD according to the CPD distribution
         * @param cpd: json with the CPD specifications
         * @param cpd_owner_label: label of the node that owns the CPD
         * @param rv_nodes: list of all the nodes in the model
         * @param cols: number of columns in the CPD. It can either be the
         * cardinality of the node or the number of parameters of a specific
         * distribution
         */
        template <class C>
        static void create_constant_cpd(const nlohmann::json& json_cpd,
                                        const string& cpd_owner_label,
                                        const RVMap& rv_nodes,
                                        int cols);

        /**
         * Splits a string into a list of doubles given a delimiter.
         *
         * @param str: string to be split
         * @param delimiter: delimiter
         *
         * @return list of tokens
         */
        static vector<double> split_string(string str, string delimiter = ",");

        /**
         * Creates a CPD that depend on other nodes (parameter nodes) and
         * associates it to the corresponding node.
         *
         * @param C: Class of the CPD according to the CPD distribution
         * @param D: Class of the distribution that comprises the CPD
         * @param cpd: json with the CPD specifications
         * @param cpd_owner_label: label of the node that owns the CPD
         * @param rv_nodes: list of all the nodes in the model
         */
        template <class C, class D>
        static void create_node_dependent_cpd(const nlohmann::json& json_cpd,
                                              const string& cpd_owner_label,
                                              const RVMap& rv_nodes);

        DynamicBayesNet create_model_from_json(const string& filepath) {
            fstream file;
            file.open(filepath);
            if (file.is_open()) {
                nlohmann::json json_model = nlohmann::json::parse(file);

                MetadataMap metadatas;
                RVMap rv_nodes;

                // Create metadatas
                create_timer_metadatas(json_model, metadatas);
                create_data_metadatas(json_model, metadatas);
                create_parameter_metadatas(json_model, metadatas);

                // Create connections between data nodes
                for (const auto& connections : json_model["connections"]) {
                    const auto& parent_metadata =
                        metadatas[connections["parent"]].at(0);
                    const auto& child_metadata =
                        metadatas[connections["child"]].at(0);

                    child_metadata->add_parent_link(
                        parent_metadata, connections["time_crossing"]);
                }

                // Create rv nodes from metadatas
                for (const auto& [label, metadata_copies] : metadatas) {
                    for (const auto& metadata : metadata_copies) {
                        if (metadata->is_timer()) {
                            rv_nodes[label].push_back(
                                make_shared<TimerNode>(metadata));
                        }
                        else {
                            rv_nodes[label].push_back(
                                make_shared<RandomVariableNode>(metadata));
                        }
                    }
                }

                // Create CPDs and connections between parameter and data nodes
                for (const auto& node : json_model["nodes"]["timers"]) {
                    create_cpds(node["cpds"], node["label"], rv_nodes);
                }
                for (const auto& node : json_model["nodes"]["parameters"]) {
                    create_cpds(node["cpds"], node["label"], rv_nodes);
                }
                for (const auto& node : json_model["nodes"]["data"]) {
                    create_cpds(node["cpds"], node["label"], rv_nodes);
                }

                // Create model and add nodes to it
                DynamicBayesNet dbn;
                for (const auto& [label, node_copies] : rv_nodes) {
                    for (const auto& node : node_copies) {
                        dbn.add_node_template(node);
                    }
                }

                return dbn;
            }
            else {
                stringstream ss;
                ss << "The file " << filepath << " does not exist.";
                throw TomcatModelException(ss.str());
            }
        }

        static void create_timer_metadatas(const nlohmann::json& json_model,
                                           MetadataMap& metadatas) {
            for (const auto& node : json_model["nodes"]["timers"]) {
                const string& label = node["label"];
                metadatas[label].push_back(make_shared<NodeMetadata>(
                    NodeMetadata::create_timer_metadata(label, 0)));
            }
        }

        static void create_data_metadatas(const nlohmann::json& json_model,
                                          MetadataMap& metadatas) {

            NodeSet nodes_with_replicable_child =
                get_nodes_with_replicable_child(json_model);

            for (const auto& node : json_model["nodes"]["data"]) {
                const string& label = node["label"];
                bool prior = false;
                if (EXISTS("prior", node)) {
                    prior = node["prior"];
                }
                if (!prior && (EXISTS(label, nodes_with_replicable_child) ||
                               node["replicable"])) {
                    metadatas[label].push_back(make_shared<NodeMetadata>(
                        NodeMetadata::create_multiple_time_link_metadata(
                            label,
                            node["replicable"],
                            false,
                            node["in_plate"],
                            node["first_time_step"],
                            node["sample_size"],
                            node["cardinality"])));
                }
                else {
                    metadatas[label].push_back(make_shared<NodeMetadata>(
                        NodeMetadata::create_single_time_link_metadata(
                            label,
                            false,
                            node["in_plate"],
                            node["first_time_step"],
                            node["sample_size"],
                            node["cardinality"])));
                }

                if (node["timer"] != "") {
                    const string& timer_label = node["timer"];
                    auto& timer_metadata = metadatas[timer_label].at(0);
                    metadatas[label].at(0)->set_timer_metadata(timer_metadata);
                }
            }
        }

        static void create_parameter_metadatas(const nlohmann::json& json_model,
                                               MetadataMap& metadatas) {

            ParamMapConfig param_config =
                get_num_param_copies_and_time_steps(json_model, metadatas);

            for (const auto& node : json_model["nodes"]["parameters"]) {
                const string& label = node["label"];
                auto [num_copies, time_step] = param_config.at(label);

                if (node["prior"]) {
                    for (int i = 0; i < num_copies; i++) {
                        stringstream derived_label;
                        derived_label << label << "_" << i;
                        metadatas[label].push_back(make_shared<NodeMetadata>(
                            NodeMetadata::create_single_time_link_metadata(
                                derived_label.str(),
                                true,
                                false,
                                time_step,
                                node["sample_size"])));
                    }
                }
                else {
                    for (int i = 0; i < num_copies; i++) {
                        stringstream derived_label;
                        derived_label << label << "_" << i;

                        metadatas[label].push_back(make_shared<NodeMetadata>(
                            NodeMetadata::create_multiple_time_link_metadata(
                                derived_label.str(),
                                false,
                                true,
                                false,
                                time_step,
                                node["sample_size"])));
                    }
                }
            }
        }

        static NodeSet
        get_nodes_with_replicable_child(const nlohmann::json& json_model) {
            NodeSet replicable_nodes;
            NodeSet nodes_with_replicable_child;

            for (const auto& node : json_model["nodes"]["timers"]) {
                // Times are always replicable
                const string& label = node["label"];
                replicable_nodes.insert(label);
            }
            for (const auto& node : json_model["nodes"]["data"]) {
                if (node["replicable"]) {
                    const string& label = node["label"];
                    replicable_nodes.insert(label);
                }
            }

            // Store the nodes that have at least one child that
            // replicates over time so we can create the right metadata
            // for the node later.
            for (const auto& connections : json_model["connections"]) {
                const string& child_label = connections["child"];
                if (EXISTS(child_label, replicable_nodes)) {
                    const string& parent_label = connections["parent"];
                    nodes_with_replicable_child.insert(parent_label);
                }
            }

            return nodes_with_replicable_child;
        }

        static ParamMapConfig
        get_num_param_copies_and_time_steps(const nlohmann::json& json_model,
                                            const MetadataMap& metadatas) {
            ParamMapConfig param_config;

            for (const auto& node : json_model["nodes"]["timers"]) {
                update_num_param_copies_and_time_steps(
                    node["cpds"], node["label"], metadatas, param_config);
            }
            for (const auto& node : json_model["nodes"]["parameters"]) {
                update_num_param_copies_and_time_steps(
                    node["cpds"], node["label"], metadatas, param_config);
            }
            for (const auto& node : json_model["nodes"]["data"]) {
                update_num_param_copies_and_time_steps(
                    node["cpds"], node["label"], metadatas, param_config);
            }

            return param_config;
        }

        static void
        update_num_param_copies_and_time_steps(const nlohmann::json& json_cpds,
                                               const string& cpd_owner_label,
                                               const MetadataMap& metadatas,
                                               ParamMapConfig& param_config) {

            for (const auto& cpd : json_cpds) {
                if (!cpd["constant"]) {
                    bool transition_cpd = false;
                    int num_copies = 1;
                    for (const string& index_node : cpd["index_nodes"]) {
                        transition_cpd =
                            transition_cpd || cpd_owner_label == index_node;
                        int cardinality =
                            metadatas.at(index_node).at(0)->get_cardinality();
                        num_copies *= cardinality;
                    }
                    const string& parameter_label = cpd["parameters"];

                    int time_step = metadatas.at(cpd_owner_label)
                                        .at(0)
                                        ->get_initial_time_step();
                    if (transition_cpd) {
                        // It depends on the previous instance of the same node.
                        time_step += 1;
                    }

                    param_config[parameter_label] =
                        make_pair(num_copies, time_step);
                }
            }
        }

        static void create_cpds(const nlohmann::json& json_cpds,
                                const string& cpd_owner_label,
                                const RVMap& rv_nodes) {
            for (const auto& json_cpd : json_cpds) {
                const string& distribution = json_cpd["distribution"];

                if (json_cpd["constant"]) {
                    if (distribution == "categorical") {
                        int cardinality = rv_nodes.at(cpd_owner_label)
                                              .at(0)
                                              ->get_metadata()
                                              ->get_cardinality();
                        create_constant_cpd<CategoricalCPD>(
                            json_cpd, cpd_owner_label, rv_nodes, cardinality);
                    }
                    else if (distribution == "poisson") {
                        create_constant_cpd<PoissonCPD>(
                            json_cpd, cpd_owner_label, rv_nodes, 1);
                    }
                    else if (distribution == "geometric") {
                        create_constant_cpd<GeometricCPD>(
                            json_cpd, cpd_owner_label, rv_nodes, 1);
                    }
                    else if (distribution == "dirichlet") {
                        int sample_size = rv_nodes.at(cpd_owner_label)
                                              .at(0)
                                              ->get_metadata()
                                              ->get_sample_size();
                        create_constant_cpd<DirichletCPD>(
                            json_cpd, cpd_owner_label, rv_nodes, sample_size);
                    }
                    else if (distribution == "gamma") {
                        create_constant_cpd<GammaCPD>(
                            json_cpd, cpd_owner_label, rv_nodes, 2);
                    }
                    else if (distribution == "gaussian") {
                        create_constant_cpd<GaussianCPD>(
                            json_cpd, cpd_owner_label, rv_nodes, 2);
                    }
                    else {
                        stringstream ss;
                        ss << "Distribution " << distribution
                           << " not supported.";
                        throw TomcatModelException(ss.str());
                    }
                }
                else {
                    if (distribution == "categorical") {
                        create_node_dependent_cpd<CategoricalCPD, Categorical>(
                            json_cpd, cpd_owner_label, rv_nodes);
                    }
                    else if (distribution == "poisson") {
                        create_node_dependent_cpd<PoissonCPD, Poisson>(
                            json_cpd, cpd_owner_label, rv_nodes);
                    }
                    else if (distribution == "geometric") {
                        create_node_dependent_cpd<GeometricCPD, Geometric>(
                            json_cpd, cpd_owner_label, rv_nodes);
                    }
                    // TODO -  The constructor of these distributions receive
                    //  a vector of nodes. In the future, adjust the
                    //  interface to allow parameter nodes to have
                    //  dependencies if necessary
                    //                    else if (distribution == "dirichlet")
                    //                    {
                    //                        create_node_dependent_cpd<DirichletCPD,
                    //                        Dirichlet>(
                    //                            json_cpd, cpd_owner_label,
                    //                            rv_nodes);
                    //                    }
                    //                    else if (distribution == "gamma") {
                    //                        create_node_dependent_cpd<GammaCPD,
                    //                        Gamma>(
                    //                            json_cpd, cpd_owner_label,
                    //                            rv_nodes);
                    //                    }
                    //                    else if (distribution == "gaussian") {
                    //                        create_node_dependent_cpd<GaussianCPD,
                    //                        Gaussian>(
                    //                            json_cpd, cpd_owner_label,
                    //                            rv_nodes);
                    //                    }
                    else {
                        stringstream ss;
                        ss << "Distribution " << distribution
                           << " not supported.";
                        throw TomcatModelException(ss.str());
                    }
                }
            }
        }

        template <class C>
        static void create_constant_cpd(const nlohmann::json& json_cpd,
                                        const string& cpd_owner_label,
                                        const RVMap& rv_nodes,
                                        int cols) {

            vector<MetadataPtr> index_metadatas;
            for (const string& index_node_label : json_cpd["index_nodes"]) {
                // Index nodes can only be data nodes,
                // and therefore, only has one copy in
                // the metadata list.
                index_metadatas.push_back(
                    rv_nodes.at(index_node_label).at(0)->get_metadata());
            }

            vector<double> parameter_values =
                split_string(json_cpd["parameters"], ",");

            int rows = 1;
            if (rv_nodes.at(cpd_owner_label)
                    .at(0)
                    ->get_metadata()
                    ->is_parameter()) {
                rows = rv_nodes.at(cpd_owner_label).size();
            }
            else {
                for (const string& index_node : json_cpd["index_nodes"]) {
                    int cardinality = rv_nodes.at(index_node)
                                          .at(0)
                                          ->get_metadata()
                                          ->get_cardinality();
                    rows *= cardinality;
                }
            }
            Eigen::MatrixXd cpd_table =
                Eigen::Map<Eigen::Matrix<double,
                                         Eigen::Dynamic,
                                         Eigen::Dynamic,
                                         Eigen::RowMajor>>(
                    parameter_values.data(), rows, cols);

            if (rv_nodes.at(cpd_owner_label).size() == 1) {
                // Full matrix as CPD
                CPDPtr cpd = make_shared<C>(index_metadatas, cpd_table);
                const auto& cpd_owner = rv_nodes.at(cpd_owner_label).at(0);
                cpd_owner->add_cpd_template(move(cpd));
            }
            else {
                // Each row of the matrix comprises the CPD of one of the
                // node copies
                for (int i = 0; i < rv_nodes.at(cpd_owner_label).size(); i++) {
                    CPDPtr cpd =
                        make_shared<C>(index_metadatas, cpd_table.row(i));
                    const auto& cpd_owner = rv_nodes.at(cpd_owner_label).at(i);
                    cpd_owner->add_cpd_template(move(cpd));
                }
            }
        }

        /**
         * Splits a string into a list of doubles given a delimiter.
         *
         * @param str: string to be split
         * @param delimiter: delimiter
         *
         * @return list of tokens
         */
        static vector<double> split_string(string str, string delimiter) {
            vector<double> split_list;
            size_t pos;
            while ((pos = str.find(delimiter)) != string::npos) {
                double token = stod(str.substr(0, pos));
                if (token == 0) {
                    token = EPSILON; // For numerical stability.
                }
                split_list.push_back(token);
                str.erase(0, pos + delimiter.length());
            }

            double token = stod(str.substr(0, pos));
            split_list.push_back(token);

            return split_list;
        }

        template <class C, class D>
        static void create_node_dependent_cpd(const nlohmann::json& json_cpd,
                                              const string& cpd_owner_label,
                                              const RVMap& rv_nodes) {

            vector<MetadataPtr> index_metadatas;
            for (const string& index_node_label : json_cpd["index_nodes"]) {
                // Index nodes can only be data nodes,
                // and therefore, only has one copy in
                // the metadata list.
                index_metadatas.push_back(
                    rv_nodes.at(index_node_label).at(0)->get_metadata());
            }

            const string& parameter_label = json_cpd["parameters"];
            vector<shared_ptr<D>> distributions;
            for (const auto& parameter_node : rv_nodes.at(parameter_label)) {
                shared_ptr<Node> node = parameter_node;
                shared_ptr<D> distribution = make_shared<D>(parameter_node);
                distributions.push_back(distribution);

                // Add parameter as a parent of the cpd owner

                const auto& cpd_owner_metadata =
                    rv_nodes.at(cpd_owner_label).at(0)->get_metadata();
                const auto& parameter_metadata = parameter_node->get_metadata();
                bool time_crossing = !parameter_metadata->is_single_time_link();
                cpd_owner_metadata->add_parent_link(parameter_metadata,
                                                    time_crossing);
            }

            CPDPtr cpd = make_shared<C>(index_metadatas, distributions);
            const auto& cpd_owner = rv_nodes.at(cpd_owner_label).at(0);
            cpd_owner->add_cpd_template(move(cpd));
        }

    } // namespace model
} // namespace tomcat