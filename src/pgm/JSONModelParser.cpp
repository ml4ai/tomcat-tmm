#include "pgm/JSONModelParser.h"

#include <fstream>

#include <boost/algorithm/string.hpp>
#include <nlohmann/json.hpp>

#include "distribution/Categorical.h"
#include "distribution/Dirichlet.h"
#include "distribution/Distribution.h"
#include "distribution/Gamma.h"
#include "distribution/Gaussian.h"
#include "distribution/Geometric.h"
#include "distribution/InverseGamma.h"
#include "distribution/Poisson.h"
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
#include "pgm/cpd/InverseGammaCPD.cpp"
#include "pgm/cpd/PoissonCPD.h"
#include "utils/Definitions.h"

/**
 * This file creates a DBN from a the model's specifications in a JSON file.
 */

using namespace std;

namespace tomcat {
    namespace model {

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        JSONModelParser::JSONModelParser(const string& model_filepath) {
            fstream file;
            file.open(model_filepath);
            if (file.is_open()) {
                this->json_model = nlohmann::json::parse(file);
            }
            else {
                stringstream ss;
                ss << "The file " << model_filepath << " does not exist.";
                throw TomcatModelException(ss.str());
            }
        }

        JSONModelParser::~JSONModelParser() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void JSONModelParser::check_model_definition() const {
            // TODO - Implement checks
        }

        DynamicBayesNet JSONModelParser::create_model() {
            // Create variables if provided
            this->create_variables();

            // Create this->metadatas
            this->create_timer_metadatas();
            this->create_data_metadatas();
            this->create_parameter_metadatas();

            // Create connections between data nodes
            for (const auto& connections : this->json_model["connections"]) {
                const auto& parent_metadata =
                    this->metadatas[connections["parent"]].at(0);
                const auto& child_metadata =
                    this->metadatas[connections["child"]].at(0);

                child_metadata->add_parent_link(parent_metadata,
                                                connections["time_crossing"]);
            }

            // Create rv nodes from this->metadatas
            for (const auto& [label, metadata_copies] : this->metadatas) {
                for (const auto& metadata : metadata_copies) {
                    if (metadata->is_timer()) {
                        this->rv_nodes[label].push_back(
                            make_shared<TimerNode>(metadata));
                    }
                    else {
                        this->rv_nodes[label].push_back(
                            make_shared<RandomVariableNode>(metadata));
                    }
                }
            }

            // Create CPDs and connections between parameter and data nodes
            for (const auto& node : this->json_model["nodes"]["timers"]) {
                create_cpds(node["cpds"], node["label"]);
            }
            for (const auto& node : this->json_model["nodes"]["parameters"]) {
                create_cpds(node["cpds"], node["label"]);
            }
            for (const auto& node : this->json_model["nodes"]["data"]) {
                create_cpds(node["cpds"], node["label"]);
            }

            // Create model and add nodes to it
            DynamicBayesNet dbn;
            for (const auto& [label, node_copies] : this->rv_nodes) {
                for (const auto& node : node_copies) {
                    dbn.add_node_template(node);
                }
            }

            return dbn;
        }

        void JSONModelParser::create_variables() {
            if (EXISTS("variables", this->json_model)) {
                for (const auto& json_variable :
                     this->json_model["variables"]) {
                    const string& var_name = json_variable["name"];
                    vector<double> values;
                    if (json_variable["default_value"].is_array()) {
                        for (double value : json_variable["default_value"]) {
                            values.push_back(value);
                        }
                    }
                    else {
                        values.push_back(json_variable["default_value"]);
                    }

                    for (double value : values) {
                        if (value == 0) {
                            value = EPSILON; // For numerical stability when
                                             // using logs
                        }
                        this->variables[var_name].push_back(
                            make_shared<NumericNode>(value));
                    }
                }
            }
        }

        void JSONModelParser::create_timer_metadatas() {
            for (const auto& node : this->json_model["nodes"]["timers"]) {
                const string& label = node["label"];
                this->metadatas[label].push_back(make_shared<NodeMetadata>(
                    NodeMetadata::create_timer_metadata(label, 0)));
            }
        }

        void JSONModelParser::create_data_metadatas() {

            NodeSet nodes_with_replicable_child =
                get_nodes_with_replicable_child();

            for (const auto& node : this->json_model["nodes"]["data"]) {
                const string& label = node["label"];
                bool prior = false;
                if (EXISTS("prior", node)) {
                    prior = node["prior"];
                }
                if (!prior && (EXISTS(label, nodes_with_replicable_child) ||
                               node["replicable"])) {
                    this->metadatas[label].push_back(make_shared<NodeMetadata>(
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
                    this->metadatas[label].push_back(make_shared<NodeMetadata>(
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
                    auto& timer_metadata = this->metadatas[timer_label].at(0);
                    this->metadatas[label].at(0)->set_timer_metadata(
                        timer_metadata);
                }
            }
        }

        void JSONModelParser::create_parameter_metadatas() {

            ParamMapConfig param_config = get_num_param_copies_and_time_steps();

            for (const auto& node : this->json_model["nodes"]["parameters"]) {
                const string& label = node["label"];
                auto [num_copies, time_step] = param_config.at(label);

                if (node["prior"]) {
                    for (int i = 0; i < num_copies; i++) {
                        stringstream derived_label;
                        derived_label << label << "_" << i;
                        this->metadatas[label].push_back(
                            make_shared<NodeMetadata>(
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

                        this->metadatas[label].push_back(make_shared<
                                                         NodeMetadata>(
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

        JSONModelParser::NodeSet
        JSONModelParser::get_nodes_with_replicable_child() {
            NodeSet replicable_nodes;
            NodeSet nodes_with_replicable_child;

            for (const auto& node : this->json_model["nodes"]["timers"]) {
                // Times are always replicable
                const string& label = node["label"];
                replicable_nodes.insert(label);
            }
            for (const auto& node : this->json_model["nodes"]["data"]) {
                if (node["replicable"]) {
                    const string& label = node["label"];
                    replicable_nodes.insert(label);
                }
            }

            // Store the nodes that have at least one child that
            // replicates over time so we can create the right metadata
            // for the node later.
            for (const auto& connections : this->json_model["connections"]) {
                const string& child_label = connections["child"];
                if (EXISTS(child_label, replicable_nodes)) {
                    const string& parent_label = connections["parent"];
                    nodes_with_replicable_child.insert(parent_label);
                }
            }

            return nodes_with_replicable_child;
        }

        JSONModelParser::ParamMapConfig
        JSONModelParser::get_num_param_copies_and_time_steps() {
            ParamMapConfig param_config;
            unordered_set<string> parameter_labels;

            for (const auto& json_parameter :
                 this->json_model["nodes"]["parameters"]) {
                parameter_labels.insert((string)json_parameter["label"]);
            }

            for (const auto& node : this->json_model["nodes"]["timers"]) {
                update_num_param_copies_and_time_steps(parameter_labels,
                                                       node["cpds"],
                                                       node["label"],
                                                       param_config);
            }
            for (const auto& node : this->json_model["nodes"]["parameters"]) {
                update_num_param_copies_and_time_steps(parameter_labels,
                                                       node["cpds"],
                                                       node["label"],
                                                       param_config);
            }
            for (const auto& node : this->json_model["nodes"]["data"]) {
                update_num_param_copies_and_time_steps(parameter_labels,
                                                       node["cpds"],
                                                       node["label"],
                                                       param_config);
            }

            return param_config;
        }

        void JSONModelParser::update_num_param_copies_and_time_steps(
            const unordered_set<string>& parameter_labels,
            const nlohmann::json& json_cpds,
            const string& cpd_owner_label,
            ParamMapConfig& param_config) {

            for (const auto& cpd : json_cpds) {
                if (!cpd["constant"]) {
                    bool transition_cpd = false;
                    int num_copies = 1;
                    for (const string& index_node : cpd["index_nodes"]) {
                        transition_cpd =
                            transition_cpd || cpd_owner_label == index_node;
                        int cardinality = this->metadatas.at(index_node)
                                              .at(0)
                                              ->get_cardinality();
                        num_copies *= cardinality;
                    }

                    vector<string> tokens;
                    boost::split(tokens,
                                 (string)cpd["parameters"],
                                 boost::is_any_of(","));

                    for (string token : tokens) {
                        boost::trim(token);
                        if (EXISTS(token, parameter_labels)) {
                            // The parameter is a random node
                            int time_step = this->metadatas.at(cpd_owner_label)
                                                .at(0)
                                                ->get_initial_time_step();
                            if (transition_cpd) {
                                // It depends on the previous instance of the
                                // same node.
                                time_step += 1;
                            }

                            param_config[token] =
                                make_pair(num_copies, time_step);
                        }
                    }
                }
            }
        }

        void JSONModelParser::create_cpds(const nlohmann::json& json_cpds,
                                          const string& cpd_owner_label) {
            for (const auto& json_cpd : json_cpds) {
                const string& distribution = json_cpd["distribution"];

                if (json_cpd["constant"]) {
                    if (distribution == "categorical") {
                        int cardinality = this->rv_nodes.at(cpd_owner_label)
                                              .at(0)
                                              ->get_metadata()
                                              ->get_cardinality();
                        create_constant_cpd<CategoricalCPD, Categorical>(
                            json_cpd, cpd_owner_label, cardinality);
                    }
                    else if (distribution == "poisson") {
                        create_constant_cpd<PoissonCPD, Poisson>(
                            json_cpd, cpd_owner_label, 1);
                    }
                    else if (distribution == "geometric") {
                        create_constant_cpd<GeometricCPD, Geometric>(
                            json_cpd, cpd_owner_label, 1);
                    }
                    else if (distribution == "dirichlet") {
                        int sample_size = this->rv_nodes.at(cpd_owner_label)
                                              .at(0)
                                              ->get_metadata()
                                              ->get_sample_size();
                        create_constant_cpd<DirichletCPD, Dirichlet>(
                            json_cpd, cpd_owner_label, sample_size);
                    }
                    else if (distribution == "gamma") {
                        create_constant_cpd<GammaCPD, Gamma>(
                            json_cpd, cpd_owner_label, 2);
                    }
                    else if (distribution == "gaussian") {
                        create_constant_cpd<GaussianCPD, Gaussian>(
                            json_cpd, cpd_owner_label, 2);
                    }
                    else if (distribution == "invgamma") {
                        create_constant_cpd<InverseGammaCPD, InverseGamma>(
                            json_cpd, cpd_owner_label, 2);
                    }
                    else {
                        stringstream ss;
                        ss << "Distribution " << distribution
                           << " not supported.";
                        throw TomcatModelException(ss.str());
                    }
                }
                else {
                    // Here are the only distributions that can have parameters
                    // that depend on another distribution (conjugate prior)
                    if (distribution == "categorical") {
                        create_node_dependent_cpd<CategoricalCPD, Categorical>(
                            json_cpd, cpd_owner_label);
                    }
                    else if (distribution == "poisson") {
                        create_node_dependent_cpd<PoissonCPD, Poisson>(
                            json_cpd, cpd_owner_label);
                    }
                    else if (distribution == "geometric") {
                        create_node_dependent_cpd<GeometricCPD, Geometric>(
                            json_cpd, cpd_owner_label);
                    }
                    else if (distribution == "gaussian") {
                        create_node_dependent_cpd<GaussianCPD, Gaussian>(
                            json_cpd, cpd_owner_label);
                    }
                    else {
                        stringstream ss;
                        ss << "Distribution " << distribution
                           << " not supported.";
                        throw TomcatModelException(ss.str());
                    }
                }
            }
        }

        template <class C, class D>
        void
        JSONModelParser::create_constant_cpd(const nlohmann::json& json_cpd,
                                             const string& cpd_owner_label,
                                             int cols) {

            vector<MetadataPtr> index_metadatas;
            for (const string& index_node_label : json_cpd["index_nodes"]) {
                // Index nodes can only be data nodes,
                // and therefore, only has one copy in
                // the metadata list.
                index_metadatas.push_back(
                    this->rv_nodes.at(index_node_label).at(0)->get_metadata());
            }

            NumNodePtrVec cpd_parameters =
                split_constant_parameters(json_cpd["parameters"]);

            int rows = 1;
            if (this->rv_nodes.at(cpd_owner_label)
                    .at(0)
                    ->get_metadata()
                    ->is_parameter()) {
                rows = this->rv_nodes.at(cpd_owner_label).size();
            }
            else {
                for (const string& index_node : json_cpd["index_nodes"]) {
                    int cardinality = this->rv_nodes.at(index_node)
                                          .at(0)
                                          ->get_metadata()
                                          ->get_cardinality();
                    rows *= cardinality;
                }
            }

            // Each row of the matrix comprises the CPD of one of the
            // node copies
            if (this->rv_nodes.at(cpd_owner_label).size() == 1) {
                // A single cpd comprised of several distributions
                vector<shared_ptr<D>> distributions;
                int i = 0;
                for (int row = 0; row < rows; row++) {
                    NodePtrVec distribution_parameters;
                    for (int col = 0; col < cols; col++) {
                        distribution_parameters.push_back(
                            cpd_parameters.at(i++));
                    }

                    shared_ptr<D> distribution =
                        make_shared<D>(distribution_parameters);
                    distributions.push_back(move(distribution));
                }

                CPDPtr cpd = make_shared<C>(index_metadatas, distributions);
                const auto& cpd_owner =
                    this->rv_nodes.at(cpd_owner_label).at(0);
                cpd_owner->add_cpd_template(move(cpd));
            }
            else {
                // One CPD per row (per parameter)
                int i = 0;
                for (int row = 0; row < rows; row++) {
                    NodePtrVec distribution_parameters;
                    for (int col = 0; col < cols; col++) {
                        distribution_parameters.push_back(
                            cpd_parameters.at(i++));
                    }

                    shared_ptr<D> distribution =
                        make_shared<D>(distribution_parameters);
                    vector<shared_ptr<D>> distributions;
                    distributions.push_back(move(distribution));
                    CPDPtr cpd = make_shared<C>(index_metadatas, distributions);
                    const auto& cpd_owner =
                        this->rv_nodes.at(cpd_owner_label).at(row);
                    cpd_owner->add_cpd_template(move(cpd));
                }
            }
        }

        NumNodePtrVec
        JSONModelParser::split_constant_parameters(const string& str,
                                                   const string& delimiter) {
            NumNodePtrVec values;
            vector<string> tokens;
            boost::split(tokens, str, boost::is_any_of(delimiter));
            for (string token : tokens) {
                boost::trim(token);
                if (EXISTS(token, this->variables)) {
                    // If a multivalue variable is used as a constant parameter,
                    // only its first value will be used.
                    values.push_back(this->variables.at(token).front());
                }
                else {
                    double value = stod(token);
                    if (value == 0) {
                        value = EPSILON; // For numerical stability with logs.
                    }
                    values.push_back(make_shared<NumericNode>(value));
                }
            }

            return values;
        }

        template <class C, class D>
        void JSONModelParser::create_node_dependent_cpd(
            const nlohmann::json& json_cpd, const string& cpd_owner_label) {

            vector<MetadataPtr> index_metadatas;
            for (const string& index_node_label : json_cpd["index_nodes"]) {
                // Index nodes can only be data nodes,
                // and therefore, only have one copy in
                // the metadata list.
                index_metadatas.push_back(
                    this->rv_nodes.at(index_node_label).at(0)->get_metadata());
            }

            // There can be a mixture of constants and random variables
            vector<string> tokens;
            boost::split(
                tokens, (string)json_cpd["parameters"], boost::is_any_of(","));

            int num_distributions = 1;
            for (string& token : tokens) {
                boost::trim(token);
                if (EXISTS(token, this->rv_nodes)) {
                    num_distributions = this->rv_nodes.at(token).size();
                    break;
                }
            }

            vector<shared_ptr<D>> distributions;
            for (int i = 0; i < num_distributions; i++) {
                NodePtrVec parameters;
                for (string& token : tokens) {
                    boost::trim(token);
                    if (EXISTS(token, this->rv_nodes)) {
                        const auto& parameter_node =
                            this->rv_nodes.at(token)[i];
                        parameters.push_back(parameter_node);

                        // Add parameter as a parent of the cpd owner
                        const auto& cpd_owner_metadata =
                            this->rv_nodes.at(cpd_owner_label)
                                .at(0)
                                ->get_metadata();
                        const auto& parameter_metadata =
                            parameter_node->get_metadata();
                        bool time_crossing =
                            !parameter_metadata->is_single_time_link();
                        cpd_owner_metadata->add_parent_link(parameter_metadata,
                                                            time_crossing);
                    }
                    else {
                        if (EXISTS(token, this->variables)) {
                            if (this->variables.at(token).size() == 1) {
                                parameters.push_back(
                                    this->variables.at(token).front());
                            }
                            else {
                                // Multivalue variable
                                parameters.push_back(
                                    this->variables.at(token)[i]);
                            }
                        }
                        else {
                            double value = stod(token);
                            if (value == 0) {
                                value = EPSILON; // For numerical stability with
                                                 // logs.
                            }
                            parameters.push_back(
                                make_shared<NumericNode>(value));
                        }
                    }
                }

                // Create a distribution with the parameter nodes declared
                shared_ptr<D> distribution = make_shared<D>(parameters);
                distributions.push_back(distribution);
            }

            CPDPtr cpd = make_shared<C>(index_metadatas, distributions);
            const auto& cpd_owner = this->rv_nodes.at(cpd_owner_label).at(0);
            cpd_owner->add_cpd_template(move(cpd));
        }

    } // namespace model
} // namespace tomcat