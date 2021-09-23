#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils/Definitions.h"
#include <nlohmann/json.hpp>

/**
 * This file creates a DBN from a the model's specifications in a JSON file.
 */

namespace tomcat {
    namespace model {

        class DynamicBayesNet;

        class JSONModelParser {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------
            JSONModelParser(const std::string& model_filepath);
            ~JSONModelParser();

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            DynamicBayesNet create_model();

          private:
            //------------------------------------------------------------------
            // Types
            //------------------------------------------------------------------
            typedef std::unordered_map<std::string, std::vector<MetadataPtr>>
                MetadataMap;
            // Keep it sorted to preserve order of computation across machines.
            typedef std::map<std::string, RVNodePtrVec> RVMap;
            typedef std::unordered_map<std::string, NumNodePtr> VarMap;
            typedef std::unordered_set<std::string> NodeSet;
            typedef std::unordered_map<std::string, std::pair<int, int>>
                ParamMapConfig;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Check to see if the definition is valid.
             */
            void check_model_definition() const;

            /**
             * Create a map of variables defined in the file, if any.
             */
            void create_variables();

            /**
             * Creates metadatas for timer nodes and adds to the list of
             * metadatas.
             */
            void create_timer_metadatas();

            /**
             * Creates metadatas for data nodes and adds to the list of
             * metadatas.
             */
            void create_data_metadatas();

            /**
             * Creates metadatas for parameter nodes and adds to the list of
             * metadatas.
             */
            void create_parameter_metadatas();

            /**
             * Gets the set of nodes that have at least one child that is
             * replicable over time.
             *
             * @return Set of nodes
             */
            NodeSet get_nodes_with_replicable_child();

            /**
             * Gets the number of copies we need to create of parameter nodes
             * based on the number of rows in the CPDs they show up, and their
             * time step based on the minimum initial time step of the nodes
             * that own these CPDs.
             *
             * @return Map between parameter node labels and their number of
             * copies and time steps.
             */
            ParamMapConfig get_num_param_copies_and_time_steps();

            /**
             * Updates the number of copies we need to create of parameter nodes
             * based on the number of rows in the CPDs they show up, and their
             * time step based on the minimum initial time step of the nodes
             * that own these CPDs.
             *
             * @param parameter_labels: set of parameter labels declared in the
             * file
             * @param json_cpds: json object containing a list of CPDs
             * @param cpd_owner_label: label of the node that owns the CPD
             * @param param_config: map between parameter node labels and their
             * number of copies and time steps.
             *
             * @return number of copies and time steps
             */
            void update_num_param_copies_and_time_steps(
                const std::unordered_set<std::string>& parameter_labels,
                const nlohmann::json& json_cpds,
                const std::string& cpd_owner_label,
                ParamMapConfig& param_config);

            /**
             * Creates a series of CPD templates and associates them to the
             * corresponding node.
             *
             * @param cpd: json with the CPD specifications
             * @param cpd_owner_label: label of the node that owns the CPD
             */
            void create_cpds(const nlohmann::json& json_cpds,
                             const std::string& cpd_owner_label);

            /**
             * Creates a constant CPD and associates it to the corresponding
             * node.
             *
             * @tparam C: Class of the CPD according to the CPD distribution
             * @param cpd: json with the CPD specifications
             * @param cpd_owner_label: label of the node that owns the CPD
             * @param cols: number of columns in the CPD. It can either be the
             * cardinality of the node or the number of parameters of a specific
             * distribution
             */
            template <class C, class D>
            void create_constant_cpd(const nlohmann::json& json_cpd,
                                     const std::string& cpd_owner_label,
                                     int cols);

            /**
             * Splits a string into a list of numeric nodes given a delimiter.
             *
             * @param str: string to be split
             * @param delimiter: delimiter
             *
             * @return list of tokens
             */
            NumNodePtrVec
            split_constant_parameters(const std::string& str,
                                      const std::string& delimiter = ",");

            /**
             * Creates a CPD that depend on other nodes (parameter nodes) and
             * associates it to the corresponding node.
             *
             * @param C: Class of the CPD according to the CPD distribution
             * @param D: Class of the distribution that comprises the CPD
             * @param cpd: json with the CPD specifications
             * @param cpd_owner_label: label of the node that owns the CPD
             */
            template <class C, class D>
            void create_node_dependent_cpd(const nlohmann::json& json_cpd,
                                           const std::string& cpd_owner_label);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            nlohmann::json json_model;
            MetadataMap metadatas;
            RVMap rv_nodes;
            VarMap variables;
        };

    } // namespace model
} // namespace tomcat