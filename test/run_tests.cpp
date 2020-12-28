#define BOOST_TEST_MODULE TomcatModelTest

#include <memory>
#include <string>
#include <vector>

#include "boost/test/included/unit_test.hpp"
#include "eigen3/Eigen/Dense"
#include <gsl/gsl_rng.h>

#include "distribution/Categorical.h"
#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "pgm/NodeMetadata.h"
#include "pgm/RandomVariableNode.h"
#include "pgm/cpd/CPD.h"
#include "pgm/cpd/CategoricalCPD.h"
#include "pgm/cpd/DirichletCPD.h"
#include "pipeline/estimation/SumProductEstimator.h"
#include "pipeline/training/DBNSamplingTrainer.h"
#include "sampling/AncestralSampler.h"
#include "sampling/GibbsSampler.h"
#include "utils/Definitions.h"

using namespace tomcat::model;
using namespace std;
using namespace Eigen;

bool is_equal(const MatrixXd& m1,
              const MatrixXd& m2,
              double tolerance = 0.00001) {
    /**
     * This function checks if the elements of two matrices are equal within a
     * tolerance value.
     */

    for (int i = 0; i < m1.rows(); i++) {
        for (int j = 0; j < m1.cols(); j++) {
            if (abs(m1(i, j) - m2(i, j)) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

struct ModelConfig {
    /**
     * Small version of the ToMCAT model for testing. This version contains the
     * Training Condition (TC), State, Green, Yellow and Player's Belief about
     * the Environment (PBAE) nodes.
     */

    typedef shared_ptr<NodeMetadata> NodeMetadataPtr;
    typedef shared_ptr<CPD> CPDPtr;
    typedef shared_ptr<Categorical> CatPtr;
    typedef shared_ptr<DynamicBayesNet> DBNPtr;
    typedef shared_ptr<RandomVariableNode> RVNodePtr;

    // Node labels
    inline static const string TC = "TrainingCondition";
    inline static const string STATE = "State";
    inline static const string GREEN = "Green";
    inline static const string YELLOW = "Yellow";
    inline static const string PBAE = "PBAE";
    inline static const string THETA_TC = "Theta_TrainingCondition";
    inline static const string THETA_STATE = "Theta_State";
    inline static const string THETA_STATE_GIVEN_TC_PBAE_STATE =
        "Theta_State_gv_TC_PBAE_State";
    inline static const string PI_GREEN_GIVEN_STATE = "Pi_Green_gv_State";
    inline static const string PI_YELLOW_GIVEN_STATE = "Pi_Yellow_gv_State";
    inline static const string PI_PBAE = "Pi_PBAE";
    inline static const string PI_PBAE_GIVEN_PBAE = "Pi_PBAE_gv_PBAE";

    // Cardinality of the data nodes.
    int TC_CARDINALITY = 3;
    int PBAE_CARDINALITY = 2;
    int STATE_CARDINALITY = 3;
    int GREEN_CARDINALITY = 2;
    int YELLOW_CARDINALITY = 2;

    // Number of parameter nodes for all possible combinations of
    // parents' assignments.
    int NUM_PI_PBAE_GIVEN_PBAE = PBAE_CARDINALITY;
    int NUM_THETA_STATE_GIVEN_TC_PBAE_STATE =
        TC_CARDINALITY * PBAE_CARDINALITY * STATE_CARDINALITY;
    int NUM_PI_GREEN_GIVEN_STATE = STATE_CARDINALITY;
    int NUM_PI_YELLOW_GIVEN_STATE = STATE_CARDINALITY;

    struct NodeMetadataCollection {
        /**
         * Struct that contains the collection of metadatas for all nodes in
         * the DBN used for testing.
         */

        // Data nodes
        NodeMetadataPtr tc;
        NodeMetadataPtr pbae;
        NodeMetadataPtr state;
        NodeMetadataPtr green;
        NodeMetadataPtr yellow;

        // Parameter nodes
        NodeMetadataPtr theta_tc;
        NodeMetadataPtr pi_pbae;
        NodeMetadataPtr theta_state;

        // One node for each combination of parent assignment.
        vector<NodeMetadataPtr> pi_pbae_given_pbae;
        vector<NodeMetadataPtr> theta_state_given_tc_pbae_state;
        vector<NodeMetadataPtr> pi_green_given_state;
        vector<NodeMetadataPtr> pi_yellow_given_state;
    };

    struct CPDTableCollection {
        /**
         * This struct contains the collection of CPD tables for the data
         * nodes in the DBN used for testing. We will set the CPD tables
         * probabilities manually to construct the test cases based on them. The
         * CPD tables of the parameter nodes' priors will always be a matrix of
         * ones for the tests, so there's no need to store them in this struct.
         */

        MatrixXd tc_prior;
        MatrixXd pbae_prior;
        MatrixXd state_prior;
        MatrixXd pbae_given_pbae;
        MatrixXd state_given_tc_pbae_state;
        MatrixXd green_given_state;
        MatrixXd yellow_given_state;
    };

    struct CPDCollection {
        /**
         * This struct contains the collection of CPD's for all the nodes in
         * the DBN used for testing.
         */

        CPDPtr tc_prior;
        CPDPtr pbae_prior;
        CPDPtr state_prior;
        CPDPtr pbae_given_pbae;
        CPDPtr state_given_tc_pbae_state;
        CPDPtr green_given_state;
        CPDPtr yellow_given_state;

        // Parameter nodes
        CPDPtr theta_tc_prior;
        CPDPtr pi_pbae_prior;
        CPDPtr theta_state_prior;
        vector<CPDPtr> pi_pbae_given_pbae_prior;
        vector<CPDPtr> theta_state_given_tc_pbae_state_prior;
        vector<CPDPtr> pi_green_given_state_prior;
        vector<CPDPtr> pi_yellow_given_state_prior;
    };

    struct RVNodeCollection {
        /**
         * This struct contains the collection of all the nodes in the DBN
         * used for testing.
         */

        RVNodePtr tc;
        RVNodePtr pbae;
        RVNodePtr state;
        RVNodePtr green;
        RVNodePtr yellow;

        // Parameter nodes
        RVNodePtr theta_tc;
        RVNodePtr pi_pbae;
        RVNodePtr theta_state;
        vector<RVNodePtr> pi_pbae_given_pbae;
        vector<RVNodePtr> theta_state_given_tc_pbae_state;
        vector<RVNodePtr> pi_green_given_state;
        vector<RVNodePtr> pi_yellow_given_state;
    };

    ModelConfig() {}

    DBNPtr create_model(bool deterministic, bool trainable) {
        NodeMetadataCollection node_metadatas = this->create_node_metadatas();
        this->create_connections(node_metadatas);
        RVNodeCollection nodes = this->create_nodes(node_metadatas);
        CPDCollection cpds = this->create_cpds(nodes, deterministic);
        this->assign_cpds_to_nodes(nodes, cpds);

        if (!trainable) {
            this->freeze_parameters(nodes);
        }

        // Create model and add nodes to it.
        DBNPtr model;
        model = make_shared<DynamicBayesNet>();
        model->add_node_template(*nodes.tc);
        model->add_node_template(*nodes.pbae);
        model->add_node_template(*nodes.state);
        model->add_node_template(*nodes.green);
        model->add_node_template(*nodes.yellow);

        // Parameter nodes
        model->add_node_template(*nodes.theta_tc);
        model->add_node_template(*nodes.pi_pbae);
        model->add_node_template(*nodes.theta_state);
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            model->add_node_template(*nodes.pi_pbae_given_pbae[i]);
        }
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
            model->add_node_template(*nodes.theta_state_given_tc_pbae_state[i]);
        }
        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            model->add_node_template(*nodes.pi_green_given_state[i]);
        }
        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            model->add_node_template(*nodes.pi_yellow_given_state[i]);
        }

        return model;
    }

    NodeMetadataCollection create_node_metadatas() {
        NodeMetadataCollection metadatas;

        metadatas.tc = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                TC, false, false, true, 1, 1, TC_CARDINALITY));

        metadatas.pbae = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                PBAE, true, false, true, 0, 1, PBAE_CARDINALITY));

        metadatas.state = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                STATE, true, false, true, 0, 1, STATE_CARDINALITY));

        metadatas.green = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                GREEN, true, false, true, 1, 1, GREEN_CARDINALITY));

        metadatas.yellow = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                YELLOW, true, false, true, 1, 1, YELLOW_CARDINALITY));

        // Parameter nodes
        metadatas.theta_tc = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                THETA_TC, false, true, false, 1, TC_CARDINALITY));

        metadatas.pi_pbae = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                PI_PBAE, false, true, false, 1, PBAE_CARDINALITY));

        metadatas.theta_state = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                THETA_STATE, false, true, false, 1, STATE_CARDINALITY));

        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            stringstream label;
            label << PI_PBAE_GIVEN_PBAE << '_' << i;

            metadatas.pi_pbae_given_pbae.push_back(make_shared<NodeMetadata>(
                NodeMetadata::create_multiple_time_link_metadata(
                    label.str(), false, true, false, 1, PBAE_CARDINALITY)));
        }

        for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
            stringstream label;
            label << THETA_STATE_GIVEN_TC_PBAE_STATE << '_' << i;

            metadatas.theta_state_given_tc_pbae_state.push_back(make_shared<
                                                                NodeMetadata>(
                NodeMetadata::create_multiple_time_link_metadata(
                    label.str(), false, true, false, 1, STATE_CARDINALITY)));
        }

        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            stringstream label;
            label << PI_GREEN_GIVEN_STATE << '_' << i;

            metadatas.pi_green_given_state.push_back(make_shared<NodeMetadata>(
                NodeMetadata::create_multiple_time_link_metadata(
                    label.str(), false, true, false, 1, GREEN_CARDINALITY)));
        }

        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            stringstream label;
            label << PI_YELLOW_GIVEN_STATE << '_' << i;

            metadatas.pi_yellow_given_state.push_back(make_shared<NodeMetadata>(
                NodeMetadata::create_multiple_time_link_metadata(
                    label.str(), false, true, false, 1, YELLOW_CARDINALITY)));
        }

        return metadatas;
    }

    RVNodeCollection create_nodes(NodeMetadataCollection& node_metadatas) {
        RVNodeCollection nodes;

        nodes.tc = make_shared<RandomVariableNode>(node_metadatas.tc);
        nodes.pbae = make_shared<RandomVariableNode>(node_metadatas.pbae);
        nodes.state = make_shared<RandomVariableNode>(node_metadatas.state);
        nodes.green = make_shared<RandomVariableNode>(node_metadatas.green);
        nodes.yellow = make_shared<RandomVariableNode>(node_metadatas.yellow);

        // Parameter nodes
        nodes.theta_tc =
            make_shared<RandomVariableNode>(node_metadatas.theta_tc);
        nodes.pi_pbae = make_shared<RandomVariableNode>(node_metadatas.pi_pbae);
        nodes.theta_state =
            make_shared<RandomVariableNode>(node_metadatas.theta_state);

        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            nodes.pi_pbae_given_pbae.push_back(make_shared<RandomVariableNode>(
                node_metadatas.pi_pbae_given_pbae[i]));
        }

        for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
            nodes.theta_state_given_tc_pbae_state.push_back(
                make_shared<RandomVariableNode>(
                    node_metadatas.theta_state_given_tc_pbae_state[i]));
        }

        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            nodes.pi_green_given_state.push_back(
                make_shared<RandomVariableNode>(
                    node_metadatas.pi_green_given_state[i]));
        }

        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            nodes.pi_yellow_given_state.push_back(
                make_shared<RandomVariableNode>(
                    node_metadatas.pi_yellow_given_state[i]));
        }

        return nodes;
    }

    void create_connections(NodeMetadataCollection& node_metadatas) {
        node_metadatas.tc->add_parent_link(node_metadatas.theta_tc, false);

        node_metadatas.pbae->add_parent_link(node_metadatas.pbae, true);
        node_metadatas.pbae->add_parent_link(node_metadatas.pi_pbae, false);
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            node_metadatas.pbae->add_parent_link(
                node_metadatas.pi_pbae_given_pbae[i], true);
        }

        node_metadatas.state->add_parent_link(node_metadatas.tc, true);
        node_metadatas.state->add_parent_link(node_metadatas.state, true);
        node_metadatas.state->add_parent_link(node_metadatas.pbae, true);
        node_metadatas.state->add_parent_link(node_metadatas.theta_state,
                                              false);
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
            node_metadatas.state->add_parent_link(
                node_metadatas.theta_state_given_tc_pbae_state[i], true);
        }

        node_metadatas.green->add_parent_link(node_metadatas.state, false);
        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            node_metadatas.green->add_parent_link(
                node_metadatas.pi_green_given_state[i], false);
        }

        node_metadatas.yellow->add_parent_link(node_metadatas.state, false);
        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            node_metadatas.yellow->add_parent_link(
                node_metadatas.pi_yellow_given_state[i], false);
        }
    }

    CPDCollection create_cpds(RVNodeCollection& nodes, bool deterministic) {
        CPDCollection cpds;
        CPDTableCollection tables = this->create_cpd_tables(deterministic);

        // Training condition
        MatrixXd theta_tc_prior = MatrixXd::Ones(1, TC_CARDINALITY);
        cpds.theta_tc_prior =
            make_shared<DirichletCPD>(DirichletCPD({}, theta_tc_prior));

        vector<CatPtr> cat_distributions = {
            make_shared<Categorical>(nodes.theta_tc)};
        nodes.theta_tc->set_assignment(tables.tc_prior);
        CategoricalCPD cpd({}, cat_distributions);
        cpds.tc_prior = make_shared<CategoricalCPD>(move(cpd));

        // PBAE at time 0
        MatrixXd pi_pbae_prior = MatrixXd::Ones(1, PBAE_CARDINALITY);
        cpds.pi_pbae_prior =
            make_shared<DirichletCPD>(DirichletCPD({}, pi_pbae_prior));

        cat_distributions = {make_shared<Categorical>(nodes.pi_pbae)};
        nodes.pi_pbae->set_assignment(tables.pbae_prior);
        cpd = CategoricalCPD({}, cat_distributions);
        cpds.pbae_prior = make_shared<CategoricalCPD>(move(cpd));

        // State at time step 0
        MatrixXd theta_state_prior = MatrixXd::Ones(1, STATE_CARDINALITY);
        cpds.theta_state_prior =
            make_shared<DirichletCPD>(DirichletCPD({}, theta_state_prior));

        cat_distributions = {make_shared<Categorical>(nodes.theta_state)};
        nodes.theta_state->set_assignment(tables.state_prior);
        cpd = CategoricalCPD({}, cat_distributions);
        cpds.state_prior = make_shared<CategoricalCPD>(move(cpd));

        // PBAE given previous PBAE
        MatrixXd pi_pbae_given_pbae_prior = MatrixXd::Ones(1, PBAE_CARDINALITY);
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            cpds.pi_pbae_given_pbae_prior.push_back(make_shared<DirichletCPD>(
                DirichletCPD({}, pi_pbae_given_pbae_prior)));
        }

        cat_distributions.clear();
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            nodes.pi_pbae_given_pbae[i]->set_assignment(
                tables.pbae_given_pbae.row(i));
            cat_distributions.push_back(
                make_shared<Categorical>(nodes.pi_pbae_given_pbae[i]));
        }
        cpd = CategoricalCPD({nodes.pbae->get_metadata()}, cat_distributions);
        cpds.pbae_given_pbae = make_shared<CategoricalCPD>(move(cpd));

        // State given TC, PBAE and State
        MatrixXd THETA_STATE_GIVEN_TC_PBAE_state_prior =
            MatrixXd::Ones(1, STATE_CARDINALITY);
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
            cpds.theta_state_given_tc_pbae_state_prior.push_back(
                make_shared<DirichletCPD>(DirichletCPD({}, theta_state_prior)));
        }

        cat_distributions.clear();
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
            nodes.theta_state_given_tc_pbae_state[i]->set_assignment(
                tables.state_given_tc_pbae_state.row(i));
            cat_distributions.push_back(make_shared<Categorical>(
                nodes.theta_state_given_tc_pbae_state[i]));
        }
        cpd = CategoricalCPD({nodes.tc->get_metadata(),
                              nodes.pbae->get_metadata(),
                              nodes.state->get_metadata()},
                             cat_distributions);
        cpds.state_given_tc_pbae_state = make_shared<CategoricalCPD>(move(cpd));

        // Green given State
        MatrixXd pi_green_given_state_prior =
            MatrixXd::Ones(1, GREEN_CARDINALITY);
        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            cpds.pi_green_given_state_prior.push_back(make_shared<DirichletCPD>(
                DirichletCPD({}, pi_green_given_state_prior)));
        }

        cat_distributions.clear();
        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            nodes.pi_green_given_state[i]->set_assignment(
                tables.green_given_state.row(i));
            cat_distributions.push_back(
                make_shared<Categorical>(nodes.pi_green_given_state[i]));
        }
        cpd = CategoricalCPD({nodes.state->get_metadata()}, cat_distributions);
        cpds.green_given_state = make_shared<CategoricalCPD>(move(cpd));

        // Yellow given State
        MatrixXd pi_yellow_given_state_prior =
            MatrixXd::Ones(1, YELLOW_CARDINALITY);
        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            cpds.pi_yellow_given_state_prior.push_back(
                make_shared<DirichletCPD>(
                    DirichletCPD({}, pi_yellow_given_state_prior)));
        }

        cat_distributions.clear();
        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            nodes.pi_yellow_given_state[i]->set_assignment(
                tables.yellow_given_state.row(i));
            cat_distributions.push_back(
                make_shared<Categorical>(nodes.pi_yellow_given_state[i]));
        }
        cpd = CategoricalCPD({nodes.state->get_metadata()}, cat_distributions);
        cpds.yellow_given_state = make_shared<CategoricalCPD>(move(cpd));

        return cpds;
    }

    CPDTableCollection create_cpd_tables(bool deterministic) {
        CPDTableCollection tables;

        tables.tc_prior = MatrixXd(1, TC_CARDINALITY);
        if (deterministic) {
            tables.tc_prior << 1, 0, 0;
        }
        else {
            tables.tc_prior << 0.5, 0.3, 0.2;
        }

        tables.pbae_prior = MatrixXd(1, PBAE_CARDINALITY);
        if (deterministic) {
            tables.pbae_prior << 0, 1;
        }
        else {
            tables.pbae_prior << 0.3, 0.7;
        }

        tables.state_prior = MatrixXd(1, STATE_CARDINALITY);
        if (deterministic) {
            tables.state_prior << 1, 0, 0;
        }
        else {
            tables.state_prior << 0.3, 0.5, 0.2;
        }

        tables.pbae_given_pbae =
            MatrixXd(NUM_PI_PBAE_GIVEN_PBAE, PBAE_CARDINALITY);
        if (deterministic) {
            tables.pbae_given_pbae << 0, 1, 1, 0;
        }
        else {
            tables.pbae_given_pbae << 0.3, 0.7, 0.7, 0.3;
        }

        tables.state_given_tc_pbae_state =
            MatrixXd(NUM_THETA_STATE_GIVEN_TC_PBAE_STATE, STATE_CARDINALITY);
        if (deterministic) {
            tables.state_given_tc_pbae_state << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                1;
        }
        else {
            tables.state_given_tc_pbae_state << 0.5, 0.3, 0.2, 0.3, 0.5, 0.2,
                0.3, 0.2, 0.5, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3,
                0.5, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3,
                0.2, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3, 0.5, 0.2,
                0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5;
        }

        tables.green_given_state =
            MatrixXd(NUM_PI_GREEN_GIVEN_STATE, GREEN_CARDINALITY);
        if (deterministic) {
            tables.green_given_state << 0, 1, 1, 0, 0, 1;
        }
        else {
            tables.green_given_state << 0.3, 0.7, 0.7, 0.3, 0.3, 0.7;
        }

        tables.yellow_given_state =
            MatrixXd(NUM_PI_YELLOW_GIVEN_STATE, YELLOW_CARDINALITY);
        if (deterministic) {
            tables.yellow_given_state << 1, 0, 0, 1, 1, 0;
        }
        else {
            tables.yellow_given_state << 0.7, 0.3, 0.3, 0.7, 0.7, 0.3;
        }

        return tables;
    }

    void assign_cpds_to_nodes(RVNodeCollection& nodes, CPDCollection& cpds) {
        nodes.theta_tc->add_cpd_template(cpds.theta_tc_prior);
        nodes.tc->add_cpd_template(cpds.tc_prior);

        nodes.pi_pbae->add_cpd_template(cpds.pi_pbae_prior);
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            nodes.pi_pbae_given_pbae[i]->add_cpd_template(
                cpds.pi_pbae_given_pbae_prior[i]);
        }
        nodes.pbae->add_cpd_template(cpds.pbae_prior);
        nodes.pbae->add_cpd_template(cpds.pbae_given_pbae);

        nodes.theta_state->add_cpd_template(cpds.theta_state_prior);
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
            nodes.theta_state_given_tc_pbae_state[i]->add_cpd_template(
                cpds.theta_state_given_tc_pbae_state_prior[i]);
        }
        nodes.state->add_cpd_template(cpds.state_prior);
        nodes.state->add_cpd_template(cpds.state_given_tc_pbae_state);

        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            nodes.pi_green_given_state[i]->add_cpd_template(
                cpds.pi_green_given_state_prior[i]);
        }
        nodes.green->add_cpd_template(cpds.green_given_state);

        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            nodes.pi_yellow_given_state[i]->add_cpd_template(
                cpds.pi_yellow_given_state_prior[i]);
        }
        nodes.yellow->add_cpd_template(cpds.yellow_given_state);
    }

    void freeze_parameters(RVNodeCollection& nodes) {
        nodes.theta_tc->freeze();

        nodes.pi_pbae->freeze();
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            nodes.pi_pbae_given_pbae[i]->freeze();
        }

        nodes.theta_state->freeze();
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
            nodes.theta_state_given_tc_pbae_state[i]->freeze();
        }

        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            nodes.pi_green_given_state[i]->freeze();
        }

        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            nodes.pi_yellow_given_state[i]->freeze();
        }
    }
};

// Data generation

BOOST_AUTO_TEST_SUITE(data_generation)

BOOST_FIXTURE_TEST_CASE(complete, ModelConfig) {
    /**
     * This test case checks if data can be generated correctly, following
     * the distributions defined in the model. A deterministic model is used
     * so that the values generated can be known in advance.
     */

    DBNPtr model = create_model(true, false);

    model->unroll(4, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.sample(gen, 1);

    Tensor3 tcs = sampler.get_samples(TC);
    MatrixXd expected_tcs(1, 4);
    expected_tcs << NO_OBS, 0, 0, 0;
    BOOST_TEST(is_equal(tcs(0, 0), expected_tcs));

    Tensor3 pbaes = sampler.get_samples(PBAE);
    MatrixXd expected_pbaes(1, 4);
    expected_pbaes << 1, 0, 1, 0;
    BOOST_TEST(is_equal(pbaes(0, 0), expected_pbaes));

    Tensor3 states = sampler.get_samples(STATE);
    MatrixXd expected_states(1, 4);
    expected_states << 0, 1, 1, 2;
    BOOST_TEST(is_equal(states(0, 0), expected_states));

    Tensor3 greens = sampler.get_samples(GREEN);
    MatrixXd expected_greens(1, 4);
    expected_greens << NO_OBS, 0, 0, 1;
    BOOST_TEST(is_equal(greens(0, 0), expected_greens));

    Tensor3 yellows = sampler.get_samples(YELLOW);
    MatrixXd expected_yellows(1, 4);
    expected_yellows << NO_OBS, 1, 1, 0;
    BOOST_TEST(is_equal(yellows(0, 0), expected_yellows));
}

BOOST_FIXTURE_TEST_CASE(truncated, ModelConfig) {
    /**
     * This test case checks if data can be generated correctly up to a time
     * t smaller than the final time T that the DBN was unrolled into. A
     * deterministic model is used.
     */

    DBNPtr model = create_model(true, false);

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.set_max_time_step_to_sample(3);
    sampler.sample(gen, 1);

    Tensor3 tcs = sampler.get_samples(TC);
    MatrixXd expected_tcs(1, 4);
    expected_tcs << NO_OBS, 0, 0, 0;
    BOOST_TEST(is_equal(tcs(0, 0), expected_tcs));

    Tensor3 pbaes = sampler.get_samples(PBAE);
    MatrixXd expected_pbaes(1, 4);
    expected_pbaes << 1, 0, 1, 0;
    BOOST_TEST(is_equal(pbaes(0, 0), expected_pbaes));

    Tensor3 states = sampler.get_samples(STATE);
    MatrixXd expected_states(1, 4);
    expected_states << 0, 1, 1, 2;
    BOOST_TEST(is_equal(states(0, 0), expected_states));

    Tensor3 greens = sampler.get_samples(GREEN);
    MatrixXd expected_greens(1, 4);
    expected_greens << NO_OBS, 0, 0, 1;
    BOOST_TEST(is_equal(greens(0, 0), expected_greens));

    Tensor3 yellows = sampler.get_samples(YELLOW);
    MatrixXd expected_yellows(1, 4);
    expected_yellows << NO_OBS, 1, 1, 0;
    BOOST_TEST(is_equal(yellows(0, 0), expected_yellows));
}

BOOST_FIXTURE_TEST_CASE(heterogeneous, ModelConfig) {
    /**
     * This test case checks if samples are correctly generated and vary
     * according to the distributions defined in the DBN. A non-deterministic
     * model is used so samples can have different values.
     */

    DBNPtr model = create_model(false, false);

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    int equal_samples_until = 4;
    int num_samples = 10;

    AncestralSampler sampler(model);
    sampler.sample(gen, num_samples);

    MatrixXd tcs = sampler.get_samples(TC)(0, 0);
    MatrixXd first_tcs = tcs.row(0);
    MatrixXd equal_samples_tcs = first_tcs.replicate<10, 1>();
    MatrixXd cropped_tcs = tcs.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_tcs =
        equal_samples_tcs.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!is_equal(cropped_tcs, cropped_etc_samples_tcs));

    MatrixXd pbaes = sampler.get_samples(PBAE)(0, 0);
    MatrixXd first_pbaes = pbaes.row(0);
    MatrixXd equal_samples_pbaes = first_pbaes.replicate<10, 1>();
    MatrixXd cropped_pbaes =
        pbaes.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_pbaes =
        equal_samples_pbaes.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!is_equal(cropped_pbaes, cropped_etc_samples_pbaes));

    MatrixXd states = sampler.get_samples(STATE)(0, 0);
    MatrixXd first_states = states.row(0);
    MatrixXd equal_samples_states = first_states.replicate<10, 1>();
    MatrixXd cropped_states =
        states.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_states =
        equal_samples_states.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!is_equal(cropped_states, cropped_etc_samples_states));

    MatrixXd greens = sampler.get_samples(GREEN)(0, 0);
    MatrixXd first_greens = greens.row(0);
    MatrixXd equal_samples_greens = first_greens.replicate<10, 1>();
    MatrixXd cropped_greens =
        greens.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_greens =
        equal_samples_greens.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!is_equal(cropped_greens, cropped_etc_samples_greens));

    MatrixXd yellows = sampler.get_samples(YELLOW)(0, 0);
    MatrixXd first_yellows = yellows.row(0);
    MatrixXd equal_samples_yellows = first_yellows.replicate<10, 1>();
    MatrixXd cropped_yellows =
        yellows.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_yellows =
        equal_samples_yellows.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!is_equal(cropped_yellows, cropped_etc_samples_yellows));
}

BOOST_FIXTURE_TEST_CASE(homogeneous, ModelConfig) {
    /**
     * This test case checks if samples are correctly generated and present
     * the same values up to a certain time t. After this time, values are
     * generated independently and different samples can have different
     * values for the nodes in the DBN. A non-deterministic model is used.
     */

    DBNPtr model = create_model(false, false);

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    int equal_samples_until = 4;
    int num_samples = 10;

    AncestralSampler sampler(model);
    sampler.set_equal_samples_time_step_limit(equal_samples_until);
    sampler.sample(gen, num_samples);

    MatrixXd tcs = sampler.get_samples(TC)(0, 0);
    MatrixXd first_tcs = tcs.row(0);
    MatrixXd equal_samples_tcs = first_tcs.replicate<10, 1>();
    MatrixXd cropped_tcs = tcs.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_tcs =
        equal_samples_tcs.block(0, 0, num_samples, equal_samples_until);
    // Samples equal up to time 4 and above because tc don't change over time
    BOOST_TEST(is_equal(cropped_tcs, cropped_etc_samples_tcs));

    MatrixXd pbaes = sampler.get_samples(PBAE)(0, 0);
    MatrixXd first_pbaes = pbaes.row(0);
    MatrixXd equal_samples_pbaes = first_pbaes.replicate<10, 1>();
    MatrixXd cropped_pbaes =
        pbaes.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_pbaes =
        equal_samples_pbaes.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(is_equal(cropped_pbaes, cropped_etc_samples_pbaes));
    BOOST_TEST(!is_equal(pbaes, equal_samples_pbaes));

    // Samplers differ after time step 4 for the other nodes.
    MatrixXd states = sampler.get_samples(STATE)(0, 0);
    MatrixXd first_states = states.row(0);
    MatrixXd equal_samples_states = first_states.replicate<10, 1>();
    MatrixXd cropped_states =
        states.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_states =
        equal_samples_states.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(is_equal(cropped_states, cropped_etc_samples_states));
    BOOST_TEST(!is_equal(states, equal_samples_states));

    MatrixXd greens = sampler.get_samples(GREEN)(0, 0);
    MatrixXd first_greens = greens.row(0);
    MatrixXd equal_samples_greens = first_greens.replicate<10, 1>();
    MatrixXd cropped_greens =
        greens.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_greens =
        equal_samples_greens.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(is_equal(cropped_greens, cropped_etc_samples_greens));
    BOOST_TEST(!is_equal(greens, equal_samples_greens));

    MatrixXd yellows = sampler.get_samples(YELLOW)(0, 0);
    MatrixXd first_yellows = yellows.row(0);
    MatrixXd equal_samples_yellows = first_yellows.replicate<10, 1>();
    MatrixXd cropped_yellows =
        yellows.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_yellows =
        equal_samples_yellows.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(is_equal(cropped_yellows, cropped_etc_samples_yellows));
    BOOST_TEST(!is_equal(yellows, equal_samples_yellows));
}

BOOST_AUTO_TEST_SUITE_END()

// Training

BOOST_AUTO_TEST_SUITE(model_training)

BOOST_FIXTURE_TEST_CASE(gibbs_sampling, ModelConfig) {
    /**
     * This test case checks if the model can learn the parameters of a
     * non-deterministic model, given data generated from such a model.
     * Observations for node TC are not provided to the sampler to capture
     * the ability of the procedure to learn the parameters given that some
     * nodes are hidden.
     */

    DBNPtr oracle = create_model(false, false);
    oracle->unroll(20, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Generate a bunch of samples to train a model from the scratch.
    AncestralSampler sampler(oracle);
    sampler.sample(gen, 1000);

    DBNPtr model = create_model(false, true);
    model->unroll(20, true);

    shared_ptr<gsl_rng> gen_training(gsl_rng_alloc(gsl_rng_mt19937));
    shared_ptr<GibbsSampler> gibbs_sampler =
        make_shared<GibbsSampler>(model, 200);
    DBNSamplingTrainer trainer(gen_training, gibbs_sampler, 200);

    double tolerance = 0.05;
    CPDTableCollection tables = this->create_cpd_tables(false);

    // Check parameter learning when tc is not provided.
    EvidenceSet data;
    data.add_data(PBAE, sampler.get_samples(PBAE));
    data.add_data(STATE, sampler.get_samples(STATE));
    data.add_data(GREEN, sampler.get_samples(GREEN));
    data.add_data(YELLOW, sampler.get_samples(YELLOW));

    trainer.prepare();
    trainer.fit(data);
    model->get_nodes_by_label(THETA_TC)[0]->get_assignment();
    MatrixXd estimated_theta_tc =
        model->get_nodes_by_label(THETA_TC)[0]->get_assignment();
    stringstream msg;
    msg << "Estimated: [" << estimated_theta_tc << "]; Expected: ["
        << tables.tc_prior << "]";
    bool check = is_equal(estimated_theta_tc, tables.tc_prior, tolerance);
    BOOST_TEST(check, msg.str());


    MatrixXd estimated_pi_pbae =
        model->get_nodes_by_label(PI_PBAE)[0]->get_assignment();
    msg = stringstream();
    msg << "Estimated: [" << estimated_pi_pbae << "]; Expected: ["
        << tables.pbae_prior << "]";
    check = is_equal(estimated_pi_pbae, tables.pbae_prior, tolerance);
    BOOST_TEST(check, msg.str());

    MatrixXd estimated_theta_state =
        model->get_nodes_by_label(THETA_STATE)[0]->get_assignment();
    msg = stringstream();
    msg << "Estimated: [" << estimated_theta_state << "]; Expected: ["
        << tables.state_prior << "]";
    check = is_equal(estimated_theta_state, tables.state_prior, tolerance);
    BOOST_TEST(check, msg.str());

    for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
        stringstream label;
        label << PI_PBAE_GIVEN_PBAE << '_' << i;
        MatrixXd estimated_pi_pbae_given_pbae =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        msg = stringstream();
        msg << "Estimated: [" << estimated_pi_pbae_given_pbae
            << "]; Expected: [" << tables.pbae_given_pbae.row(i) << "]";
        check = is_equal(estimated_pi_pbae_given_pbae,
                         tables.pbae_given_pbae.row(i),
                         tolerance);
        BOOST_TEST(check, msg.str());
    }

    for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
        stringstream label;
        label << THETA_STATE_GIVEN_TC_PBAE_STATE << '_' << i;
        MatrixXd estimated_theta_state_given_tc_pbae_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        msg = stringstream();
        msg << "Estimated: [" << estimated_theta_state_given_tc_pbae_state
            << "]; Expected: [" << tables.state_given_tc_pbae_state.row(i)
            << "]";
        check = is_equal(estimated_theta_state_given_tc_pbae_state,
                         tables.state_given_tc_pbae_state.row(i),
                         tolerance);
        BOOST_TEST(check, msg.str());
    }

    for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
        stringstream label;
        label << PI_GREEN_GIVEN_STATE << '_' << i;

        MatrixXd estimated_pi_green_given_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        msg = stringstream();
        msg << "Estimated: [" << estimated_pi_green_given_state
            << "]; Expected: [" << tables.green_given_state.row(i) << "]";
        check = is_equal(estimated_pi_green_given_state,
                         tables.green_given_state.row(i),
                         tolerance);
        BOOST_TEST(check, msg.str());
    }

    for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
        stringstream label;
        label << PI_YELLOW_GIVEN_STATE << '_' << i;

        MatrixXd estimated_pi_yellow_given_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        msg = stringstream();
        msg << "Estimated: [" << estimated_pi_yellow_given_state
            << "]; Expected: [" << tables.yellow_given_state.row(i) << "]";
        check = is_equal(estimated_pi_yellow_given_state,
                         tables.yellow_given_state.row(i),
                         tolerance);
        BOOST_TEST(check, msg.str());
    }
}

BOOST_AUTO_TEST_SUITE_END()

// Inference

BOOST_AUTO_TEST_SUITE(estimation)

BOOST_FIXTURE_TEST_CASE(sum_product, ModelConfig) {
    /**
     * This test case checks if the sum-product procedure can estimate
     * correctly the marginal probabilities over time of the nodes Green and
     * TC from a non-deterministic model.
     */

    DBNPtr deterministic_model = create_model(true, false);
    deterministic_model->unroll(4, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // One sample from the deterministic model will be used for estimation.
    AncestralSampler sampler(deterministic_model);
    sampler.sample(gen, 1);

    DBNPtr pre_trained_model = create_model(false, false);
    pre_trained_model->unroll(4, true);
    SumProductEstimator green_estimator_h1(
        pre_trained_model, 1, GREEN, VectorXd::Constant(1, 1));
    SumProductEstimator yellow_estimator_h1(
        pre_trained_model, 1, YELLOW, VectorXd::Constant(1, 1));
    SumProductEstimator green_estimator_h3(
        pre_trained_model, 3, GREEN, VectorXd::Constant(1, 1));
    SumProductEstimator yellow_estimator_h3(
        pre_trained_model, 3, YELLOW, VectorXd::Constant(1, 1));
    SumProductEstimator tc_estimator(pre_trained_model, 0, TC);

    // Green node is observed
    EvidenceSet data;
    data.add_data(GREEN, sampler.get_samples(GREEN));

    green_estimator_h1.set_show_progress(false);
    green_estimator_h1.prepare();
    green_estimator_h1.estimate(data);
    yellow_estimator_h1.set_show_progress(false);
    yellow_estimator_h1.prepare();
    yellow_estimator_h1.estimate(data);
    green_estimator_h3.set_show_progress(false);
    green_estimator_h3.prepare();
    green_estimator_h3.estimate(data);
    yellow_estimator_h3.set_show_progress(false);
    yellow_estimator_h3.prepare();
    yellow_estimator_h3.estimate(data);
    tc_estimator.set_show_progress(false);
    tc_estimator.prepare();
    tc_estimator.estimate(data);

    MatrixXd green_estimates_h1 =
        green_estimator_h1.get_estimates().estimates[0];
    MatrixXd yellow_estimates_h1 =
        yellow_estimator_h1.get_estimates().estimates[0];
    MatrixXd green_estimates_h3 =
        green_estimator_h3.get_estimates().estimates[0];
    MatrixXd yellow_estimates_h3 =
        yellow_estimator_h3.get_estimates().estimates[0];
    vector<MatrixXd> tc_estimates = tc_estimator.get_estimates().estimates;

    MatrixXd expected_green_h1(1, 4);
    expected_green_h1 << 0.56788, 0.56589, 0.566867, 0.565993;
    BOOST_TEST(is_equal(green_estimates_h1, expected_green_h1));

    MatrixXd expected_yellow_h1(1, 4);
    expected_yellow_h1 << 0.43212, 0.43411, 0.433133, 0.434007;
    BOOST_TEST(is_equal(yellow_estimates_h1, expected_yellow_h1));

    MatrixXd expected_green_h3(1, 4);
    expected_green_h3 << 0.91875, 0.918454, 0.918545, 0.918483;
    BOOST_TEST(is_equal(green_estimates_h3, expected_green_h3));

    MatrixXd expected_yellow_h3(1, 4);
    expected_yellow_h3 << 0.818174, 0.818918, 0.818499, 0.818842;
    BOOST_TEST(is_equal(yellow_estimates_h3, expected_yellow_h3));

    MatrixXd expected_tc(3, 3);
    expected_tc << 0.500324, 0.506335, 0.504282, 0.294363, 0.286289, 0.291487,
        0.205313, 0.207376, 0.204232;
    BOOST_TEST(is_equal(tc_estimates[0], expected_tc.row(0)));
    BOOST_TEST(is_equal(tc_estimates[1], expected_tc.row(1)));
    BOOST_TEST(is_equal(tc_estimates[2], expected_tc.row(2)));
}

BOOST_AUTO_TEST_SUITE_END()
