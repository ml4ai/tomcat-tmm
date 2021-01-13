#pragma once

#include "distribution/Categorical.h"
#include "distribution/Poisson.h"
#include "pgm/DynamicBayesNet.h"
#include "pgm/NodeMetadata.h"
#include "pgm/TimerNode.h"
#include "pgm/RandomVariableNode.h"
#include "pgm/cpd/CPD.h"
#include "pgm/cpd/CategoricalCPD.h"
#include "pgm/cpd/DirichletCPD.h"
#include "pgm/cpd/GammaCPD.h"
#include "pgm/cpd/PoissonCPD.h"
#include "utils/Definitions.h"

using namespace tomcat::model;
using namespace std;
using namespace Eigen;

class MockPoisson : public Poisson {
  public:
    MockPoisson(const std::shared_ptr<Node>& lambda) : Poisson(lambda) {}

    VectorXd sample(const std::shared_ptr<gsl_rng>& random_generator,
                    int parameter_idx) const override {

        // Return the rounded mean of the Poisson distribution as the sampled
        // value
        int rounde_mean =
            this->parameters[0]->get_assignment()(parameter_idx, 0);
        return VectorXd::Constant(1, rounde_mean);
    }

    std::unique_ptr<Distribution> clone() const override {
        unique_ptr<MockPoisson> new_distribution =
            make_unique<MockPoisson>(*this);
        new_distribution->parameters[0] =
            new_distribution->parameters[0]->clone();

        return new_distribution;
    }
};

typedef shared_ptr<NodeMetadata> NodeMetadataPtr;
typedef shared_ptr<CPD> CPDPtr;
typedef shared_ptr<Categorical> CatPtr;
typedef shared_ptr<Poisson> PoiPtr;
typedef shared_ptr<DynamicBayesNet> DBNPtr;
typedef shared_ptr<RandomVariableNode> RVNodePtr;
typedef shared_ptr<TimerNode> TimerNodePtr;

struct HMM {
    /**
     * Small version of the ToMCAT model for testing. This version contains the
     * Training Condition (TC), State, Green, Yellow and Player's Belief about
     * the Environment (PBAE) nodes.
     */

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

    HMM() {}

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
        model->add_node_template(nodes.tc);
        model->add_node_template(nodes.pbae);
        model->add_node_template(nodes.state);
        model->add_node_template(nodes.green);
        model->add_node_template(nodes.yellow);

        // Parameter nodes
        model->add_node_template(nodes.theta_tc);
        model->add_node_template(nodes.pi_pbae);
        model->add_node_template(nodes.theta_state);
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            model->add_node_template(nodes.pi_pbae_given_pbae[i]);
        }
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
            model->add_node_template(nodes.theta_state_given_tc_pbae_state[i]);
        }
        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            model->add_node_template(nodes.pi_green_given_state[i]);
        }
        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            model->add_node_template(nodes.pi_yellow_given_state[i]);
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
            NodeMetadata::create_single_time_link_metadata(
                THETA_TC, true, false, 1, TC_CARDINALITY));

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
        node_metadatas.pbae->add_parent_link(node_metadatas.pi_pbae, true);
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            node_metadatas.pbae->add_parent_link(
                node_metadatas.pi_pbae_given_pbae[i], true);
        }

        node_metadatas.state->add_parent_link(node_metadatas.tc, true);
        node_metadatas.state->add_parent_link(node_metadatas.state, true);
        node_metadatas.state->add_parent_link(node_metadatas.pbae, true);
        node_metadatas.state->add_parent_link(node_metadatas.theta_state, true);
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
            node_metadatas.state->add_parent_link(
                node_metadatas.theta_state_given_tc_pbae_state[i], true);
        }

        node_metadatas.green->add_parent_link(node_metadatas.state, false);
        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            node_metadatas.green->add_parent_link(
                node_metadatas.pi_green_given_state[i], true);
        }

        node_metadatas.yellow->add_parent_link(node_metadatas.state, false);
        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            node_metadatas.yellow->add_parent_link(
                node_metadatas.pi_yellow_given_state[i], true);
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

struct HSMM {
    /**
     * Small version of the semi-Markov ToMCAT model for testing. This version
     * contains the Training Condition (TC), State, Green, Yellow, Player's
     * Belief about the Environment (PBAE) nodes and a timer node associated
     * with the transition of the node State. The timer depends on the
     * Training Condition, PBAE and State and the State only depends on the
     * previous state and its timer.
     */

    // Node labels
    inline static const string TC = "TrainingCondition";
    inline static const string STATE = "State";
    inline static const string GREEN = "Green";
    inline static const string YELLOW = "Yellow";
    inline static const string PBAE = "PBAE";
    inline static const string TIMER = "Timer";
    inline static const string THETA_TC = "Theta_TrainingCondition";
    inline static const string THETA_STATE = "Theta_State";
    inline static const string THETA_STATE_GIVEN_STATE = "Theta_State_gv_State";
    inline static const string PI_GREEN_GIVEN_STATE = "Pi_Green_gv_State";
    inline static const string PI_YELLOW_GIVEN_STATE = "Pi_Yellow_gv_State";
    inline static const string PI_PBAE = "Pi_PBAE";
    inline static const string PI_PBAE_GIVEN_PBAE = "Pi_PBAE_gv_PBAE";
    inline static const string LAMBDA_TIMER_GIVEN_TC_PBAE_STATE =
        "Lambda_Timer_gv_TC_PBAE_State";

    // Cardinality of the data nodes.
    int TC_CARDINALITY = 3;
    int PBAE_CARDINALITY = 2;
    int STATE_CARDINALITY = 3;
    int GREEN_CARDINALITY = 2;
    int YELLOW_CARDINALITY = 2;

    // Number of parameter nodes for all possible combinations of
    // parents' assignments.
    int NUM_PI_PBAE_GIVEN_PBAE = PBAE_CARDINALITY;
    int NUM_THETA_STATE_GIVEN_STATE = STATE_CARDINALITY;
    int NUM_LAMBDA_TIMER_GIVEN_TC_PBAE_STATE =
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
        NodeMetadataPtr timer;

        // Parameter nodes
        NodeMetadataPtr theta_tc;
        NodeMetadataPtr pi_pbae;
        NodeMetadataPtr theta_state;

        // One node for each combination of parent assignment.
        vector<NodeMetadataPtr> pi_pbae_given_pbae;
        vector<NodeMetadataPtr> theta_state_given_state;
        vector<NodeMetadataPtr> pi_green_given_state;
        vector<NodeMetadataPtr> pi_yellow_given_state;
        vector<NodeMetadataPtr> lambda_timer_given_tc_pbae_state;
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
        MatrixXd state_given_state;
        MatrixXd green_given_state;
        MatrixXd yellow_given_state;
        MatrixXd timer_given_tc_pbae_state; // Poisson means
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
        CPDPtr state_given_state;
        CPDPtr green_given_state;
        CPDPtr yellow_given_state;
        CPDPtr timer_given_tc_pbae_state;

        // Parameter nodes
        CPDPtr theta_tc_prior;
        CPDPtr pi_pbae_prior;
        CPDPtr theta_state_prior;
        vector<CPDPtr> pi_pbae_given_pbae_prior;
        vector<CPDPtr> theta_state_given_state_prior;
        vector<CPDPtr> pi_green_given_state_prior;
        vector<CPDPtr> pi_yellow_given_state_prior;
        vector<CPDPtr> lambda_timer_given_tc_pbae_state_prior;
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
        TimerNodePtr timer;

        // Parameter nodes
        RVNodePtr theta_tc;
        RVNodePtr pi_pbae;
        RVNodePtr theta_state;
        vector<RVNodePtr> pi_pbae_given_pbae;
        vector<RVNodePtr> theta_state_given_state;
        vector<RVNodePtr> pi_green_given_state;
        vector<RVNodePtr> pi_yellow_given_state;
        vector<RVNodePtr> lambda_timer_given_tc_pbae_state;
    };

    HSMM() {}

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
        model->add_node_template(nodes.tc);
        model->add_node_template(nodes.pbae);
        model->add_node_template(nodes.state);
        model->add_node_template(nodes.green);
        model->add_node_template(nodes.yellow);
        model->add_node_template(nodes.timer);

        // Parameter nodes
        model->add_node_template(nodes.theta_tc);
        model->add_node_template(nodes.pi_pbae);
        model->add_node_template(nodes.theta_state);
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            model->add_node_template(nodes.pi_pbae_given_pbae[i]);
        }
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE; i++) {
            model->add_node_template(nodes.theta_state_given_state[i]);
        }
        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            model->add_node_template(nodes.pi_green_given_state[i]);
        }
        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            model->add_node_template(nodes.pi_yellow_given_state[i]);
        }
        for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_PBAE_STATE; i++) {
            model->add_node_template(
                nodes.lambda_timer_given_tc_pbae_state[i]);
        }

        return model;
    }

    NodeMetadataCollection create_node_metadatas() {
        NodeMetadataCollection metadatas;

        metadatas.tc = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                TC, false, false, true, 0, 1, TC_CARDINALITY));

        metadatas.pbae = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                PBAE, true, false, true, 0, 1, PBAE_CARDINALITY));

        metadatas.timer = make_shared<NodeMetadata>(
            NodeMetadata::create_timer_metadata(TIMER, 0));

        metadatas.state = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                STATE, true, false, true, 0, 1, STATE_CARDINALITY));
        metadatas.state->set_timer_metadata(metadatas.timer);

        metadatas.green = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                GREEN, true, false, true, 1, 1, GREEN_CARDINALITY));

        metadatas.yellow = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                YELLOW, true, false, true, 1, 1, YELLOW_CARDINALITY));

        // Parameter nodes
        metadatas.theta_tc = make_shared<NodeMetadata>(
            NodeMetadata::create_single_time_link_metadata(
                THETA_TC, true, false, 0, TC_CARDINALITY));

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

        for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE; i++) {
            stringstream label;
            label << THETA_STATE_GIVEN_STATE << '_' << i;

            metadatas.theta_state_given_state.push_back(make_shared<
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

        for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_PBAE_STATE; i++) {
            stringstream label;
            label << LAMBDA_TIMER_GIVEN_TC_PBAE_STATE << '_' << i;

            metadatas.lambda_timer_given_tc_pbae_state.push_back(
                make_shared<NodeMetadata>(
                    NodeMetadata::create_multiple_time_link_metadata(
                        label.str(), false, true, false, 1, 1)));
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
        nodes.timer = make_shared<TimerNode>(node_metadatas.timer);

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

        for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE; i++) {
            nodes.theta_state_given_state.push_back(
                make_shared<RandomVariableNode>(
                    node_metadatas.theta_state_given_state[i]));
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

        for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_PBAE_STATE; i++) {
            nodes.lambda_timer_given_tc_pbae_state.push_back(
                make_shared<RandomVariableNode>(
                    node_metadatas.lambda_timer_given_tc_pbae_state[i]));
        }

        return nodes;
    }

    void create_connections(NodeMetadataCollection& node_metadatas) {
        node_metadatas.tc->add_parent_link(node_metadatas.theta_tc, false);

        node_metadatas.pbae->add_parent_link(node_metadatas.pbae, true);
        node_metadatas.pbae->add_parent_link(node_metadatas.pi_pbae, true);
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            node_metadatas.pbae->add_parent_link(
                node_metadatas.pi_pbae_given_pbae[i], true);
        }

        node_metadatas.state->add_parent_link(node_metadatas.state, true);
        node_metadatas.state->add_parent_link(node_metadatas.theta_state, true);
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE; i++) {
            node_metadatas.state->add_parent_link(
                node_metadatas.theta_state_given_state[i], true);
        }

        node_metadatas.green->add_parent_link(node_metadatas.state, false);
        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            node_metadatas.green->add_parent_link(
                node_metadatas.pi_green_given_state[i], true);
        }

        node_metadatas.yellow->add_parent_link(node_metadatas.state, false);
        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            node_metadatas.yellow->add_parent_link(
                node_metadatas.pi_yellow_given_state[i], true);
        }

        node_metadatas.timer->add_parent_link(node_metadatas.tc, true);
        node_metadatas.timer->add_parent_link(node_metadatas.pbae, false);
        node_metadatas.timer->add_parent_link(node_metadatas.state, false);
        for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_PBAE_STATE; i++) {
            node_metadatas.state->add_parent_link(
                node_metadatas.lambda_timer_given_tc_pbae_state[i], true);
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

        // State given State
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE; i++) {
            MatrixXd theta_state_given_state_prior =
                MatrixXd::Ones(1, STATE_CARDINALITY);
            theta_state_given_state_prior(0, i) = EPSILON; // Transition to the
            // same state will be handled by the timer.
            cpds.theta_state_given_state_prior.push_back(
                make_shared<DirichletCPD>(DirichletCPD({}, theta_state_prior)));
        }

        cat_distributions.clear();
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE; i++) {
            nodes.theta_state_given_state[i]->set_assignment(
                tables.state_given_state.row(i));
            cat_distributions.push_back(
                make_shared<Categorical>(nodes.theta_state_given_state[i]));
        }
        cpd = CategoricalCPD({nodes.state->get_metadata()}, cat_distributions);
        cpds.state_given_state = make_shared<CategoricalCPD>(move(cpd));

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

        // Timer given TC, PBAE and State
        MatrixXd lambda_timer_given_state_prior =
            MatrixXd::Ones(1, 2); // Gamma(1, 1)
        for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_PBAE_STATE; i++) {
            cpds.lambda_timer_given_tc_pbae_state_prior.push_back(
                make_shared<GammaCPD>(
                    GammaCPD({}, lambda_timer_given_state_prior)));
        }

        vector<PoiPtr> poi_distributions;
        for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_PBAE_STATE; i++) {
            nodes.lambda_timer_given_tc_pbae_state[i]->set_assignment(
                tables.timer_given_tc_pbae_state.row(i));
            poi_distributions.push_back(make_shared<MockPoisson>(
                nodes.lambda_timer_given_tc_pbae_state[i]));
        }
        PoissonCPD poi_cpd = PoissonCPD({nodes.tc->get_metadata(),
                                         nodes.pbae->get_metadata(),
                                         nodes.state->get_metadata()},
                                        poi_distributions);
        cpds.timer_given_tc_pbae_state = make_shared<PoissonCPD>(move(poi_cpd));

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

        tables.state_given_state =
            MatrixXd(NUM_THETA_STATE_GIVEN_STATE, STATE_CARDINALITY);
        if (deterministic) {
            tables.state_given_state << 0, 1, 0, 0, 0, 1, 1, 0, 0;
        }
        else {
            tables.state_given_state << EPSILON, 0.8, 0.2, 0.2, EPSILON, 0.8,
                0.8, 0.2, EPSILON;
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

        // Lambdas for all possible parent combinations
        tables.timer_given_tc_pbae_state =
            MatrixXd(NUM_LAMBDA_TIMER_GIVEN_TC_PBAE_STATE, 1);
        tables.timer_given_tc_pbae_state << 0, 2, 4, 1, 3, 5, 2, 4, 1, 3, 5, 0,
            4, 1, 3, 5, 0, 2;

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
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE; i++) {
            nodes.theta_state_given_state[i]->add_cpd_template(
                cpds.theta_state_given_state_prior[i]);
        }
        nodes.state->add_cpd_template(cpds.state_prior);
        nodes.state->add_cpd_template(cpds.state_given_state);

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

        for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_PBAE_STATE; i++) {
            nodes.lambda_timer_given_tc_pbae_state[i]->add_cpd_template(
                cpds.lambda_timer_given_tc_pbae_state_prior[i]);
        }
        nodes.timer->add_cpd_template(cpds.timer_given_tc_pbae_state);
    }

    void freeze_parameters(RVNodeCollection& nodes) {
        nodes.theta_tc->freeze();

        nodes.pi_pbae->freeze();
        for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
            nodes.pi_pbae_given_pbae[i]->freeze();
        }

        nodes.theta_state->freeze();
        for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE; i++) {
            nodes.theta_state_given_state[i]->freeze();
        }

        for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
            nodes.pi_green_given_state[i]->freeze();
        }

        for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
            nodes.pi_yellow_given_state[i]->freeze();
        }

        for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_PBAE_STATE; i++) {
            nodes.lambda_timer_given_tc_pbae_state[i]->freeze();
        }
    }
};