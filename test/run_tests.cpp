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

struct ModelConfig {
    /**
     * Small version of the ToMCAT model for testing.
     * This version contains the Training Condition, State, Green, Yellow and
     * Player's Belief about the Environment (PBAE) node.
     */

    typedef shared_ptr<NodeMetadata> NodeMetadataPtr;
    typedef shared_ptr<CPD> CPDPtr;
    typedef shared_ptr<Categorical> CatPtr;
    typedef shared_ptr<DynamicBayesNet> DBNPtr;
    typedef shared_ptr<RandomVariableNode> RVNodePtr;
    typedef vector<NodeMetadataPtr> ParentsMetadata;

    // Node names
    inline static const string Q = "TrainingCondition";
    inline static const string STATE = "State";
    inline static const string SG = "Green";
    inline static const string SY = "Yellow";
    inline static const string PBAE = "PBAE";
    inline static const string THETA_Q = "Theta_TrainingCondition";
    inline static const string THETA_STATE = "Theta_State";
    inline static const string THETA_STATE_GIVEN_Q_PBAE_STATE =
        "Theta_State_gv_Q_PBAE_State";
    inline static const string PI_SG_GIVEN_STATE = "Pi_Green_gv_State";
    inline static const string PI_SY_GIVEN_STATE = "Pi_Yellow_gv_State";
    inline static const string PI_PBAE = "Pi_PBAE";
    inline static const string PI_PBAE_GIVEN_PBAE = "Pi_PBAE_gv_PBAE";

    struct NodeMetadataCollection {
        NodeMetadataPtr q_metadata;
        NodeMetadataPtr state_metadata;
        NodeMetadataPtr green_metadata;
        NodeMetadataPtr yellow_metadata;
        NodeMetadataPtr pbae_metadata;

        // Parameter nodes
        NodeMetadataPtr theta_q_metadata;
        NodeMetadataPtr theta_state_metadata;
        vector<NodeMetadataPtr> theta_state_given_q_pbae_state_metadata;
        vector<NodeMetadataPtr> pi_green_given_state_metadata;
        vector<NodeMetadataPtr> pi_yellow_given_state_metadata;
        NodeMetadataPtr pi_pbae_metadata;
        vector<NodeMetadataPtr> pi_pbae_given_pbae_metadata;
    };

    struct CPDTableCollection {
        MatrixXd q_prior_cpd_table;
        MatrixXd state_prior_cpd_table;
        MatrixXd state_given_q_pbae_state_cpd_table;
        MatrixXd green_given_state_cpd_table;
        MatrixXd yellow_given_state_cpd_table;
        MatrixXd pbae_prior_cpd_table;
        MatrixXd pbae_given_pbae_cpd_table;
    };

    struct CPDCollection {
        CPDPtr q_prior_cpd;
        CPDPtr state_prior_cpd;
        CPDPtr state_given_q_pbae_state_cpd;
        CPDPtr green_given_state_cpd;
        CPDPtr yellow_given_state_cpd;
        CPDPtr pbae_prior_cpd;
        CPDPtr pbae_given_pbae_cpd;

        CPDPtr theta_q_prior_cpd;
        CPDPtr theta_state_prior_cpd;
        vector<CPDPtr> theta_state_given_q_pbae_state_prior_cpd;
        CPDPtr pi_pbae_prior_cpd;
        vector<CPDPtr> pi_pbae_given_pbae_prior_cpd;
        vector<CPDPtr> pi_green_given_state_prior_cpd;
        vector<CPDPtr> pi_yellow_given_state_prior_cpd;
    };

    struct RVNodeCollection {
        RVNodePtr q;
        RVNodePtr state;
        RVNodePtr pbae;
        RVNodePtr green;
        RVNodePtr yellow;

        // Parameter nodes
        RVNodePtr theta_q;
        RVNodePtr theta_state;
        vector<RVNodePtr> theta_state_given_q_pbae_state;
        RVNodePtr pi_pbae;
        vector<RVNodePtr> pi_pbae_given_pbae;
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

        DBNPtr model;
        model = make_shared<DynamicBayesNet>();
        model->add_node_template(*nodes.q);
        model->add_node_template(*nodes.state);
        model->add_node_template(*nodes.pbae);
        model->add_node_template(*nodes.green);
        model->add_node_template(*nodes.yellow);
        model->add_node_template(*nodes.theta_q);
        model->add_node_template(*nodes.theta_state);
        for (int i = 0; i < 18; i++) {
            model->add_node_template(*nodes.theta_state_given_q_pbae_state[i]);
        }
        model->add_node_template(*nodes.pi_pbae);
        for (int i = 0; i < 2; i++) {
            model->add_node_template(*nodes.pi_pbae_given_pbae[i]);
        }
        for (int i = 0; i < 3; i++) {
            model->add_node_template(*nodes.pi_green_given_state[i]);
            model->add_node_template(*nodes.pi_yellow_given_state[i]);
        }

        return model;
    }

    NodeMetadataCollection create_node_metadatas() {
        NodeMetadataCollection metadatas;

        metadatas.q_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                Q, false, false, true, 1, 1, 3));

        metadatas.state_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                STATE, true, false, true, 0, 1, 3));

        metadatas.green_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                SG, true, false, true, 1, 1, 2));

        metadatas.yellow_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                SY, true, false, true, 1, 1, 2));

        metadatas.pbae_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                PBAE, true, false, true, 0, 1, 2));

        // Parameter nodes
        metadatas.theta_q_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                THETA_Q, false, true, false, 1, 3));

        metadatas.theta_state_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                THETA_STATE, false, true, false, 1, 3));

        for (int i = 0; i < 18; i++) {
            stringstream label;
            label << THETA_STATE_GIVEN_Q_PBAE_STATE << '_' << i;

            metadatas.theta_state_given_q_pbae_state_metadata.push_back(
                make_shared<NodeMetadata>(
                    NodeMetadata::create_multiple_time_link_metadata(
                        label.str(), false, true, false, 1, 3)));
        }

        for (int i = 0; i < 3; i++) {
            stringstream label_green;
            stringstream label_yellow;
            label_green << PI_SG_GIVEN_STATE << '_' << i;
            label_yellow << PI_SY_GIVEN_STATE << '_' << i;

            metadatas.pi_green_given_state_metadata.push_back(
                make_shared<NodeMetadata>(
                    NodeMetadata::create_multiple_time_link_metadata(
                        label_green.str(), false, true, false, 1, 2)));

            metadatas.pi_yellow_given_state_metadata.push_back(
                make_shared<NodeMetadata>(
                    NodeMetadata::create_multiple_time_link_metadata(
                        label_yellow.str(), false, true, false, 1, 2)));
        }

        metadatas.pi_pbae_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                PI_PBAE, false, true, false, 1, 2));

        for (int i = 0; i < 2; i++) {
            stringstream label;
            label << PI_PBAE_GIVEN_PBAE << '_' << i;

            metadatas.pi_pbae_given_pbae_metadata.push_back(
                make_shared<NodeMetadata>(
                    NodeMetadata::create_multiple_time_link_metadata(
                        label.str(), false, true, false, 1, 2)));
        }

        return metadatas;
    }

    RVNodeCollection create_nodes(NodeMetadataCollection& node_metadatas) {
        RVNodeCollection nodes;

        nodes.q = make_shared<RandomVariableNode>(node_metadatas.q_metadata);
        nodes.state =
            make_shared<RandomVariableNode>(node_metadatas.state_metadata);
        nodes.pbae =
            make_shared<RandomVariableNode>(node_metadatas.pbae_metadata);
        nodes.green =
            make_shared<RandomVariableNode>(node_metadatas.green_metadata);
        nodes.yellow =
            make_shared<RandomVariableNode>(node_metadatas.yellow_metadata);

        nodes.theta_q =
            make_shared<RandomVariableNode>(node_metadatas.theta_q_metadata);
        nodes.theta_state = make_shared<RandomVariableNode>(
            node_metadatas.theta_state_metadata);

        // One parameter node for each combination of training condition, PBAE
        // and previous state.
        for (int i = 0; i < 18; i++) {
            nodes.theta_state_given_q_pbae_state.push_back(
                make_shared<RandomVariableNode>(
                    node_metadatas.theta_state_given_q_pbae_state_metadata[i]));
        }
        nodes.pi_pbae =
            make_shared<RandomVariableNode>(node_metadatas.pi_pbae_metadata);

        // One parameter node for each previous PBAE
        for (int i = 0; i < 2; i++) {
            nodes.pi_pbae_given_pbae.push_back(make_shared<RandomVariableNode>(
                node_metadatas.pi_pbae_given_pbae_metadata[i]));
        }

        for (int i = 0; i < 3; i++) {
            nodes.pi_green_given_state.push_back(
                make_shared<RandomVariableNode>(
                    node_metadatas.pi_green_given_state_metadata[i]));
            nodes.pi_yellow_given_state.push_back(
                make_shared<RandomVariableNode>(
                    node_metadatas.pi_yellow_given_state_metadata[i]));
        }

        return nodes;
    }

    void create_connections(NodeMetadataCollection& node_metadatas) {
        node_metadatas.q_metadata->add_parent_link(
            node_metadatas.theta_q_metadata, false);
        node_metadatas.state_metadata->add_parent_link(
            node_metadatas.theta_state_metadata, false);
        for (int i = 0; i < 18; i++) {
            node_metadatas.state_metadata->add_parent_link(
                node_metadatas.theta_state_given_q_pbae_state_metadata[i],
                true);
        }
        node_metadatas.state_metadata->add_parent_link(
            node_metadatas.q_metadata, true);
        node_metadatas.state_metadata->add_parent_link(
            node_metadatas.state_metadata, true);
        node_metadatas.state_metadata->add_parent_link(
            node_metadatas.pbae_metadata, true);
        node_metadatas.pbae_metadata->add_parent_link(
            node_metadatas.pi_pbae_metadata, false);
        for (int i = 0; i < 2; i++) {
            node_metadatas.pbae_metadata->add_parent_link(
                node_metadatas.pi_pbae_given_pbae_metadata[i], true);
        }
        node_metadatas.pbae_metadata->add_parent_link(
            node_metadatas.pbae_metadata, true);
        for (int i = 0; i < 3; i++) {
            node_metadatas.green_metadata->add_parent_link(
                node_metadatas.pi_green_given_state_metadata[i], false);
            node_metadatas.yellow_metadata->add_parent_link(
                node_metadatas.pi_yellow_given_state_metadata[i], false);
        }
        node_metadatas.green_metadata->add_parent_link(
            node_metadatas.state_metadata, false);
        node_metadatas.yellow_metadata->add_parent_link(
            node_metadatas.state_metadata, false);
    }

    CPDCollection create_cpds(RVNodeCollection& nodes, bool deterministic) {
        CPDCollection cpds;
        CPDTableCollection tables = this->create_cpd_tables(deterministic);

        // Training condition
        MatrixXd theta_q_prior_cpd_table = MatrixXd::Ones(1, 3);
        cpds.theta_q_prior_cpd = make_shared<DirichletCPD>(
            DirichletCPD({}, theta_q_prior_cpd_table));

        vector<CatPtr> cat_distributions = {
            make_shared<Categorical>(nodes.theta_q)};
        nodes.theta_q->set_assignment(tables.q_prior_cpd_table);
        CategoricalCPD cpd({}, cat_distributions);
        cpds.q_prior_cpd = make_shared<CategoricalCPD>(move(cpd));

        // State at time step 0
        MatrixXd theta_state_prior_cpd_table = MatrixXd::Ones(1, 3);
        cpds.theta_state_prior_cpd = make_shared<DirichletCPD>(
            DirichletCPD({}, theta_state_prior_cpd_table));

        cat_distributions = {make_shared<Categorical>(nodes.theta_state)};
        nodes.theta_state->set_assignment(tables.state_prior_cpd_table);
        cpd = CategoricalCPD({}, cat_distributions);
        cpds.state_prior_cpd = make_shared<CategoricalCPD>(move(cpd));

        // State given TC, PBAE and State
        MatrixXd theta_state_given_q_pbae_state_prior_cpd_table =
            MatrixXd::Ones(1, 3);
        for (int i = 0; i < 18; i++) {
            cpds.theta_state_given_q_pbae_state_prior_cpd.push_back(
                make_shared<DirichletCPD>(
                    DirichletCPD({}, theta_state_prior_cpd_table)));
        }

        cat_distributions.clear();
        for (int i = 0; i < 18; i++) {
            nodes.theta_state_given_q_pbae_state[i]->set_assignment(
                tables.state_given_q_pbae_state_cpd_table.row(i));
            cat_distributions.push_back(make_shared<Categorical>(
                nodes.theta_state_given_q_pbae_state[i]));
        }
        cpd = CategoricalCPD({nodes.q->get_metadata(),
                              nodes.pbae->get_metadata(),
                              nodes.state->get_metadata()},
                             cat_distributions);
        cpds.state_given_q_pbae_state_cpd =
            make_shared<CategoricalCPD>(move(cpd));

        // PBAE at time 0
        MatrixXd pi_pbae_prior_cpd_table = MatrixXd::Ones(1, 2);
        cpds.pi_pbae_prior_cpd = make_shared<DirichletCPD>(
            DirichletCPD({}, pi_pbae_prior_cpd_table));

        cat_distributions = {make_shared<Categorical>(nodes.pi_pbae)};
        nodes.pi_pbae->set_assignment(tables.pbae_prior_cpd_table);
        cpd = CategoricalCPD({}, cat_distributions);
        cpds.pbae_prior_cpd = make_shared<CategoricalCPD>(move(cpd));

        // PBAE given previous PBAE
        MatrixXd pi_pbae_given_pbae_prior_cpd_table = MatrixXd::Ones(1, 2);
        for (int i = 0; i < 2; i++) {
            cpds.pi_pbae_given_pbae_prior_cpd.push_back(
                make_shared<DirichletCPD>(
                    DirichletCPD({}, pi_pbae_given_pbae_prior_cpd_table)));
        }

        cat_distributions.clear();
        for (int i = 0; i < 2; i++) {
            nodes.pi_pbae_given_pbae[i]->set_assignment(
                tables.pbae_given_pbae_cpd_table.row(i));
            cat_distributions.push_back(
                make_shared<Categorical>(nodes.pi_pbae_given_pbae[i]));
        }
        cpd = CategoricalCPD({nodes.pbae->get_metadata()}, cat_distributions);
        cpds.pbae_given_pbae_cpd = make_shared<CategoricalCPD>(move(cpd));

        // Green given State
        MatrixXd pi_green_given_state_prior_cpd_table = MatrixXd::Ones(1, 2);
        for (int i = 0; i < 3; i++) {
            cpds.pi_green_given_state_prior_cpd.push_back(
                make_shared<DirichletCPD>(
                    DirichletCPD({}, pi_green_given_state_prior_cpd_table)));
        }

        cat_distributions.clear();
        for (int i = 0; i < 3; i++) {
            nodes.pi_green_given_state[i]->set_assignment(
                tables.green_given_state_cpd_table.row(i));
            cat_distributions.push_back(
                make_shared<Categorical>(nodes.pi_green_given_state[i]));
        }
        cpd = CategoricalCPD({nodes.state->get_metadata()}, cat_distributions);
        cpds.green_given_state_cpd = make_shared<CategoricalCPD>(move(cpd));

        // Yellow given State
        MatrixXd pi_yellow_given_state_prior_cpd_table = MatrixXd::Ones(1, 2);
        for (int i = 0; i < 3; i++) {
            cpds.pi_yellow_given_state_prior_cpd.push_back(
                make_shared<DirichletCPD>(
                    DirichletCPD({}, pi_yellow_given_state_prior_cpd_table)));
        }

        cat_distributions.clear();
        for (int i = 0; i < 3; i++) {
            nodes.pi_yellow_given_state[i]->set_assignment(
                tables.yellow_given_state_cpd_table.row(i));
            cat_distributions.push_back(
                make_shared<Categorical>(nodes.pi_yellow_given_state[i]));
        }
        cpd = CategoricalCPD({nodes.state->get_metadata()}, cat_distributions);
        cpds.yellow_given_state_cpd = make_shared<CategoricalCPD>(move(cpd));

        return cpds;
    }

    CPDTableCollection create_cpd_tables(bool deterministic) {
        CPDTableCollection tables;

        // Training condition
        tables.q_prior_cpd_table = MatrixXd(1, 3);
        if (deterministic) {
            tables.q_prior_cpd_table << 1, 0, 0;
        }
        else {
            tables.q_prior_cpd_table << 0.5, 0.3, 0.2;
        }

        tables.state_prior_cpd_table = MatrixXd(1, 3);
        if (deterministic) {
            tables.state_prior_cpd_table << 1, 0, 0;
        }
        else {
            tables.state_prior_cpd_table << 0.3, 0.5, 0.2;
        }

        tables.state_given_q_pbae_state_cpd_table = MatrixXd(18, 3);
        if (deterministic) {
            tables.state_given_q_pbae_state_cpd_table << 1, 0, 0, 0, 1, 0, 0, 0,
                1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1;
        }
        else {
            tables.state_given_q_pbae_state_cpd_table << 0.5, 0.3, 0.2, 0.3,
                0.5, 0.2, 0.3, 0.2, 0.5, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3,
                0.2, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3, 0.2, 0.5,
                0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3,
                0.5, 0.2, 0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5;
        }

        tables.pbae_prior_cpd_table = MatrixXd(1, 2);
        if (deterministic) {
            tables.pbae_prior_cpd_table << 0, 1;
        }
        else {
            tables.pbae_prior_cpd_table << 0.3, 0.7;
        }

        tables.pbae_given_pbae_cpd_table = MatrixXd(2, 2);
        if (deterministic) {
            tables.pbae_given_pbae_cpd_table << 0, 1, 1, 0;
        }
        else {
            tables.pbae_given_pbae_cpd_table << 0.3, 0.7, 0.7, 0.3;
        }

        tables.green_given_state_cpd_table = MatrixXd(3, 2);
        if (deterministic) {
            tables.green_given_state_cpd_table << 0, 1, 1, 0, 0, 1;
        }
        else {
            tables.green_given_state_cpd_table << 0.3, 0.7, 0.7, 0.3, 0.3, 0.7;
        }

        tables.yellow_given_state_cpd_table = MatrixXd(3, 2);
        if (deterministic) {
            tables.yellow_given_state_cpd_table << 1, 0, 0, 1, 1, 0;
        }
        else {
            tables.yellow_given_state_cpd_table << 0.7, 0.3, 0.3, 0.7, 0.7, 0.3;
        }

        return tables;
    }

    void assign_cpds_to_nodes(RVNodeCollection& nodes, CPDCollection& cpds) {
        nodes.theta_q->add_cpd_template(cpds.theta_q_prior_cpd);
        nodes.q->add_cpd_template(cpds.q_prior_cpd);

        nodes.theta_state->add_cpd_template(cpds.theta_state_prior_cpd);
        for (int i = 0; i < 18; i++) {
            nodes.theta_state_given_q_pbae_state[i]->add_cpd_template(
                cpds.theta_state_given_q_pbae_state_prior_cpd[i]);
        }
        nodes.state->add_cpd_template(cpds.state_prior_cpd);
        nodes.state->add_cpd_template(cpds.state_given_q_pbae_state_cpd);

        nodes.pi_pbae->add_cpd_template(cpds.pi_pbae_prior_cpd);
        for (int i = 0; i < 2; i++) {
            nodes.pi_pbae_given_pbae[i]->add_cpd_template(
                cpds.pi_pbae_given_pbae_prior_cpd[i]);
        }
        nodes.pbae->add_cpd_template(cpds.pbae_prior_cpd);
        nodes.pbae->add_cpd_template(cpds.pbae_given_pbae_cpd);

        for (int i = 0; i < 3; i++) {
            nodes.pi_green_given_state[i]->add_cpd_template(
                cpds.pi_green_given_state_prior_cpd[i]);
            nodes.pi_yellow_given_state[i]->add_cpd_template(
                cpds.pi_yellow_given_state_prior_cpd[i]);
        }
        nodes.green->add_cpd_template(cpds.green_given_state_cpd);
        nodes.yellow->add_cpd_template(cpds.yellow_given_state_cpd);
    }

    void freeze_parameters(RVNodeCollection& nodes) {
        nodes.theta_q->freeze();
        nodes.theta_state->freeze();
        for (int i = 0; i < 18; i++) {
            nodes.theta_state_given_q_pbae_state[i]->freeze();
        }
        nodes.pi_pbae->freeze();
        for (int i = 0; i < 2; i++) {
            nodes.pi_pbae_given_pbae[i]->freeze();
        }
        for (int i = 0; i < 3; i++) {
            nodes.pi_green_given_state[i]->freeze();
            nodes.pi_yellow_given_state[i]->freeze();
        }
    }
};

// Data generation

BOOST_AUTO_TEST_SUITE(data_generation)

BOOST_FIXTURE_TEST_CASE(complete, ModelConfig) {
    DBNPtr model = create_model(true, false);

    model->unroll(4, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.sample(gen, 1);

    Tensor3 qs = sampler.get_samples(Q);
    MatrixXd expected_qs(1, 4);
    expected_qs << NO_OBS, 0, 0, 0;
    BOOST_TEST(qs(0, 0).isApprox(expected_qs));

    Tensor3 states = sampler.get_samples(STATE);
    MatrixXd expected_states(1, 4);
    expected_states << 0, 1, 1, 2;
    BOOST_TEST(states(0, 0).isApprox(expected_states));

    Tensor3 pbaes = sampler.get_samples(PBAE);
    MatrixXd expected_pbaes(1, 4);
    expected_pbaes << 1, 0, 1, 0;
    BOOST_TEST(pbaes(0, 0).isApprox(expected_pbaes));

    Tensor3 greens = sampler.get_samples(SG);
    MatrixXd expected_greens(1, 4);
    expected_greens << NO_OBS, 0, 0, 1;
    BOOST_TEST(greens(0, 0).isApprox(expected_greens));

    Tensor3 yellows = sampler.get_samples(SY);
    MatrixXd expected_yellows(1, 4);
    expected_yellows << NO_OBS, 1, 1, 0;
    BOOST_TEST(yellows(0, 0).isApprox(expected_yellows));
}

BOOST_FIXTURE_TEST_CASE(truncated, ModelConfig) {
    DBNPtr model = create_model(true, false);

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.set_max_time_step_to_sample(3);
    sampler.sample(gen, 1);

    Tensor3 qs = sampler.get_samples(Q);
    MatrixXd expected_qs(1, 4);
    expected_qs << NO_OBS, 0, 0, 0;
    BOOST_TEST(qs(0, 0).isApprox(expected_qs));

    Tensor3 states = sampler.get_samples(STATE);
    MatrixXd expected_states(1, 4);
    expected_states << 0, 1, 1, 2;
    BOOST_TEST(states(0, 0).isApprox(expected_states));

    Tensor3 pbaes = sampler.get_samples(PBAE);
    MatrixXd expected_pbaes(1, 4);
    expected_pbaes << 1, 0, 1, 0;
    BOOST_TEST(pbaes(0, 0).isApprox(expected_pbaes));

    Tensor3 greens = sampler.get_samples(SG);
    MatrixXd expected_greens(1, 4);
    expected_greens << NO_OBS, 0, 0, 1;
    BOOST_TEST(greens(0, 0).isApprox(expected_greens));

    Tensor3 yellows = sampler.get_samples(SY);
    MatrixXd expected_yellows(1, 4);
    expected_yellows << NO_OBS, 1, 1, 0;
    BOOST_TEST(yellows(0, 0).isApprox(expected_yellows));
}

BOOST_FIXTURE_TEST_CASE(heterogeneous, ModelConfig) {
    DBNPtr model = create_model(false, false);

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    int equal_samples_until = 4;
    int num_samples = 10;

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.sample(gen, num_samples);

    MatrixXd qs = sampler.get_samples(Q)(0, 0);
    MatrixXd first_qs = qs.row(0);
    MatrixXd equal_samples_qs = first_qs.replicate<10, 1>();
    MatrixXd cropped_qs = qs.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_eq_samples_qs =
        equal_samples_qs.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!cropped_qs.isApprox(cropped_eq_samples_qs));

    MatrixXd states = sampler.get_samples(STATE)(0, 0);
    MatrixXd first_states = states.row(0);
    MatrixXd equal_samples_states = first_states.replicate<10, 1>();
    MatrixXd cropped_states =
        states.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_eq_samples_states =
        equal_samples_states.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!cropped_states.isApprox(cropped_eq_samples_states));

    MatrixXd pbaes = sampler.get_samples(PBAE)(0, 0);
    MatrixXd first_pbaes = pbaes.row(0);
    MatrixXd equal_samples_pbaes = first_pbaes.replicate<10, 1>();
    MatrixXd cropped_pbaes =
        pbaes.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_eq_samples_pbaes =
        equal_samples_pbaes.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!cropped_pbaes.isApprox(cropped_eq_samples_pbaes));

    MatrixXd greens = sampler.get_samples(SG)(0, 0);
    MatrixXd first_greens = greens.row(0);
    MatrixXd equal_samples_greens = first_greens.replicate<10, 1>();
    MatrixXd cropped_greens =
        greens.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_eq_samples_greens =
        equal_samples_greens.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!cropped_greens.isApprox(cropped_eq_samples_greens));

    MatrixXd yellows = sampler.get_samples(SY)(0, 0);
    MatrixXd first_yellows = yellows.row(0);
    MatrixXd equal_samples_yellows = first_yellows.replicate<10, 1>();
    MatrixXd cropped_yellows =
        yellows.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_eq_samples_yellows =
        equal_samples_yellows.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!cropped_yellows.isApprox(cropped_eq_samples_yellows));
}

BOOST_FIXTURE_TEST_CASE(homogeneous, ModelConfig) {
    DBNPtr model = create_model(false, false);

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    int equal_samples_until = 4;
    int num_samples = 10;

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.set_equal_samples_time_step_limit(equal_samples_until);
    sampler.sample(gen, num_samples);

    MatrixXd qs = sampler.get_samples(Q)(0, 0);
    MatrixXd first_qs = qs.row(0);
    MatrixXd equal_samples_qs = first_qs.replicate<10, 1>();
    MatrixXd cropped_qs = qs.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_eq_samples_qs =
        equal_samples_qs.block(0, 0, num_samples, equal_samples_until);
    // Samples equal up to time 4 and above because Q don't change over time
    BOOST_TEST(cropped_qs.isApprox(cropped_eq_samples_qs));

    // Samplers differ after time step 4 for the other nodes.
    MatrixXd states = sampler.get_samples(STATE)(0, 0);
    MatrixXd first_states = states.row(0);
    MatrixXd equal_samples_states = first_states.replicate<10, 1>();
    MatrixXd cropped_states =
        states.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_eq_samples_states =
        equal_samples_states.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(cropped_states.isApprox(cropped_eq_samples_states));
    BOOST_TEST(!states.isApprox(equal_samples_states));

    MatrixXd pbaes = sampler.get_samples(PBAE)(0, 0);
    MatrixXd first_pbaes = pbaes.row(0);
    MatrixXd equal_samples_pbaes = first_pbaes.replicate<10, 1>();
    MatrixXd cropped_pbaes =
        pbaes.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_eq_samples_pbaes =
        equal_samples_pbaes.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(cropped_pbaes.isApprox(cropped_eq_samples_pbaes));
    BOOST_TEST(!pbaes.isApprox(equal_samples_pbaes));

    MatrixXd greens = sampler.get_samples(SG)(0, 0);
    MatrixXd first_greens = greens.row(0);
    MatrixXd equal_samples_greens = first_greens.replicate<10, 1>();
    MatrixXd cropped_greens =
        greens.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_eq_samples_greens =
        equal_samples_greens.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(cropped_greens.isApprox(cropped_eq_samples_greens));
    BOOST_TEST(!greens.isApprox(equal_samples_greens));

    MatrixXd yellows = sampler.get_samples(SY)(0, 0);
    MatrixXd first_yellows = yellows.row(0);
    MatrixXd equal_samples_yellows = first_yellows.replicate<10, 1>();
    MatrixXd cropped_yellows =
        yellows.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_eq_samples_yellows =
        equal_samples_yellows.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(cropped_yellows.isApprox(cropped_eq_samples_yellows));
    BOOST_TEST(!yellows.isApprox(equal_samples_yellows));
}

BOOST_AUTO_TEST_SUITE_END()

// Model training

BOOST_AUTO_TEST_SUITE(model_training)

BOOST_FIXTURE_TEST_CASE(gibbs_sampling, ModelConfig) {
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

    double tolerance = 0.15;
    CPDTableCollection tables = this->create_cpd_tables(false);

    // Check parameter learning when Q is not provided.
    EvidenceSet data;
    data.add_data(SG, sampler.get_samples(SG));
    data.add_data(SY, sampler.get_samples(SY));
    data.add_data(STATE, sampler.get_samples(STATE));
    data.add_data(PBAE, sampler.get_samples(PBAE));

    trainer.prepare();
    trainer.fit(data);
    model->get_nodes_by_label(THETA_Q)[0]->get_assignment();
    MatrixXd estimated_theta_q =
        model->get_nodes_by_label(THETA_Q)[0]->get_assignment();
    BOOST_TEST(estimated_theta_q.isApprox(tables.q_prior_cpd_table, tolerance));

    MatrixXd estimated_theta_state =
        model->get_nodes_by_label(THETA_STATE)[0]->get_assignment();
    BOOST_TEST(estimated_theta_state.isApprox(tables.state_prior_cpd_table,
                                              tolerance));

    MatrixXd estimated_pi_pbae =
        model->get_nodes_by_label(PI_PBAE)[0]->get_assignment();
    BOOST_TEST(
        estimated_pi_pbae.isApprox(tables.pbae_prior_cpd_table, tolerance));

    for (int i = 0; i < 18; i++) {
        stringstream label;
        label << THETA_STATE_GIVEN_Q_PBAE_STATE << '_' << i;
        MatrixXd estimated_theta_state_given_q_pbae_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        BOOST_TEST(estimated_theta_state_given_q_pbae_state.isApprox(
            tables.state_given_q_pbae_state_cpd_table.row(i), tolerance));

        if (!estimated_theta_state_given_q_pbae_state.isApprox(
                tables.state_given_q_pbae_state_cpd_table.row(i), tolerance)) {
            cout << estimated_theta_state_given_q_pbae_state << endl;
            cout << tables.state_given_q_pbae_state_cpd_table.row(i) << endl
                 << endl;
        }
    }

    for (int i = 0; i < 2; i++) {
        stringstream label;
        label << PI_PBAE_GIVEN_PBAE << '_' << i;
        MatrixXd estimated_pi_pbae_given_pbae =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        BOOST_TEST(estimated_pi_pbae_given_pbae.isApprox(
            tables.pbae_given_pbae_cpd_table.row(i), tolerance));
    }

    for (int i = 0; i < 3; i++) {
        stringstream label_green;
        stringstream label_yellow;
        label_green << PI_SG_GIVEN_STATE << '_' << i;
        label_yellow << PI_SY_GIVEN_STATE << '_' << i;

        MatrixXd estimated_pi_sg_given_state =
            model->get_nodes_by_label(label_green.str())[0]->get_assignment();
        BOOST_TEST(estimated_pi_sg_given_state.isApprox(
            tables.green_given_state_cpd_table.row(i), tolerance));

        MatrixXd estimated_pi_sy_given_state =
            model->get_nodes_by_label(label_yellow.str())[0]->get_assignment();
        BOOST_TEST(estimated_pi_sy_given_state.isApprox(
            tables.yellow_given_state_cpd_table.row(i), tolerance));
    }
}

BOOST_AUTO_TEST_SUITE_END()

// Inference

BOOST_AUTO_TEST_SUITE(estimation)

BOOST_FIXTURE_TEST_CASE(sum_product, ModelConfig) {
    DBNPtr deterministic_model = create_model(true, false);
    deterministic_model->unroll(4, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // One sample from the deterministic model will be used for estimation.
    AncestralSampler sampler(deterministic_model);
    sampler.sample(gen, 1);

    DBNPtr pre_trained_model = create_model(false, false);
    pre_trained_model->unroll(4, true);
    SumProductEstimator green_estimator_h1(
        pre_trained_model, 1, SG, VectorXd::Constant(1, 1));
    SumProductEstimator yellow_estimator_h1(
        pre_trained_model, 1, SY, VectorXd::Constant(1, 1));
    SumProductEstimator green_estimator_h3(
        pre_trained_model, 3, SG, VectorXd::Constant(1, 1));
    SumProductEstimator yellow_estimator_h3(
        pre_trained_model, 3, SY, VectorXd::Constant(1, 1));
    SumProductEstimator q_estimator(pre_trained_model, 0, Q);

    // Green node is observed
    EvidenceSet data;
    data.add_data(SG, sampler.get_samples(SG));

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
    q_estimator.set_show_progress(false);
    q_estimator.prepare();
    q_estimator.estimate(data);

    MatrixXd green_estimates_h1 =
        green_estimator_h1.get_estimates().estimates[0];
    MatrixXd yellow_estimates_h1 =
        yellow_estimator_h1.get_estimates().estimates[0];
    MatrixXd green_estimates_h3 =
        green_estimator_h3.get_estimates().estimates[0];
    MatrixXd yellow_estimates_h3 =
        yellow_estimator_h3.get_estimates().estimates[0];
    vector<MatrixXd> q_estimates = q_estimator.get_estimates().estimates;

    double tolerance = 0.00001;

    MatrixXd expected_green_h1(1, 4);
    expected_green_h1 << 0.56788, 0.56589, 0.566867, 0.565993;
    BOOST_TEST(green_estimates_h1.isApprox(expected_green_h1, tolerance));

    MatrixXd expected_yellow_h1(1, 4);
    expected_yellow_h1 << 0.43212, 0.43411, 0.433133, 0.434007;
    BOOST_TEST(yellow_estimates_h1.isApprox(expected_yellow_h1, tolerance));

    MatrixXd expected_green_h3(1, 4);
    expected_green_h3 << 0.91875, 0.918454, 0.918545, 0.918483;
    BOOST_TEST(green_estimates_h3.isApprox(expected_green_h3, tolerance));

    MatrixXd expected_yellow_h3(1, 4);
    expected_yellow_h3 << 0.818174, 0.818918, 0.818499, 0.818842;
    BOOST_TEST(yellow_estimates_h3.isApprox(expected_yellow_h3, tolerance));

    MatrixXd expected_q(3, 3);
    expected_q << 0.500324, 0.506335, 0.504282, 0.294363, 0.286289, 0.291487,
        0.205313, 0.207376, 0.204232;
    BOOST_TEST(q_estimates[0].isApprox(expected_q.row(0), tolerance));
    BOOST_TEST(q_estimates[1].isApprox(expected_q.row(1), tolerance));
    BOOST_TEST(q_estimates[2].isApprox(expected_q.row(2), tolerance));
}

BOOST_AUTO_TEST_SUITE_END()
