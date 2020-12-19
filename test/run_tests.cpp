#define BOOST_TEST_MODULE TomcatModelTest

#include <memory>
#include <string>
#include <vector>

#include "boost/test/included/unit_test.hpp"
#include "eigen3/Eigen/Dense"
#include <gsl/gsl_rng.h>

#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "pgm/NodeMetadata.h"
#include "pgm/RandomVariableNode.h"
#include "pgm/cpd/CPD.h"
#include "pgm/cpd/CategoricalCPD.h"
#include "pipeline/estimation/SumProductEstimator.h"
#include "sampling/AncestralSampler.h"
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
    typedef shared_ptr<CategoricalCPD> CPDPtr;
    typedef shared_ptr<DynamicBayesNet> DBNPtr;
    typedef vector<NodeMetadataPtr> ParentsMetadata;

    // Node names
    inline static const string Q = "TrainingCondition";
    inline static const string STATE = "State";
    inline static const string SG = "Green";
    inline static const string SY = "Yellow";
    inline static const string PBAE = "PBAE";

    struct NodeMetadataCollection {
        NodeMetadataPtr q_metadata;
        NodeMetadataPtr state_metadata;
        NodeMetadataPtr green_metadata;
        NodeMetadataPtr yellow_metadata;
        NodeMetadataPtr pbae_metadata;
    };

    struct CPDCollection {
        CPDPtr q_prior_cpd;
        CPDPtr state_prior_cpd;
        CPDPtr pbae_prior_cpd;
        CPDPtr state_given_q_pbae_state_cpd;
        CPDPtr pbae_given_pbae_cpd;
        CPDPtr green_given_state_cpd;
        CPDPtr yellow_given_state_cpd;
    };

    ModelConfig() {}

    DBNPtr create_pre_trained_model() {
        NodeMetadataCollection node_metadatas = this->create_node_metadatas();
        CPDCollection cpds = this->create_pre_trained_cpds(node_metadatas);
        return this->create_model(node_metadatas, cpds);
    }

    DBNPtr create_deterministic_model() {
        NodeMetadataCollection node_metadatas = this->create_node_metadatas();
        CPDCollection cpds = this->create_deterministic_cpds(node_metadatas);
        return this->create_model(node_metadatas, cpds);
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

        return metadatas;
    }

    void create_connections(NodeMetadataCollection node_metadatas) {
        node_metadatas.state_metadata->add_parent_link(
            node_metadatas.q_metadata, true);
        node_metadatas.state_metadata->add_parent_link(
            node_metadatas.state_metadata, true);
        node_metadatas.state_metadata->add_parent_link(
            node_metadatas.pbae_metadata, true);
        node_metadatas.pbae_metadata->add_parent_link(
            node_metadatas.pbae_metadata, true);
        node_metadatas.green_metadata->add_parent_link(
            node_metadatas.state_metadata, false);
        node_metadatas.yellow_metadata->add_parent_link(
            node_metadatas.state_metadata, false);
    }

    CPDCollection
    create_deterministic_cpds(NodeMetadataCollection node_metadatas) {
        CPDCollection cpds;

        MatrixXd q_prior_cpd_table(1, 3);
        q_prior_cpd_table << 1, 0, 0;
        cpds.q_prior_cpd =
            make_shared<CategoricalCPD>(CategoricalCPD({}, q_prior_cpd_table));

        MatrixXd state_prior_cpd_table(1, 3);
        state_prior_cpd_table << 1, 0, 0;
        cpds.state_prior_cpd = make_shared<CategoricalCPD>(
            CategoricalCPD({}, state_prior_cpd_table));

        MatrixXd pbae_prior_cpd_table(1, 2);
        pbae_prior_cpd_table << 0, 1;
        cpds.pbae_prior_cpd = make_shared<CategoricalCPD>(
            CategoricalCPD({}, pbae_prior_cpd_table));

        // Next state is given by Q + PBAE + Previous State MOD 3
        MatrixXd state_given_q_pbae_state_cpd_table(18, 3);
        state_given_q_pbae_state_cpd_table << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
            0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
        ParentsMetadata state_parents_metadata = {
            node_metadatas.q_metadata,
            node_metadatas.pbae_metadata,
            node_metadatas.state_metadata};
        cpds.state_given_q_pbae_state_cpd = make_shared<CategoricalCPD>(
            state_parents_metadata, state_given_q_pbae_state_cpd_table);

        // Next PBAE is given by previous PBAE + 1 MOD 2
        MatrixXd pbae_given_pbae_cpd_table(2, 2);
        pbae_given_pbae_cpd_table << 0, 1, 1, 0;
        ParentsMetadata pbae_parents_metadata = {node_metadatas.pbae_metadata};
        cpds.pbae_given_pbae_cpd = make_shared<CategoricalCPD>(
            pbae_parents_metadata, pbae_given_pbae_cpd_table);

        // Green is given by State + 1 MOD 2
        MatrixXd green_given_state_cpd_table(3, 2);
        green_given_state_cpd_table << 0, 1, 1, 0, 0, 1;
        ParentsMetadata green_parents_metadata = {
            node_metadatas.state_metadata};
        cpds.green_given_state_cpd = make_shared<CategoricalCPD>(
            green_parents_metadata, green_given_state_cpd_table);

        // Yellow is given by State + 2 MOD 2
        MatrixXd yellow_given_state_cpd_table(3, 2);
        yellow_given_state_cpd_table << 1, 0, 0, 1, 1, 0;
        ParentsMetadata yellow_parents_metadata = {
            node_metadatas.state_metadata};
        cpds.yellow_given_state_cpd = make_shared<CategoricalCPD>(
            yellow_parents_metadata, yellow_given_state_cpd_table);

        return cpds;
    }

    CPDCollection
    create_pre_trained_cpds(NodeMetadataCollection node_metadatas) {
        CPDCollection cpds;

        MatrixXd q_prior_cpd_table(1, 3);
        q_prior_cpd_table << 0.5, 0.3, 0.2;
        cpds.q_prior_cpd =
            make_shared<CategoricalCPD>(CategoricalCPD({}, q_prior_cpd_table));

        MatrixXd state_prior_cpd_table(1, 3);
        state_prior_cpd_table << 0.3, 0.5, 0.2;
        cpds.state_prior_cpd = make_shared<CategoricalCPD>(
            CategoricalCPD({}, state_prior_cpd_table));

        MatrixXd pbae_prior_cpd_table(1, 2);
        pbae_prior_cpd_table << 0.3, 0.7;
        cpds.pbae_prior_cpd = make_shared<CategoricalCPD>(
            CategoricalCPD({}, pbae_prior_cpd_table));

        // Next state is given by Q + PBAE + Previous State MOD 3
        MatrixXd state_given_q_pbae_state_cpd_table(18, 3);
        state_given_q_pbae_state_cpd_table << 0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.3,
            0.2, 0.5, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3, 0.5,
            0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2,
            0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.5,
            0.3, 0.2, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5;
        ParentsMetadata state_parents_metadata = {
            node_metadatas.q_metadata,
            node_metadatas.pbae_metadata,
            node_metadatas.state_metadata};
        cpds.state_given_q_pbae_state_cpd = make_shared<CategoricalCPD>(
            state_parents_metadata, state_given_q_pbae_state_cpd_table);

        // Next PBAE is given by previous PBAE + 1 MOD 2
        MatrixXd pbae_given_pbae_cpd_table(2, 2);
        pbae_given_pbae_cpd_table << 0.3, 0.7, 0.7, 0.3;
        ParentsMetadata pbae_parents_metadata = {node_metadatas.pbae_metadata};
        cpds.pbae_given_pbae_cpd = make_shared<CategoricalCPD>(
            pbae_parents_metadata, pbae_given_pbae_cpd_table);

        // P(Green == State + 1 MOD 2) = 0.7
        MatrixXd green_given_state_cpd_table(3, 2);
        green_given_state_cpd_table << 0.3, 0.7, 0.7, 0.3, 0.3, 0.7;
        ParentsMetadata green_parents_metadata = {
            node_metadatas.state_metadata};
        cpds.green_given_state_cpd = make_shared<CategoricalCPD>(
            green_parents_metadata, green_given_state_cpd_table);

        // P(Yellow == State + 2 MOD 2) = 0.7
        MatrixXd yellow_given_state_cpd_table(3, 2);
        yellow_given_state_cpd_table << 0.7, 0.3, 0.3, 0.7, 0.7, 0.3;
        ParentsMetadata yellow_parents_metadata = {
            node_metadatas.state_metadata};
        cpds.yellow_given_state_cpd = make_shared<CategoricalCPD>(
            yellow_parents_metadata, yellow_given_state_cpd_table);

        return cpds;
    }

    DBNPtr create_model(NodeMetadataCollection node_metadatas,
                        CPDCollection cpds) {
        this->create_connections(node_metadatas);

        // RV Nodes
        RandomVariableNode q(node_metadatas.q_metadata);
        q.add_cpd_template(cpds.q_prior_cpd);
        RandomVariableNode state(node_metadatas.state_metadata);
        state.add_cpd_template(cpds.state_prior_cpd);
        state.add_cpd_template(cpds.state_given_q_pbae_state_cpd);
        RandomVariableNode pbae(node_metadatas.pbae_metadata);
        pbae.add_cpd_template(cpds.pbae_prior_cpd);
        pbae.add_cpd_template(cpds.pbae_given_pbae_cpd);
        RandomVariableNode green(node_metadatas.green_metadata);
        green.add_cpd_template(cpds.green_given_state_cpd);
        RandomVariableNode yellow(node_metadatas.yellow_metadata);
        yellow.add_cpd_template(cpds.yellow_given_state_cpd);

        DBNPtr model;
        model = make_shared<DynamicBayesNet>();
        model->add_node_template(q);
        model->add_node_template(state);
        model->add_node_template(pbae);
        model->add_node_template(green);
        model->add_node_template(yellow);

        return model;
    }
};

BOOST_AUTO_TEST_SUITE(ancestral_sampling)

BOOST_FIXTURE_TEST_CASE(complete, ModelConfig) {
    DBNPtr model = create_deterministic_model();

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
    DBNPtr model = create_deterministic_model();

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
    DBNPtr model = create_pre_trained_model();

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
    DBNPtr model = create_pre_trained_model();

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

BOOST_AUTO_TEST_SUITE(estimation)

BOOST_FIXTURE_TEST_CASE(sum_product, ModelConfig) {
    DBNPtr deterministic_model = create_deterministic_model();
    deterministic_model->unroll(4, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // One sample from the deterministic model will be used for estimation.
    AncestralSampler sampler(deterministic_model);
    sampler.sample(gen, 1);

    DBNPtr pre_trained_model = create_pre_trained_model();
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
