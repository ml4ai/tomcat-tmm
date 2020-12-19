#define BOOST_TEST_MODULE TomcatModelTest

#include <memory>
#include <string>
#include <vector>

#include "boost/test/included/unit_test.hpp"
#include "eigen3/Eigen/Dense"
#include <gsl/gsl_rng.h>

#include "pgm/DynamicBayesNet.h"
#include "pgm/NodeMetadata.h"
#include "pgm/RandomVariableNode.h"
#include "pgm/cpd/CPD.h"
#include "pgm/cpd/CategoricalCPD.h"
#include "sampling/AncestralSampler.h"
#include "utils/Definitions.h"

using namespace tomcat::model;
using namespace std;
using namespace Eigen;

struct DeterministicModelConfig {
    /**
     * Small version of the ToMCAT model with deterministic CPDs for testing.
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

    DBNPtr model;

    DeterministicModelConfig() { this->create_model(); }

    void create_model() {
        // Metadata
        NodeMetadataPtr q_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                Q, false, false, true, 1, 1, 3));

        NodeMetadataPtr state_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                STATE, true, false, true, 0, 1, 3));

        NodeMetadataPtr green_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                SG, true, false, true, 1, 1, 2));

        NodeMetadataPtr yellow_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                SY, true, false, true, 1, 1, 2));

        NodeMetadataPtr pbae_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                PBAE, true, false, true, 0, 1, 2));

        // Connections
        state_metadata->add_parent_link(q_metadata, true);
        state_metadata->add_parent_link(state_metadata, true);
        state_metadata->add_parent_link(pbae_metadata, true);
        pbae_metadata->add_parent_link(pbae_metadata, true);
        green_metadata->add_parent_link(state_metadata, false);
        yellow_metadata->add_parent_link(state_metadata, false);

        // CPDs
        MatrixXd q_prior_cpd_table(1, 3);
        q_prior_cpd_table << 1, 0, 0;
        CPDPtr q_prior_cpd =
            make_shared<CategoricalCPD>(CategoricalCPD({}, q_prior_cpd_table));

        MatrixXd state_prior_cpd_table(1, 3);
        state_prior_cpd_table << 1, 0, 0;
        CPDPtr state_prior_cpd = make_shared<CategoricalCPD>(
            CategoricalCPD({}, state_prior_cpd_table));

        MatrixXd pbae_prior_cpd_table(1, 2);
        pbae_prior_cpd_table << 0, 1;
        CPDPtr pbae_prior_cpd = make_shared<CategoricalCPD>(
            CategoricalCPD({}, pbae_prior_cpd_table));

        // Next state is given by Q + PBAE + Previous State MOD 3
        MatrixXd state_given_q_pbae_state_cpd_table(18, 3);
        state_given_q_pbae_state_cpd_table << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
            0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
        ParentsMetadata state_parents_metadata = {
            q_metadata, pbae_metadata, state_metadata};
        CPDPtr state_given_q_pbae_state_cpd = make_shared<CategoricalCPD>(
            state_parents_metadata, state_given_q_pbae_state_cpd_table);

        // Next PBAE is given by previous PBAE + 1 MOD 2
        MatrixXd pbae_given_pbae_cpd_table(2, 2);
        pbae_given_pbae_cpd_table << 0, 1, 1, 0;
        ParentsMetadata pbae_parents_metadata = {pbae_metadata};
        CPDPtr pbae_given_pbae_cpd = make_shared<CategoricalCPD>(
            pbae_parents_metadata, pbae_given_pbae_cpd_table);

        // Green is given by State + 1 MOD 2
        MatrixXd green_given_state_cpd_table(3, 2);
        green_given_state_cpd_table << 0, 1, 1, 0, 0, 1;
        ParentsMetadata green_parents_metadata = {state_metadata};
        CPDPtr green_given_state_cpd = make_shared<CategoricalCPD>(
            green_parents_metadata, green_given_state_cpd_table);

        // Yellow is given by State + 2 MOD 2
        MatrixXd yellow_given_state_cpd_table(3, 2);
        yellow_given_state_cpd_table << 1, 0, 0, 1, 1, 0;
        ParentsMetadata yellow_parents_metadata = {state_metadata};
        CPDPtr yellow_given_state_cpd = make_shared<CategoricalCPD>(
            yellow_parents_metadata, yellow_given_state_cpd_table);

        // RV Nodes
        RandomVariableNode q(q_metadata);
        q.add_cpd_template(q_prior_cpd);
        RandomVariableNode state(state_metadata);
        state.add_cpd_template(state_prior_cpd);
        state.add_cpd_template(state_given_q_pbae_state_cpd);
        RandomVariableNode pbae(pbae_metadata);
        pbae.add_cpd_template(pbae_prior_cpd);
        pbae.add_cpd_template(pbae_given_pbae_cpd);
        RandomVariableNode green(green_metadata);
        green.add_cpd_template(green_given_state_cpd);
        RandomVariableNode yellow(yellow_metadata);
        yellow.add_cpd_template(yellow_given_state_cpd);

        this->model = make_shared<DynamicBayesNet>();
        this->model->add_node_template(q);
        this->model->add_node_template(state);
        this->model->add_node_template(pbae);
        this->model->add_node_template(green);
        this->model->add_node_template(yellow);
    }
};

struct PreTrainedModelConfig {
    /**
     * Small version of the ToMCAT model with pre-defined CPDs for testing.
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

    DBNPtr model;

    PreTrainedModelConfig() { this->create_model(); }

    void create_model() {
        // Metadata
        NodeMetadataPtr q_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                Q, false, false, true, 1, 1, 3));

        NodeMetadataPtr state_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                STATE, true, false, true, 0, 1, 3));

        NodeMetadataPtr green_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                SG, true, false, true, 1, 1, 2));

        NodeMetadataPtr yellow_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                SY, true, false, true, 1, 1, 2));

        NodeMetadataPtr pbae_metadata = make_shared<NodeMetadata>(
            NodeMetadata::create_multiple_time_link_metadata(
                PBAE, true, false, true, 0, 1, 2));

        // Connections
        state_metadata->add_parent_link(q_metadata, true);
        state_metadata->add_parent_link(state_metadata, true);
        state_metadata->add_parent_link(pbae_metadata, true);
        pbae_metadata->add_parent_link(pbae_metadata, true);
        green_metadata->add_parent_link(state_metadata, false);
        yellow_metadata->add_parent_link(state_metadata, false);

        // CPDs
        MatrixXd q_prior_cpd_table(1, 3);
        q_prior_cpd_table << 0.5, 0.3, 0.2;
        CPDPtr q_prior_cpd =
            make_shared<CategoricalCPD>(CategoricalCPD({}, q_prior_cpd_table));

        MatrixXd state_prior_cpd_table(1, 3);
        state_prior_cpd_table << 0.3, 0.5, 0.2;
        CPDPtr state_prior_cpd = make_shared<CategoricalCPD>(
            CategoricalCPD({}, state_prior_cpd_table));

        MatrixXd pbae_prior_cpd_table(1, 2);
        pbae_prior_cpd_table << 0.3, 0.7;
        CPDPtr pbae_prior_cpd = make_shared<CategoricalCPD>(
            CategoricalCPD({}, pbae_prior_cpd_table));

        // Next state is given by Q + PBAE + Previous State MOD 3
        MatrixXd state_given_q_pbae_state_cpd_table(18, 3);
        state_given_q_pbae_state_cpd_table << 0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.3,
            0.2, 0.5, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3, 0.5,
            0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2,
            0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.5,
            0.3, 0.2, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5;
        ParentsMetadata state_parents_metadata = {
            q_metadata, pbae_metadata, state_metadata};
        CPDPtr state_given_q_pbae_state_cpd = make_shared<CategoricalCPD>(
            state_parents_metadata, state_given_q_pbae_state_cpd_table);

        // Next PBAE is given by previous PBAE + 1 MOD 2
        MatrixXd pbae_given_pbae_cpd_table(2, 2);
        pbae_given_pbae_cpd_table << 0.3, 0.7, 0.7, 0.3;
        ParentsMetadata pbae_parents_metadata = {pbae_metadata};
        CPDPtr pbae_given_pbae_cpd = make_shared<CategoricalCPD>(
            pbae_parents_metadata, pbae_given_pbae_cpd_table);

        // P(Green == State + 1 MOD 2) = 0.7
        MatrixXd green_given_state_cpd_table(3, 2);
        green_given_state_cpd_table << 0.3, 0.7, 0.7, 0.3, 0.3, 0.7;
        ParentsMetadata green_parents_metadata = {state_metadata};
        CPDPtr green_given_state_cpd = make_shared<CategoricalCPD>(
            green_parents_metadata, green_given_state_cpd_table);

        // P(Yellow == State + 2 MOD 2) = 0.7
        MatrixXd yellow_given_state_cpd_table(3, 2);
        yellow_given_state_cpd_table << 0.7, 0.3, 0.3, 0.7, 0.7, 0.3;
        ParentsMetadata yellow_parents_metadata = {state_metadata};
        CPDPtr yellow_given_state_cpd = make_shared<CategoricalCPD>(
            yellow_parents_metadata, yellow_given_state_cpd_table);

        // RV Nodes
        RandomVariableNode q(q_metadata);
        q.add_cpd_template(q_prior_cpd);
        RandomVariableNode state(state_metadata);
        state.add_cpd_template(state_prior_cpd);
        state.add_cpd_template(state_given_q_pbae_state_cpd);
        RandomVariableNode pbae(pbae_metadata);
        pbae.add_cpd_template(pbae_prior_cpd);
        pbae.add_cpd_template(pbae_given_pbae_cpd);
        RandomVariableNode green(green_metadata);
        green.add_cpd_template(green_given_state_cpd);
        RandomVariableNode yellow(yellow_metadata);
        yellow.add_cpd_template(yellow_given_state_cpd);

        this->model = make_shared<DynamicBayesNet>();
        this->model->add_node_template(q);
        this->model->add_node_template(state);
        this->model->add_node_template(pbae);
        this->model->add_node_template(green);
        this->model->add_node_template(yellow);
    }
};

BOOST_AUTO_TEST_SUITE(ancestral_sampling)

BOOST_FIXTURE_TEST_CASE(complete, DeterministicModelConfig) {
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

BOOST_FIXTURE_TEST_CASE(truncated, DeterministicModelConfig) {
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

BOOST_FIXTURE_TEST_CASE(heterogeneous, PreTrainedModelConfig) {
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

BOOST_FIXTURE_TEST_CASE(homogeneous, PreTrainedModelConfig) {
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
