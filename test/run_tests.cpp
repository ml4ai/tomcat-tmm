#define BOOST_TEST_MODULE TomcatModelTest

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "boost/test/included/unit_test.hpp"
#include "eigen3/Eigen/Dense"
#include <boost/filesystem.hpp>
#include <gsl/gsl_rng.h>

#include "distribution/Distribution.h"
#include "distribution/Gamma.h"
#include "distribution/Poisson.h"
#include "mock_models.h"
#include "pgm/EvidenceSet.h"
#include "pgm/inference/SegmentExpansionFactorNode.h"
#include "pgm/inference/SegmentMarginalizationFactorNode.h"
#include "pgm/inference/SegmentTransitionFactorNode.h"
#include "pgm/inference/VariableNode.h"
#include "pipeline/estimation/CompoundSamplerEstimator.h"
#include "pipeline/estimation/SamplerEstimator.h"
#include "pipeline/estimation/SumProductEstimator.h"
#include "pipeline/training/DBNSamplingTrainer.h"
#include "sampling/AncestralSampler.h"
#include "sampling/GibbsSampler.h"
#include "utils/Definitions.h"

using namespace tomcat::model;
using namespace std;
using namespace Eigen;
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;
namespace fs = boost::filesystem;

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

string get_matrix_check_msg(const MatrixXd& estimated,
                            const MatrixXd& expected) {
    stringstream msg;
    msg << "Estimated: [" << estimated << "]; Expected: [" << expected << "]";

    return msg.str();
}

pair<bool, string> check_matrix_eq(const MatrixXd& estimated,
                                   const MatrixXd& expected,
                                   double tolerance = 0.00001) {

    string msg = get_matrix_check_msg(estimated, expected);
    bool equal = is_equal(estimated, expected, tolerance);

    return make_pair(equal, msg);
}

bool check_tensor_eq(Tensor3& estimated,
                     Tensor3& expected,
                     double tolerance = 0.00001) {

    if (estimated.get_shape() != expected.get_shape()) {
        return false;
    }

    for (int i = 0; i < estimated.get_shape().at(0); i++) {
        for (int j = 0; j < estimated.get_shape().at(1); j++) {
            for (int k = 0; k < estimated.get_shape().at(2); k++) {
                if (abs(estimated(i, j, k) - expected(i, j, k)) > tolerance) {
                    return false;
                }
            }
        }
    }

    return true;
}

BOOST_AUTO_TEST_SUITE(distribution)

BOOST_AUTO_TEST_CASE(poisson, *utf::tolerance(0.00001)) {

    Poisson poisson(4);
    double pdf = poisson.get_pdf(Eigen::VectorXd::Constant(1, 3));
    BOOST_TEST(pdf == 0.195366);
}

BOOST_AUTO_TEST_CASE(gamma, *utf::tolerance(0.00001)) {

    Gamma gamma(3, 2);
    double pdf = gamma.get_pdf(Eigen::VectorXd::Constant(1, 4));
    BOOST_TEST(pdf == 0.135335);
}

BOOST_AUTO_TEST_SUITE_END()

// Data generation

BOOST_AUTO_TEST_SUITE(data_generation)

BOOST_FIXTURE_TEST_CASE(complete, HMM) {
    /**
     * This test case checks if data can be generated correctly, following
     * the distributions defined in the model. A deterministic model is used
     * so that the values generated can be known in advance.
     */

    DBNPtr model = create_model(true, false);

    int time_steps = 4;
    model->unroll(time_steps, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.sample(gen, 1);

    MatrixXd tcs = sampler.get_samples(TC)(0, 0);
    MatrixXd expected_tcs(1, time_steps);
    expected_tcs << NO_OBS, 0, 0, 0;
    auto check = check_matrix_eq(tcs, expected_tcs);
    BOOST_TEST(check.first, check.second);

    MatrixXd pbaes = sampler.get_samples(PBAE)(0, 0);
    MatrixXd expected_pbaes(1, time_steps);
    expected_pbaes << 1, 0, 1, 0;
    check = check_matrix_eq(pbaes, expected_pbaes);
    BOOST_TEST(check.first, check.second);

    MatrixXd states = sampler.get_samples(STATE)(0, 0);
    MatrixXd expected_states(1, time_steps);
    expected_states << 0, 1, 1, 2;
    check = check_matrix_eq(states, expected_states);
    BOOST_TEST(check.first, check.second);

    MatrixXd greens = sampler.get_samples(GREEN)(0, 0);
    MatrixXd expected_greens(1, time_steps);
    expected_greens << NO_OBS, 0, 0, 1;
    check = check_matrix_eq(greens, expected_greens);
    BOOST_TEST(check.first, check.second);

    MatrixXd yellows = sampler.get_samples(YELLOW)(0, 0);
    MatrixXd expected_yellows(1, time_steps);
    expected_yellows << NO_OBS, 1, 1, 0;
    check = check_matrix_eq(yellows, expected_yellows);
    BOOST_TEST(check.first, check.second);
}

BOOST_FIXTURE_TEST_CASE(truncated, HMM) {
    /**
     * This test case checks if data can be generated correctly up to a time
     * t smaller than the final time T that the DBN was unrolled into. A
     * deterministic model is used.
     */

    DBNPtr model = create_model(true, false);

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    int time_steps = 4;
    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.set_max_time_step_to_sample(time_steps - 1);
    sampler.sample(gen, 1);

    MatrixXd tcs = sampler.get_samples(TC)(0, 0);
    MatrixXd expected_tcs(1, time_steps);
    expected_tcs << NO_OBS, 0, 0, 0;
    auto check = check_matrix_eq(tcs, expected_tcs);
    BOOST_TEST(check.first, check.second);

    MatrixXd pbaes = sampler.get_samples(PBAE)(0, 0);
    MatrixXd expected_pbaes(1, time_steps);
    expected_pbaes << 1, 0, 1, 0;
    check = check_matrix_eq(pbaes, expected_pbaes);
    BOOST_TEST(check.first, check.second);

    MatrixXd states = sampler.get_samples(STATE)(0, 0);
    MatrixXd expected_states(1, time_steps);
    expected_states << 0, 1, 1, 2;
    check = check_matrix_eq(states, expected_states);
    BOOST_TEST(check.first, check.second);

    MatrixXd greens = sampler.get_samples(GREEN)(0, 0);
    MatrixXd expected_greens(1, time_steps);
    expected_greens << NO_OBS, 0, 0, 1;
    check = check_matrix_eq(greens, expected_greens);
    BOOST_TEST(check.first, check.second);

    MatrixXd yellows = sampler.get_samples(YELLOW)(0, 0);
    MatrixXd expected_yellows(1, time_steps);
    expected_yellows << NO_OBS, 1, 1, 0;
    check = check_matrix_eq(yellows, expected_yellows);
    BOOST_TEST(check.first, check.second);
}

BOOST_FIXTURE_TEST_CASE(heterogeneous, HMM) {
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

BOOST_FIXTURE_TEST_CASE(homogeneous, HMM) {
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

BOOST_FIXTURE_TEST_CASE(semi_markov, HSMM) {
    /**
     * This test case checks if data can be generated correctly, following
     * the distributions defined in the semi-Markov model. A deterministic model
     * is used so that the values generated can be known in advance.
     */

    DBNPtr model = create_model(true, false);

    int time_steps = 15;
    model->unroll(time_steps, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.sample(gen, 1);

    MatrixXd tcs = sampler.get_samples(TC)(0, 0);
    MatrixXd expected_tcs = MatrixXd::Zero(1, time_steps);
    auto check = check_matrix_eq(tcs, expected_tcs);
    BOOST_TEST(check.first, check.second);

    MatrixXd pbaes = sampler.get_samples(PBAE)(0, 0);
    MatrixXd expected_pbaes(1, time_steps);
    expected_pbaes << 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;
    check = check_matrix_eq(pbaes, expected_pbaes);
    BOOST_TEST(check.first, check.second);

    MatrixXd states = sampler.get_samples(STATE)(0, 0);
    MatrixXd expected_states(1, time_steps);
    expected_states << 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0;
    check = check_matrix_eq(states, expected_states);
    BOOST_TEST(check.first, check.second);

    MatrixXd greens = sampler.get_samples(GREEN)(0, 0);
    MatrixXd expected_greens(1, time_steps);
    expected_greens << NO_OBS, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1;
    check = check_matrix_eq(greens, expected_greens);
    BOOST_TEST(check.first, check.second);

    MatrixXd yellows = sampler.get_samples(YELLOW)(0, 0);
    MatrixXd expected_yellows(1, time_steps);
    expected_yellows << NO_OBS, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0;
    check = check_matrix_eq(yellows, expected_yellows);
    BOOST_TEST(check.first, check.second);

    MatrixXd timer = sampler.get_samples(TIMER)(0, 0);
    MatrixXd expected_timer(1, time_steps);
    expected_timer << 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 2, 1, 0, 0;
    check = check_matrix_eq(timer, expected_timer);
    BOOST_TEST(check.first, check.second);
}

BOOST_AUTO_TEST_SUITE_END()

// Training

BOOST_AUTO_TEST_SUITE(model_training)

BOOST_FIXTURE_TEST_CASE(gibbs_sampling_hmm, HMM) {
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
        make_shared<GibbsSampler>(model, 200, 1);
    gibbs_sampler->set_show_progress(false);
    DBNSamplingTrainer trainer(gen_training, gibbs_sampler, 200);

    double tolerance = 0.05;
    CPDTableCollection tables = this->create_cpd_tables(false);

    // Check parameter learning when tc is not provided.
    EvidenceSet data;
    data.add_data(PBAE, sampler.get_samples(PBAE));
    data.add_data(STATE, sampler.get_samples(STATE));
    data.add_data(GREEN, sampler.get_samples(GREEN));
    data.add_data(YELLOW, sampler.get_samples(YELLOW));

    // Fix some parameters to avoid permutation of TC and
    // PBAE.
    const shared_ptr<RandomVariableNode>& theta_tc =
        dynamic_pointer_cast<RandomVariableNode>(
            model->get_nodes_by_label(THETA_TC)[0]);
    theta_tc->set_assignment(tables.tc_prior);
    theta_tc->freeze();

    for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE_TC_PBAE;
         i = i + PBAE_CARDINALITY) {
        stringstream label;
        label << THETA_STATE_GIVEN_STATE_TC_PBAE << "_" << i;
        const shared_ptr<RandomVariableNode>& theta_state =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        theta_state->set_assignment(tables.state_given_state_tc_pbae.row(i));
        theta_state->freeze();
    }

    trainer.prepare();
    trainer.fit(data);
    model->get_nodes_by_label(THETA_TC)[0]->get_assignment();
    MatrixXd estimated_theta_tc =
        model->get_nodes_by_label(THETA_TC)[0]->get_assignment();
    auto check =
        check_matrix_eq(estimated_theta_tc, tables.tc_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    MatrixXd estimated_pi_pbae =
        model->get_nodes_by_label(PI_PBAE)[0]->get_assignment();
    check = check_matrix_eq(estimated_pi_pbae, tables.pbae_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    MatrixXd estimated_theta_state =
        model->get_nodes_by_label(THETA_STATE)[0]->get_assignment();
    check =
        check_matrix_eq(estimated_theta_state, tables.state_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
        stringstream label;
        label << PI_PBAE_GIVEN_PBAE << '_' << i;
        MatrixXd estimated_pi_pbae_given_pbae =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_pi_pbae_given_pbae,
                                tables.pbae_given_pbae.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE_TC_PBAE; i++) {
        stringstream label;
        label << THETA_STATE_GIVEN_STATE_TC_PBAE << '_' << i;
        MatrixXd estimated_theta_state_given_tc_pbae_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_theta_state_given_tc_pbae_state,
                                tables.state_given_state_tc_pbae.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
        stringstream label;
        label << PI_GREEN_GIVEN_STATE << '_' << i;

        MatrixXd estimated_pi_green_given_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_pi_green_given_state,
                                tables.green_given_state.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
        stringstream label;
        label << PI_YELLOW_GIVEN_STATE << '_' << i;

        MatrixXd estimated_pi_yellow_given_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_pi_yellow_given_state,
                                tables.yellow_given_state.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }
}

BOOST_FIXTURE_TEST_CASE(gibbs_sampling_hsmm_durations, HSMM) {
    /**
     * This test case checks if the model can learn the duration parameters of a
     * non-deterministic semi-Markov model, given data generated from such a
     * model. Observations for node TC are not provided to the sampler to
     * capture the ability of the procedure to learn the parameters given that
     * some nodes are hidden.
     */

    DBNPtr oracle = create_model(false, false);
    oracle->unroll(100, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Generate a bunch of samples to train a model from the scratch.
    AncestralSampler sampler(oracle);
    sampler.sample(gen, 500);

    DBNPtr model = create_model(false, true);
    model->unroll(100, true);

    shared_ptr<gsl_rng> gen_training(gsl_rng_alloc(gsl_rng_mt19937));
    shared_ptr<GibbsSampler> gibbs_sampler =
        make_shared<GibbsSampler>(model, 50, 1);
    gibbs_sampler->set_show_progress(false);
    DBNSamplingTrainer trainer(gen_training, gibbs_sampler, 100);

    CPDTableCollection tables = this->create_cpd_tables(false);

    // Given the distributions for state transition and emission of green
    // and yellow, observations of PBAE, GREEN and YELLOW, can we determine the
    // distribution over training conditions and durations?
    EvidenceSet data;
    data.add_data(PBAE, sampler.get_samples(PBAE));
    data.add_data(GREEN, sampler.get_samples(GREEN));
    data.add_data(YELLOW, sampler.get_samples(YELLOW));

    const shared_ptr<RandomVariableNode>& theta_tc =
        dynamic_pointer_cast<RandomVariableNode>(
            model->get_nodes_by_label(THETA_TC)[0]);
    theta_tc->set_assignment(tables.tc_prior);
    theta_tc->freeze();

    // To avoid permutation, we need to provide some answers.
    for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_STATE;
         i += STATE_CARDINALITY) {
        stringstream label;
        label << LAMBDA_TIMER_GIVEN_TC_STATE << '_' << i;
        const shared_ptr<RandomVariableNode>& lambda_timer =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        lambda_timer->set_assignment(tables.timer_given_tc_state.row(i));
        lambda_timer->freeze();
    }

    for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE_PBAE; i++) {
        stringstream label;
        label << THETA_STATE_GIVEN_STATE_PBAE << '_' << i;
        const shared_ptr<RandomVariableNode>& theta_state =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        theta_state->set_assignment(tables.state_given_state_pbae.row(i));
        theta_state->freeze();
    }

    for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
        stringstream label;
        label << PI_GREEN_GIVEN_STATE << '_' << i;
        const shared_ptr<RandomVariableNode>& pi_green =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        pi_green->set_assignment(tables.green_given_state.row(i));
        pi_green->freeze();
    }

    for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
        stringstream label;
        label << PI_YELLOW_GIVEN_STATE << '_' << i;
        const shared_ptr<RandomVariableNode>& pi_yellow =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        pi_yellow->set_assignment(tables.yellow_given_state.row(i));
        pi_yellow->freeze();
    }

    trainer.prepare();
    trainer.fit(data);

    double tolerance = 0.05;
    MatrixXd estimated_pi_pbae =
        model->get_nodes_by_label(PI_PBAE)[0]->get_assignment();
    auto check =
        check_matrix_eq(estimated_pi_pbae, tables.pbae_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    MatrixXd estimated_theta_state =
        model->get_nodes_by_label(THETA_STATE)[0]->get_assignment();
    check =
        check_matrix_eq(estimated_theta_state, tables.state_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
        stringstream label;
        label << PI_PBAE_GIVEN_PBAE << '_' << i;
        MatrixXd estimated_pi_pbae_given_pbae =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_pi_pbae_given_pbae,
                                tables.pbae_given_pbae.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    tolerance = 0.8; // higher tolerance to lambda
    for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_STATE; i++) {
        stringstream label;
        label << LAMBDA_TIMER_GIVEN_TC_STATE << '_' << i;
        MatrixXd estimated_lambda_timer_given_tc_pbae_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_lambda_timer_given_tc_pbae_state,
                                tables.timer_given_tc_state.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }
}

BOOST_FIXTURE_TEST_CASE(gibbs_sampling_hsmm_transitions, HSMM) {
    /**
     * This test case checks if the model can learn the transition parameters
     * of a* non-deterministic semi-Markov model, given data generated from such
     * a model. Observations for node TC are not provided to the sampler to
     * capture the ability of the procedure to learn the parameters given that
     * some nodes are hidden.
     */

    DBNPtr oracle = create_model(false, false);
    oracle->unroll(100, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Generate a bunch of samples to train a model from the scratch.
    AncestralSampler sampler(oracle);
    sampler.sample(gen, 500);

    DBNPtr model = create_model(false, true);
    model->unroll(100, true);

    shared_ptr<gsl_rng> gen_training(gsl_rng_alloc(gsl_rng_mt19937));
    shared_ptr<GibbsSampler> gibbs_sampler =
        make_shared<GibbsSampler>(model, 50, 1);
    gibbs_sampler->set_show_progress(false);
    DBNSamplingTrainer trainer(gen_training, gibbs_sampler, 100);

    CPDTableCollection tables = this->create_cpd_tables(false);

    // Given the distributions for state duration and emission of green
    // and yellow, observations of PBAE, GREEN and YELLOW, can we determine the
    // distribution over training conditions and state transition?
    EvidenceSet data;
    data.add_data(PBAE, sampler.get_samples(PBAE));
    data.add_data(GREEN, sampler.get_samples(GREEN));
    data.add_data(YELLOW, sampler.get_samples(YELLOW));

    const shared_ptr<RandomVariableNode>& theta_tc =
        dynamic_pointer_cast<RandomVariableNode>(
            model->get_nodes_by_label(THETA_TC)[0]);
    theta_tc->set_assignment(tables.tc_prior);
    theta_tc->freeze();

    for (int i = 0; i < NUM_LAMBDA_TIMER_GIVEN_TC_STATE; i++) {
        stringstream label;
        label << LAMBDA_TIMER_GIVEN_TC_STATE << '_' << i;
        const shared_ptr<RandomVariableNode>& lambda_timer =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        lambda_timer->set_assignment(tables.timer_given_tc_state.row(i));
        lambda_timer->freeze();
    }

    //        // To avoid permutation, we need to provide some answers.
    //    for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE_PBAE; i++) {
    //        stringstream label;
    //        label << THETA_STATE_GIVEN_STATE_PBAE << '_' << i;
    //        const shared_ptr<RandomVariableNode>& theta_state =
    //            dynamic_pointer_cast<RandomVariableNode>(
    //                model->get_nodes_by_label(label.str())[0]);
    //        theta_state->set_assignment(tables.state_given_state_pbae.row(i));
    //        theta_state->freeze();
    //    }

    for (int i = 0; i < NUM_PI_GREEN_GIVEN_STATE; i++) {
        stringstream label;
        label << PI_GREEN_GIVEN_STATE << '_' << i;
        const shared_ptr<RandomVariableNode>& pi_green =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        pi_green->set_assignment(tables.green_given_state.row(i));
        pi_green->freeze();
    }

    for (int i = 0; i < NUM_PI_YELLOW_GIVEN_STATE; i++) {
        stringstream label;
        label << PI_YELLOW_GIVEN_STATE << '_' << i;
        const shared_ptr<RandomVariableNode>& pi_yellow =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        pi_yellow->set_assignment(tables.yellow_given_state.row(i));
        pi_yellow->freeze();
    }

    trainer.prepare();
    trainer.fit(data);

    double tolerance = 0.06;
    MatrixXd estimated_pi_pbae =
        model->get_nodes_by_label(PI_PBAE)[0]->get_assignment();
    auto check =
        check_matrix_eq(estimated_pi_pbae, tables.pbae_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    MatrixXd estimated_theta_state =
        model->get_nodes_by_label(THETA_STATE)[0]->get_assignment();
    check =
        check_matrix_eq(estimated_theta_state, tables.state_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    for (int i = 0; i < NUM_THETA_STATE_GIVEN_STATE_PBAE; i++) {
        stringstream label;
        label << THETA_STATE_GIVEN_STATE_PBAE << '_' << i;
        MatrixXd estimated_theta_state_given_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_theta_state_given_state,
                                tables.state_given_state_pbae.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    for (int i = 0; i < NUM_PI_PBAE_GIVEN_PBAE; i++) {
        stringstream label;
        label << PI_PBAE_GIVEN_PBAE << '_' << i;
        MatrixXd estimated_pi_pbae_given_pbae =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_pi_pbae_given_pbae,
                                tables.pbae_given_pbae.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }
}

BOOST_AUTO_TEST_SUITE_END()

// Inference

BOOST_AUTO_TEST_SUITE(estimation)

BOOST_FIXTURE_TEST_CASE(sum_product, HMM) {
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
    auto check = check_matrix_eq(green_estimates_h1, expected_green_h1);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_yellow_h1(1, 4);
    expected_yellow_h1 << 0.43212, 0.43411, 0.433133, 0.434007;
    check = check_matrix_eq(yellow_estimates_h1, expected_yellow_h1);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_green_h3(1, 4);
    expected_green_h3 << 0.91875, 0.918454, 0.918545, 0.918483;
    check = check_matrix_eq(green_estimates_h3, expected_green_h3);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_yellow_h3(1, 4);
    expected_yellow_h3 << 0.818174, 0.818918, 0.818499, 0.818842;
    check = check_matrix_eq(yellow_estimates_h3, expected_yellow_h3);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_tc(3, 4);
    expected_tc << -1, 0.500324, 0.506335, 0.504282, -1, 0.294363, 0.286289,
        0.291487, -1, 0.205313, 0.207376, 0.204232;
    check = check_matrix_eq(tc_estimates[0], expected_tc.row(0));
    BOOST_TEST(check.first, check.second);
    check = check_matrix_eq(tc_estimates[1], expected_tc.row(1));
    BOOST_TEST(check.first, check.second);
    check = check_matrix_eq(tc_estimates[2], expected_tc.row(2));
    BOOST_TEST(check.first, check.second);
}

BOOST_FIXTURE_TEST_CASE(gs_ai_hmm, HMM) {
    /**
     * This test case checks if the Gibbs sampling procedure can approximate
     * correctly the marginal probabilities over time of the nodes Green and
     * TC from a non-deterministic model. The approximations should be close
     * enough to the ones obtained by the sum-product algorithm.
     */

    DBNPtr deterministic_model = create_model(true, false);
    deterministic_model->unroll(4, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // One sample from the deterministic model will be used for estimation.
    AncestralSampler sampler(deterministic_model);
    sampler.sample(gen, 1);

    DBNPtr pre_trained_model = create_model(false, false);
    pre_trained_model->unroll(4, true);

    shared_ptr<SamplerEstimator> green_estimator_h1 =
        make_shared<SamplerEstimator>(
            pre_trained_model, 1, GREEN, VectorXd::Constant(1, 1));
    shared_ptr<SamplerEstimator> yellow_estimator_h1 =
        make_shared<SamplerEstimator>(
            pre_trained_model, 1, YELLOW, VectorXd::Constant(1, 1));
    shared_ptr<SamplerEstimator> green_estimator_h3 =
        make_shared<SamplerEstimator>(
            pre_trained_model, 3, GREEN, VectorXd::Constant(1, 1));
    shared_ptr<SamplerEstimator> yellow_estimator_h3 =
        make_shared<SamplerEstimator>(
            pre_trained_model, 3, YELLOW, VectorXd::Constant(1, 1));
    shared_ptr<SamplerEstimator> tc_estimator =
        make_shared<SamplerEstimator>(pre_trained_model, 0, TC);

    shared_ptr<GibbsSampler> gibbs =
        make_shared<GibbsSampler>(pre_trained_model, 0, 1);
    gibbs->set_show_progress(false);
    gibbs->set_trainable(false);
    CompoundSamplerEstimator sampler_estimator(gibbs, gen, 2000);
    sampler_estimator.set_show_progress(false);
    sampler_estimator.add_base_estimator(green_estimator_h1);
    sampler_estimator.add_base_estimator(yellow_estimator_h1);
    sampler_estimator.add_base_estimator(green_estimator_h3);
    sampler_estimator.add_base_estimator(yellow_estimator_h3);
    sampler_estimator.add_base_estimator(tc_estimator);

    // Green node is observed
    EvidenceSet data;
    data.add_data(GREEN, sampler.get_samples(GREEN));

    sampler_estimator.prepare();
    sampler_estimator.estimate(data);

    MatrixXd green_estimates_h1 =
        green_estimator_h1->get_estimates().estimates[0];
    MatrixXd yellow_estimates_h1 =
        yellow_estimator_h1->get_estimates().estimates[0];
    MatrixXd green_estimates_h3 =
        green_estimator_h3->get_estimates().estimates[0];
    MatrixXd yellow_estimates_h3 =
        yellow_estimator_h3->get_estimates().estimates[0];
    vector<MatrixXd> tc_estimates = tc_estimator->get_estimates().estimates;

    double tolerance = 0.03;

    MatrixXd expected_green_h1(1, 4);
    expected_green_h1 << 0.56788, 0.56589, 0.566867, 0.565993;
    auto check =
        check_matrix_eq(green_estimates_h1, expected_green_h1, tolerance);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_yellow_h1(1, 4);
    expected_yellow_h1 << 0.43212, 0.43411, 0.433133, 0.434007;
    check = check_matrix_eq(yellow_estimates_h1, expected_yellow_h1, tolerance);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_green_h3(1, 4);
    expected_green_h3 << 0.91875, 0.918454, 0.918545, 0.918483;
    check = check_matrix_eq(green_estimates_h3, expected_green_h3, tolerance);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_yellow_h3(1, 4);
    expected_yellow_h3 << 0.818174, 0.818918, 0.818499, 0.818842;
    check = check_matrix_eq(yellow_estimates_h3, expected_yellow_h3, tolerance);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_tc(3, 4);
    expected_tc << -1, 0.500324, 0.506335, 0.504282, -1, 0.294363, 0.286289,
        0.291487, -1, 0.205313, 0.207376, 0.204232;
    check = check_matrix_eq(tc_estimates[0], expected_tc.row(0), tolerance);
    BOOST_TEST(check.first, check.second);
    check = check_matrix_eq(tc_estimates[1], expected_tc.row(1), tolerance);
    BOOST_TEST(check.first, check.second);
    check = check_matrix_eq(tc_estimates[2], expected_tc.row(2), tolerance);
    BOOST_TEST(check.first, check.second);
}

BOOST_FIXTURE_TEST_CASE(gs_ai_hsmm, ShortHSMM) {
    /**
     * This test case checks if the Gibbs sampling procedure can approximate
     * correctly the marginal probabilities over time of the node Green
     * in a hidden semi-Markov model.
     */

    DBNPtr deterministic_model = create_model(true, false);
    deterministic_model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // One sample from the deterministic model will be used for estimation.
    AncestralSampler sampler(deterministic_model);
    sampler.sample(gen, 1);

    DBNPtr pre_trained_model = create_model(false, false);
    pre_trained_model->unroll(4, true);

    shared_ptr<SamplerEstimator> state_estimator_h1 =
        make_shared<SamplerEstimator>(pre_trained_model, 1, STATE);
    shared_ptr<SamplerEstimator> green_estimator_h1 =
        make_shared<SamplerEstimator>(
            pre_trained_model, 1, GREEN, VectorXd::Constant(1, 1));

    shared_ptr<GibbsSampler> gibbs =
        make_shared<GibbsSampler>(pre_trained_model, 0, 1);
    gibbs->set_show_progress(false);
    gibbs->set_trainable(false);
    CompoundSamplerEstimator sampler_estimator(gibbs, gen, 10000);
    sampler_estimator.set_show_progress(false);
    sampler_estimator.add_base_estimator(state_estimator_h1);
    sampler_estimator.add_base_estimator(green_estimator_h1);

    // Green node is observed
    EvidenceSet data;
    data.add_data(GREEN, sampler.get_samples(GREEN));

    sampler_estimator.prepare();
    sampler_estimator.estimate(data);

    vector<MatrixXd> state_estimates_h1 =
        state_estimator_h1->get_estimates().estimates;

    MatrixXd green_estimates_h1 =
        green_estimator_h1->get_estimates().estimates[0];

    double tolerance = 0.02;

    MatrixXd expected_state_h1(1, 3);
    expected_state_h1 << 0.6321, 0.3412, 0.1748;
    auto check =
        check_matrix_eq(state_estimates_h1[0], expected_state_h1, tolerance);
    BOOST_TEST(check.first, check.second);

    expected_state_h1 << 0.2943, 0.4949, 0.5802;
    check =
        check_matrix_eq(state_estimates_h1[1], expected_state_h1, tolerance);
    BOOST_TEST(check.first, check.second);

    expected_state_h1 << 0.0736, 0.1639, 0.2450;
    check =
        check_matrix_eq(state_estimates_h1[2], expected_state_h1, tolerance);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_green_h1(1, 3);
    expected_green_h1 << 0.6308, 0.5034, 0.4393;
    check = check_matrix_eq(green_estimates_h1, expected_green_h1, tolerance);
    BOOST_TEST(check.first, check.second);
}

BOOST_AUTO_TEST_CASE(segment_extension_factor) {
    // Testing whether the extension factor node is working properly at
    // producing an output message for two duration dependencies and
    // intermediary segment node in two subsequent time steps.

    int state_cardinality = 3;
    int x_cardinality = 2;
    int y_cardinality = 2;

    // P(duration | State, X, Y)
    DistributionPtrVec duration_distributions = {make_shared<Poisson>(1),
                                                 make_shared<Poisson>(2),
                                                 make_shared<Poisson>(3),
                                                 make_shared<Poisson>(4),
                                                 make_shared<Poisson>(5),
                                                 make_shared<Poisson>(6),
                                                 make_shared<Poisson>(7),
                                                 make_shared<Poisson>(8),
                                                 make_shared<Poisson>(9),
                                                 make_shared<Poisson>(10),
                                                 make_shared<Poisson>(11),
                                                 make_shared<Poisson>(12)};

    CPD::TableOrderingMap duration_ordering_map;
    duration_ordering_map["State"] =
        ParentIndexing(0, state_cardinality, x_cardinality * y_cardinality);
    duration_ordering_map["X"] =
        ParentIndexing(1, x_cardinality, y_cardinality);
    duration_ordering_map["Y"] = ParentIndexing(2, y_cardinality, 1);

    SegmentExpansionFactorNode expansion_factor(
        "f:Expansion", 1, duration_distributions, duration_ordering_map);

    // P(duration | State, X, Y)
    Eigen::MatrixXd transition_probs(
        state_cardinality * x_cardinality * y_cardinality, state_cardinality);
    transition_probs << 0, 0.3, 0.7, 0, 0.4, 0.6, 0, 0.1, 0.9, 0, 0.2, 0.8, 0.4,
        0, 0.6, 0.5, 0, 0.5, 0.4, 0, 0.6, 0.5, 0, 0.5, 0.5, 0.5, 0, 0.6, 0.4, 0,
        1, 0, 0, 0.1, 0.9, 0;

    CPD::TableOrderingMap transition_ordering_map;
    transition_ordering_map["State"] =
        ParentIndexing(0, state_cardinality, x_cardinality);
    transition_ordering_map["X"] = ParentIndexing(1, x_cardinality, 1);
    transition_ordering_map["Y"] = ParentIndexing(2, y_cardinality, 1);

    SegTransFactorNodePtr transition_factor =
        make_shared<SegmentTransitionFactorNode>("f:Transition",
                                                 1,
                                                 transition_probs,
                                                 transition_ordering_map,
                                                 duration_ordering_map);

    VarNodePtr segment = make_shared<VariableNode>("Segment", 0);
    VarNodePtr x = make_shared<VariableNode>("X", 1, x_cardinality);
    VarNodePtr y = make_shared<VariableNode>("Y", 1, y_cardinality);

    int num_data_points = 2;
    Eigen::MatrixXd prob_xy(num_data_points, x_cardinality);
    prob_xy << 0.3, 0.7, 0.8, 0.2;
    Tensor3 msg_from_xy = Tensor3(prob_xy);

    Tensor3 msg_from_seg0 = Tensor3::ones(
        num_data_points, x_cardinality * y_cardinality, state_cardinality);

    // First time step
    int time_step = 1;
    expansion_factor.set_incoming_message_from(segment,
                                               time_step - 1,
                                               time_step,
                                               msg_from_seg0,
                                               MessageNode::Direction::forward);
    expansion_factor.set_incoming_message_from(
        x, time_step, time_step, msg_from_xy, MessageNode::Direction::forward);
    expansion_factor.set_incoming_message_from(
        y, time_step, time_step, msg_from_xy, MessageNode::Direction::forward);

    Tensor3 msg2trans = expansion_factor.get_outward_message_to(
        transition_factor,
        time_step,
        time_step,
        MessageNode::Direction::forward);

    // Suppose the transition table is the identity and the message that
    // comes from it is just a replica of the message from the next state
    Eigen::MatrixXd msg_from_trans_matrix(1,
                                          (time_step + 1) * state_cardinality);
    msg_from_trans_matrix << 0.1, 0.1, 0.8, 0.1, 0.1, 0.8;
    msg_from_trans_matrix =
        msg_from_trans_matrix.replicate(x_cardinality * y_cardinality, 1);
    Tensor3 msg_from_trans(msg_from_trans_matrix);
    msg_from_trans = msg_from_trans.repeat(num_data_points, 0);

    expansion_factor.set_incoming_message_from(
        transition_factor,
        time_step,
        time_step,
        msg_from_trans,
        MessageNode::Direction::backwards);

    Tensor3 msg2x = expansion_factor.get_outward_message_to(
        x, time_step, time_step, MessageNode::Direction::backwards);
    Tensor3 msg2y = expansion_factor.get_outward_message_to(
        y, time_step, time_step, MessageNode::Direction::backwards);

    int num_rows = x_cardinality * y_cardinality;
    vector<Eigen::MatrixXd> expected_msg2trans_matrices(num_data_points);
    expected_msg2trans_matrices[0] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    expected_msg2trans_matrices[0] << 0.033109149705430, 0.000606415229918,
        0.000011106882368, 0.056890850294570, 0.089393584770082,
        0.089988893117632, 0.028420409479689, 0.000520537957100,
        0.000009533985250, 0.181579590520311, 0.209479462042900,
        0.209990466014750, 0.010455284357251, 0.000191495212766,
        0.000003507357166, 0.199544715642749, 0.209808504787234,
        0.209996492642834, 0.008974663055480, 0.000164376687672,
        0.000003010664053, 0.481025336944520, 0.489835623312328,
        0.489996989335947;
    expected_msg2trans_matrices[1] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    expected_msg2trans_matrices[1] << 0.235442842349723, 0.004312286079415,
        0.000078982274615, 0.404557157650277, 0.635687713920585,
        0.639921017725385, 0.021653645317858, 0.000396600348267,
        0.000007263988762, 0.138346354682142, 0.159603399651733,
        0.159992736011238, 0.007965930938858, 0.000145901114489,
        0.000002672272126, 0.152034069061142, 0.159854098885511,
        0.159997327727874, 0.000732625555549, 0.000013418505116,
        0.000000245768494, 0.039267374444451, 0.039986581494884,
        0.039999754231506;
    Tensor3 expected_msg2trans(expected_msg2trans_matrices);

    Eigen::MatrixXd expected_msg2x_matrix =
        Eigen::MatrixXd::Ones(num_data_points, x_cardinality);
    Tensor3 expected_msg2x(expected_msg2x_matrix);

    Eigen::MatrixXd expected_msg2y_matrix =
        Eigen::MatrixXd::Ones(num_data_points, y_cardinality);
    Tensor3 expected_msg2y(expected_msg2y_matrix);

    BOOST_TEST(check_tensor_eq(msg2trans, expected_msg2trans));
    BOOST_TEST(check_tensor_eq(msg2x, expected_msg2x));
    BOOST_TEST(check_tensor_eq(msg2y, expected_msg2y));

    // Second time step
    SegmentExpansionFactorNode next_expansion_factor("f:Expansion",
                                                     time_step,
                                                     duration_distributions,
                                                     duration_ordering_map);
    SegTransFactorNodePtr next_transition_factor =
        make_shared<SegmentTransitionFactorNode>("f:Transition",
                                                 time_step,
                                                 transition_probs,
                                                 transition_ordering_map,
                                                 duration_ordering_map);

    VarNodePtr next_x =
        make_shared<VariableNode>("X", time_step, x_cardinality);
    VarNodePtr next_y =
        make_shared<VariableNode>("Y", time_step, y_cardinality);

    Tensor3 msg_from_prev_seg = msg2trans * msg_from_trans;
    msg_from_prev_seg.normalize_rows();

    time_step = 2;
    next_expansion_factor.set_incoming_message_from(
        segment,
        time_step - 1,
        time_step,
        msg_from_prev_seg,
        MessageNode::Direction::forward);
    next_expansion_factor.set_incoming_message_from(
        next_x,
        time_step,
        time_step,
        msg_from_xy,
        MessageNode::Direction::forward);
    next_expansion_factor.set_incoming_message_from(
        next_y,
        time_step,
        time_step,
        msg_from_xy,
        MessageNode::Direction::forward);

    msg2trans = next_expansion_factor.get_outward_message_to(
        next_transition_factor,
        time_step,
        time_step,
        MessageNode::Direction::forward);

    msg_from_trans_matrix =
        Eigen::MatrixXd(1, (time_step + 1) * state_cardinality);
    msg_from_trans_matrix << 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8;
    msg_from_trans_matrix =
        msg_from_trans_matrix.replicate(x_cardinality * y_cardinality, 1);
    msg_from_trans = Tensor3(msg_from_trans_matrix);
    msg_from_trans = msg_from_trans.repeat(num_data_points, 0);

    next_expansion_factor.set_incoming_message_from(
        next_transition_factor,
        time_step,
        time_step,
        msg_from_trans,
        MessageNode::Direction::backwards);

    msg2x = next_expansion_factor.get_outward_message_to(
        next_x, time_step, time_step, MessageNode::Direction::backwards);
    msg2y = next_expansion_factor.get_outward_message_to(
        next_y, time_step, time_step, MessageNode::Direction::backwards);

    expected_msg2trans_matrices[0] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    expected_msg2trans_matrices[0] << 0.004528932519672, 0.000303616214327,
        0.000079970649607, 0.002092897421413, 0.000060232923624,
        0.000008884409336, 0.002378170058914, 0.008636150862049,
        0.071911144941058, 0.006068710312601, 0.000312451802719,
        0.000076272228275, 0.002457412531305, 0.000051924767251,
        0.000007626841926, 0.012473877156093, 0.020635623430030,
        0.167916100929799, 0.003188639102885, 0.000134064111040,
        0.000030864789923, 0.000993474640015, 0.000019132059174,
        0.000002805838870, 0.016817886257099, 0.020846803829787,
        0.167966329371207, 0.003606302890959, 0.000131506864361,
        0.000028902389709, 0.000881028636781, 0.000016432154544,
        0.000002408516444, 0.044512668472260, 0.048852060981095,
        0.391968689093847;

    expected_msg2trans_matrices[1] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    expected_msg2trans_matrices[1] << 0.032205742362116, 0.002159048635212,
        0.000568680174981, 0.014882826107829, 0.000428323012437,
        0.000063178021943, 0.016911431530055, 0.061412628352351,
        0.511368141803076, 0.004623779285791, 0.000238058516358,
        0.000058112173924, 0.001872314309566, 0.000039561727429,
        0.000005810927182, 0.009503906404643, 0.015722379756213,
        0.127936076898894, 0.002429439316484, 0.000102144084602,
        0.000023516030418, 0.000756933059059, 0.000014576806989,
        0.000002137781996, 0.012813627624457, 0.015883279108409,
        0.127974346187586, 0.000294392072731, 0.000010735254234,
        0.000002359378752, 0.000071920705043, 0.000001341400371,
        0.000000196613587, 0.003633687222225, 0.003987923345396,
        0.031997444007661;
    expected_msg2trans = Tensor3(expected_msg2trans_matrices);

    expected_msg2x_matrix << 0.66, 0.66, 0.66, 0.66;
    expected_msg2x = Tensor3(expected_msg2x_matrix);

    expected_msg2y_matrix << 0.66, 0.66, 0.66, 0.66;
    expected_msg2y = Tensor3(expected_msg2y_matrix);

    BOOST_TEST(check_tensor_eq(msg2trans, expected_msg2trans));
    BOOST_TEST(check_tensor_eq(msg2x, expected_msg2x));
    BOOST_TEST(check_tensor_eq(msg2y, expected_msg2y));
}

BOOST_AUTO_TEST_CASE(segment_transition_factor) {
    // Testing whether the transition factor node is working properly at
    // producing an output message to the final segment and for a transition
    // dependency that is not a dependency of the segment duration distribution.

    int state_cardinality = 3;
    int x_cardinality = 2;
    int y_cardinality = 2;
    int time_step = 1;

    // P(duration | State, X)
    DistributionPtrVec duration_distributions = {make_shared<Poisson>(1),
                                                 make_shared<Poisson>(2),
                                                 make_shared<Poisson>(3),
                                                 make_shared<Poisson>(4),
                                                 make_shared<Poisson>(5),
                                                 make_shared<Poisson>(6)};

    CPD::TableOrderingMap duration_ordering_map;
    duration_ordering_map["State"] =
        ParentIndexing(0, state_cardinality, x_cardinality);
    duration_ordering_map["X"] = ParentIndexing(1, x_cardinality, 1);

    SegExpFactorNodePtr expansion_factor =
        make_shared<SegmentExpansionFactorNode>(
            "f:Expansion", 1, duration_distributions, duration_ordering_map);

    // P(transition | State, X, Y)
    Eigen::MatrixXd transition_probs(
        state_cardinality * x_cardinality * y_cardinality, state_cardinality);
    transition_probs << 0, 0.3, 0.7, 0, 0.4, 0.6, 0, 0.1, 0.9, 0, 0.2, 0.8, 0.4,
        0, 0.6, 0.5, 0, 0.5, 0.4, 0, 0.6, 0.5, 0, 0.5, 0.5, 0.5, 0, 0.6, 0.4, 0,
        1, 0, 0, 0.1, 0.9, 0;

    CPD::TableOrderingMap transition_ordering_map;
    transition_ordering_map["State"] =
        ParentIndexing(0, state_cardinality, x_cardinality);
    transition_ordering_map["X"] = ParentIndexing(1, x_cardinality, 1);
    transition_ordering_map["Y"] = ParentIndexing(2, y_cardinality, 1);

    SegmentTransitionFactorNode transition_factor("f:Transition",
                                                  time_step,
                                                  transition_probs,
                                                  transition_ordering_map,
                                                  duration_ordering_map);

    VarNodePtr segment = make_shared<VariableNode>("Segment", time_step);
    VarNodePtr x = make_shared<VariableNode>("X", time_step, x_cardinality);
    VarNodePtr y = make_shared<VariableNode>("Y", time_step, y_cardinality);

    int num_data_points = 1;
    Eigen::MatrixXd prob_x(num_data_points, x_cardinality);
    prob_x << 0.3, 0.7;
    Tensor3 msg_from_x = Tensor3(prob_x);

    Eigen::MatrixXd prob_y(num_data_points, y_cardinality);
    prob_y << 0.2, 0.8;
    Tensor3 msg_from_y = Tensor3(prob_y);

    transition_factor.set_incoming_message_from(
        x, time_step, time_step, msg_from_x, MessageNode::Direction::forward);
    transition_factor.set_incoming_message_from(
        y, time_step, time_step, msg_from_y, MessageNode::Direction::forward);

    int num_rows = x_cardinality;
    vector<Eigen::MatrixXd> msg_from_seg_matrices(num_data_points);
    msg_from_seg_matrices[0] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    msg_from_seg_matrices[0] << 0.011036383235143, 0.001493612051036,
        0.001617107279781, 0.018963616764857, 0.028506387948964,
        0.238382892720219, 0.009473469826563, 0.001282094722211,
        0.001388101218933, 0.060526530173437, 0.068717905277788,
        0.558611898781066;
    Tensor3 msg_from_seg(msg_from_seg_matrices);

    transition_factor.set_incoming_message_from(
        expansion_factor,
        time_step,
        time_step,
        msg_from_seg,
        MessageNode::Direction::forward);

    Eigen::MatrixXd obs_matrix(num_rows, (time_step + 1) * state_cardinality);
    obs_matrix << 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1;
    Tensor3 obs(obs_matrix);

    transition_factor.set_incoming_message_from(
        segment, time_step, time_step, obs, MessageNode::Direction::backwards);

    Tensor3 msg2seg = transition_factor.get_outward_message_to(
        segment, time_step, time_step, MessageNode::Direction::forward);

    vector<Eigen::MatrixXd> expected_msg2seg_matrices(num_data_points);
    expected_msg2seg_matrices[0] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    expected_msg2seg_matrices[0] << 0.001654856006770, 0.004873010686862,
        0.007619235872327, 0.018963616764857, 0.028506387948964,
        0.238382892720219, 0.001004073807963, 0.002704657446413,
        0.008434934513331, 0.060526530173437, 0.068717905277788,
        0.558611898781066;

    Tensor3 expected_msg2seg(expected_msg2seg_matrices);

    BOOST_TEST(check_tensor_eq(msg2seg, expected_msg2seg));

    Tensor3 msg2y = transition_factor.get_outward_message_to(
        y, time_step, time_step, MessageNode::Direction::backwards);
    msg2y = (msg2y * msg_from_y);
    msg2y.normalize_rows();

    Eigen::MatrixXd expected_msg2y_matrix(num_data_points, y_cardinality);
    expected_msg2y_matrix << 0.20057599096994938037, 0.79942400903005061963;
    Tensor3 expected_msg2y(expected_msg2y_matrix);

    BOOST_TEST(check_tensor_eq(msg2y, expected_msg2y));
}

BOOST_AUTO_TEST_CASE(segment_marginalization_factor) {
    // Testing whether the marginalization factor node is working
    // properly at producing an output message to a segment and to a
    // timer controlled node that regulates a segment.

    int state_cardinality = 3;
    int time_step = 2;
    SegmentMarginalizationFactorNode marginalization_factor(
        "f:Marginalization", time_step, 1, "State");
    VarNodePtr segment = make_shared<VariableNode>("Segment", time_step);
    VarNodePtr state =
        make_shared<VariableNode>("State", time_step, state_cardinality);

    int num_data_points = 3;
    Eigen::MatrixXd msg_from_state_matrix(num_data_points, state_cardinality);
    msg_from_state_matrix << 0.3, 0.4, 0.3, 0.2, 0.5, 0.3, 0.1, 0.3, 0.6;
    Tensor3 msg_from_state(msg_from_state_matrix);

    int num_duration_parents = 2;
    vector<Eigen::MatrixXd> msg_from_segment_matrices(num_data_points);
    msg_from_segment_matrices[0] = Eigen::MatrixXd(
        num_duration_parents, (time_step + 1) * state_cardinality);
    msg_from_segment_matrices[0] << 0.1, 0.2, 0.1, 0.6, 0.7, 0.4, 0.6, 0.7, 0.3,
        0.2, 0.4, 0.9, 0.7, 0.5, 0.6, 0.2, 0.3, 0.4;
    msg_from_segment_matrices[1] = Eigen::MatrixXd(
        num_duration_parents, (time_step + 1) * state_cardinality);
    msg_from_segment_matrices[1] << 0.3, 0.1, 0.2, 0.5, 0.4, 0.6, 0.7, 0.8, 0.3,
        0.4, 0.2, 0.5, 0.7, 0.8, 0.4, 0.8, 0.3, 0.7;
    msg_from_segment_matrices[2] = Eigen::MatrixXd(
        num_duration_parents, (time_step + 1) * state_cardinality);
    msg_from_segment_matrices[2] << 0.4, 0.3, 0.2, 0.6, 0.8, 0.4, 0.9, 0.3, 0.4,
        0.8, 0.3, 0.6, 0.7, 0.8, 0.7, 0.5, 0.4, 0.1;
    Tensor3 msg_from_segment(msg_from_segment_matrices);

    marginalization_factor.set_incoming_message_from(
        state,
        time_step,
        time_step,
        msg_from_state,
        MessageNode::Direction::backwards);
    marginalization_factor.set_incoming_message_from(
        segment,
        time_step,
        time_step,
        msg_from_segment,
        MessageNode::Direction::forward);

    Eigen::MatrixXd expected_msg2state_matrix(num_data_points,
                                              state_cardinality);
    expected_msg2state_matrix << 2.4, 2.8, 2.7, 3.4, 2.6, 2.7, 3.9, 2.9, 2.4;
    Tensor3 expected_msg2state(expected_msg2state_matrix);

    Tensor3 msg2state = marginalization_factor.get_outward_message_to(
        state, time_step, time_step, MessageNode::Direction::forward);

    BOOST_TEST(check_tensor_eq(msg2state, expected_msg2state));

    Eigen::MatrixXd expected_msg2segment_matrix(
        num_data_points, state_cardinality * (time_step + 1));
    expected_msg2segment_matrix << 0.3, 0.4, 0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 0.3,
        0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.1, 0.3,
        0.6, 0.1, 0.3, 0.6;
    Tensor3 expected_msg2segment(expected_msg2segment_matrix);
    expected_msg2segment = expected_msg2segment.reshape(
        num_data_points, 1, state_cardinality * (time_step + 1));

    Tensor3 msg2segment = marginalization_factor.get_outward_message_to(
        segment, time_step, time_step, MessageNode::Direction::backwards);

    BOOST_TEST(check_tensor_eq(msg2segment, expected_msg2segment));
}

BOOST_AUTO_TEST_CASE(edhmm_exact_inference) {
    // Testing whether exact inference works for a simple edhmm model
    // with duration an transition dependent on a multitime node.

    fs::current_path("../../test");
    cout << fs::current_path() << endl;

    EvidenceSet data("data/edhmm_exact");

    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/edhmm_exact.json"));
    model->unroll(data.get_time_steps(), true);

    FactorGraph::create_from_unrolled_dbn(*model).print_graph(cout);

    SumProductEstimator state_estimator(model, 0, "State");

    state_estimator.set_show_progress(true);
    state_estimator.prepare();
    state_estimator.estimate(data);

    MatrixXd state_estimates = state_estimator.get_estimates().estimates[1];

    cout << state_estimates << endl;

    SumProductEstimator x_estimator(model, 0, "X");

    x_estimator.set_show_progress(true);
    x_estimator.prepare();
    x_estimator.estimate(data);

    MatrixXd x_estimates = x_estimator.get_estimates().estimates[0];

    cout << x_estimates << endl;

    //    BOOST_TEST(check_tensor_eq(msg2segment, expected_msg2segment));
}

BOOST_AUTO_TEST_SUITE_END()