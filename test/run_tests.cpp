#define BOOST_TEST_MODULE TomcatModelTest

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "boost/test/included/unit_test.hpp"
#include "eigen3/Eigen/Dense"
#include <gsl/gsl_rng.h>

#include "distribution/Gamma.h"
#include "distribution/Poisson.h"
#include "mock_models.h"
#include "pgm/EvidenceSet.h"
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

    model->unroll(4, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.sample(gen, 1);

    MatrixXd tcs = sampler.get_samples(TC)(0, 0);
    MatrixXd expected_tcs(1, 4);
    expected_tcs << NO_OBS, 0, 0, 0;
    auto check = check_matrix_eq(tcs, expected_tcs);
    BOOST_TEST(check.first, check.second);

    MatrixXd pbaes = sampler.get_samples(PBAE)(0, 0);
    MatrixXd expected_pbaes(1, 4);
    expected_pbaes << 1, 0, 1, 0;
    check = check_matrix_eq(pbaes, expected_pbaes);
    BOOST_TEST(check.first, check.second);

    MatrixXd states = sampler.get_samples(STATE)(0, 0);
    MatrixXd expected_states(1, 4);
    expected_states << 0, 1, 1, 2;
    check = check_matrix_eq(states, expected_states);
    BOOST_TEST(check.first, check.second);

    MatrixXd greens = sampler.get_samples(GREEN)(0, 0);
    MatrixXd expected_greens(1, 4);
    expected_greens << NO_OBS, 0, 0, 1;
    check = check_matrix_eq(greens, expected_greens);
    BOOST_TEST(check.first, check.second);

    MatrixXd yellows = sampler.get_samples(YELLOW)(0, 0);
    MatrixXd expected_yellows(1, 4);
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

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.set_max_time_step_to_sample(3);
    sampler.sample(gen, 1);

    MatrixXd tcs = sampler.get_samples(TC)(0, 0);
    MatrixXd expected_tcs(1, 4);
    expected_tcs << NO_OBS, 0, 0, 0;
    auto check = check_matrix_eq(tcs, expected_tcs);
    BOOST_TEST(check.first, check.second);

    MatrixXd pbaes = sampler.get_samples(PBAE)(0, 0);
    MatrixXd expected_pbaes(1, 4);
    expected_pbaes << 1, 0, 1, 0;
    check = check_matrix_eq(pbaes, expected_pbaes);
    BOOST_TEST(check.first, check.second);

    MatrixXd states = sampler.get_samples(STATE)(0, 0);
    MatrixXd expected_states(1, 4);
    expected_states << 0, 1, 1, 2;
    check = check_matrix_eq(states, expected_states);
    BOOST_TEST(check.first, check.second);

    MatrixXd greens = sampler.get_samples(GREEN)(0, 0);
    MatrixXd expected_greens(1, 4);
    expected_greens << NO_OBS, 0, 0, 1;
    check = check_matrix_eq(greens, expected_greens);
    BOOST_TEST(check.first, check.second);

    MatrixXd yellows = sampler.get_samples(YELLOW)(0, 0);
    MatrixXd expected_yellows(1, 4);
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

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.sample(gen, 1);

    MatrixXd tcs = sampler.get_samples(TC)(0, 0);
    MatrixXd expected_tcs = MatrixXd::Zero(1, 10);
    auto check = check_matrix_eq(tcs, expected_tcs);
    BOOST_TEST(check.first, check.second);

    MatrixXd pbaes = sampler.get_samples(PBAE)(0, 0);
    MatrixXd expected_pbaes(1, 10);
    expected_pbaes << 1, 0, 1, 0, 1, 0, 1, 0, 1, 0;
    check = check_matrix_eq(pbaes, expected_pbaes);
    BOOST_TEST(check.first, check.second);

    MatrixXd states = sampler.get_samples(STATE)(0, 0);
    MatrixXd expected_states(1, 10);
    expected_states << 0, 0, 1, 1, 1, 1, 2, 2, 2, 2;
    check = check_matrix_eq(states, expected_states);
    BOOST_TEST(check.first, check.second);

    MatrixXd greens = sampler.get_samples(GREEN)(0, 0);
    MatrixXd expected_greens(1, 10);
    expected_greens << NO_OBS, 1, 0, 0, 0, 0, 1, 1, 1, 1;
    check = check_matrix_eq(greens, expected_greens);
    BOOST_TEST(check.first, check.second);

    MatrixXd yellows = sampler.get_samples(YELLOW)(0, 0);
    MatrixXd expected_yellows(1, 10);
    expected_yellows << NO_OBS, 0, 1, 1, 1, 1, 0, 0, 0, 0;
    check = check_matrix_eq(yellows, expected_yellows);
    BOOST_TEST(check.first, check.second);

    MatrixXd timer = sampler.get_samples(TIMER)(0, 0);
    MatrixXd expected_timer(1, 10);
    expected_timer << 1, 0, 3, 2, 1, 0, 5, 4, 3, 2;
    check = check_matrix_eq(timer, expected_timer);
    BOOST_TEST(check.first, check.second);
}

BOOST_AUTO_TEST_SUITE_END()

// Training

BOOST_AUTO_TEST_SUITE(model_training)

BOOST_FIXTURE_TEST_CASE(gibbs_sampling, HMM) {
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
        make_shared<GibbsSampler>(model, 200, 4);
    DBNSamplingTrainer trainer(gen_training, gibbs_sampler, 200);

    double tolerance = 0.05;
    CPDTableCollection tables = this->create_cpd_tables(false);

    // Check parameter learning when tc is not provided.
    EvidenceSet data;
    data.add_data(PBAE, sampler.get_samples(PBAE));
    data.add_data(STATE, sampler.get_samples(STATE));
    data.add_data(GREEN, sampler.get_samples(GREEN));
    data.add_data(YELLOW, sampler.get_samples(YELLOW));

    // Set and freeze the first THETA_STATE_GIVEN_TC_PBAE_STATE to avoid
    // permutation of TC values.
    const shared_ptr<RandomVariableNode>& theta_state_0 =
        dynamic_pointer_cast<RandomVariableNode>(model->get_nodes_by_label(
            THETA_STATE_GIVEN_TC_PBAE_STATE + "_0")[0]);
    theta_state_0->set_assignment(tables.state_given_tc_pbae_state.row(0));
    theta_state_0->freeze();

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

    // Skip the first one that was set manually to avoid TC permutation and
    // therefore has no sample to be retrieved.
    for (int i = 0; i < NUM_THETA_STATE_GIVEN_TC_PBAE_STATE; i++) {
        stringstream label;
        label << THETA_STATE_GIVEN_TC_PBAE_STATE << '_' << i;
        MatrixXd estimated_theta_state_given_tc_pbae_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_theta_state_given_tc_pbae_state,
                                tables.state_given_tc_pbae_state.row(i),
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

    MatrixXd expected_tc(3, 3);
    expected_tc << 0.500324, 0.506335, 0.504282, 0.294363, 0.286289, 0.291487,
        0.205313, 0.207376, 0.204232;
    check = check_matrix_eq(tc_estimates[0], expected_tc.row(0));
    BOOST_TEST(check.first, check.second);
    check = check_matrix_eq(tc_estimates[1], expected_tc.row(1));
    BOOST_TEST(check.first, check.second);
    check = check_matrix_eq(tc_estimates[2], expected_tc.row(2));
    BOOST_TEST(check.first, check.second);
}

BOOST_AUTO_TEST_SUITE_END()
