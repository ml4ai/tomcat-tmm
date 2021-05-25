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
#include "pgm/inference/MarginalizationFactorNode.h"
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
        FactorGraph::create_from_unrolled_dbn(*pre_trained_model).print_graph(cout);
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
                                                 make_shared<Poisson>(6)};

    CPD::TableOrderingMap duration_ordering_map;
    duration_ordering_map["State"] =
        ParentIndexing(0, state_cardinality, x_cardinality * y_cardinality);
    duration_ordering_map["X"] = ParentIndexing(1, x_cardinality, 1);

    CPD::TableOrderingMap total_ordering_map;
    total_ordering_map["State"] =
        ParentIndexing(0, state_cardinality, x_cardinality * y_cardinality);
    total_ordering_map["X"] = ParentIndexing(1, x_cardinality, y_cardinality);
    total_ordering_map["Y"] = ParentIndexing(2, y_cardinality, 1);

    SegmentExpansionFactorNode expansion_factor("f:Expansion",
                                                1,
                                                duration_distributions,
                                                duration_ordering_map,
                                                total_ordering_map);

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
    VarNodePtr xy =
        make_shared<VariableNode>("XY", 1, x_cardinality * y_cardinality);

    int num_data_points = 2;
    Eigen::MatrixXd prob_xy(num_data_points, x_cardinality * y_cardinality);
    prob_xy << 0.09, 0.21, 0.21, 0.49, 0.04, 0.16, 0.16, 0.64;
    Tensor3 msg_from_xy = Tensor3(prob_xy);

    Eigen::MatrixXd msg_from_seg0_matrix(x_cardinality * y_cardinality,
                                         state_cardinality);
    msg_from_seg0_matrix << 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1,
        0.1, 0.8;
    vector<Eigen::MatrixXd> msg_from_seg0_matrices(num_data_points);
    msg_from_seg0_matrices[0] = msg_from_seg0_matrix;
    msg_from_seg0_matrices[1] = msg_from_seg0_matrix;
    Tensor3 msg_from_seg0(msg_from_seg0_matrices);

    // First time step
    int time_step = 1;
    expansion_factor.set_incoming_message_from(segment,
                                               time_step - 1,
                                               time_step,
                                               msg_from_seg0,
                                               MessageNode::Direction::forward);
    expansion_factor.set_incoming_message_from(
        xy, time_step, time_step, msg_from_xy, MessageNode::Direction::forward);

    Tensor3 msg2trans = expansion_factor.get_outward_message_to(
        transition_factor,
        time_step,
        time_step,
        MessageNode::Direction::forward);

    // Suppose the transition table is the identity and the message that
    // comes from it is just a replica of the message from the next state
    vector<Eigen::MatrixXd> msg_from_trans_matrices(num_data_points);
    msg_from_trans_matrices[0] = Eigen::MatrixXd(
        x_cardinality * y_cardinality, (time_step + 1) * state_cardinality);
    msg_from_trans_matrices[0] << 0.100000000000000, 0.380000000000000,
        0.380000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.100000000000000, 0.450000000000000,
        0.450000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.100000000000000, 0.450000000000000,
        0.450000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.100000000000000, 0.450000000000000,
        0.450000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000;
    msg_from_trans_matrices[1] = msg_from_trans_matrices[0];
    Tensor3 msg_from_trans(msg_from_trans_matrices);

    expansion_factor.set_incoming_message_from(
        transition_factor,
        time_step,
        time_step,
        msg_from_trans,
        MessageNode::Direction::backwards);

    Tensor3 msg2xy = expansion_factor.get_outward_message_to(
        xy, time_step, time_step, MessageNode::Direction::backwards);

    int num_rows = x_cardinality * y_cardinality;
    vector<Eigen::MatrixXd> expected_msg2trans_matrices(num_data_points);
    expected_msg2trans_matrices[0] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    expected_msg2trans_matrices[0] << 0.003310914970543, 0.000448083615311,
        0.000485132183934, 0.005689085029457, 0.008551916384689,
        0.071514867816066, 0.007725468264600, 0.001045528435725,
        0.001131975095846, 0.013274531735400, 0.019954471564275,
        0.166868024904154, 0.002842040947969, 0.000384628416663,
        0.000416430365680, 0.018157959052031, 0.020615371583337,
        0.167583569634320, 0.006631428878594, 0.000897466305548,
        0.000971670853253, 0.042368571121406, 0.048102533694452,
        0.391028329146747;
    expected_msg2trans_matrices[1] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    expected_msg2trans_matrices[1] << 0.001471517764686, 0.000199148273471,
        0.000215614303971, 0.002528482235314, 0.003800851726529,
        0.031784385696029, 0.005886071058743, 0.000796593093886,
        0.000862457215883, 0.010113928941257, 0.015203406906114,
        0.127137542784117, 0.002165364531786, 0.000293050222220,
        0.000317280278613, 0.013834635468214, 0.015706949777780,
        0.127682719721387, 0.008661458127143, 0.001172200888879,
        0.001269121114453, 0.055338541872857, 0.062827799111121,
        0.510730878885547;
    Tensor3 expected_msg2trans(expected_msg2trans_matrices);

    Eigen::MatrixXd expected_msg2xy_matrix(num_data_points,
                                           x_cardinality * y_cardinality);
    expected_msg2xy_matrix << 0.237820037329873, 0.238993098190703,
        0.261593432239712, 0.261593432239712, 0.237820037329873,
        0.238993098190703, 0.261593432239712, 0.261593432239712;
    Tensor3 expected_msg2xy(expected_msg2xy_matrix);

    BOOST_TEST(check_tensor_eq(msg2trans, expected_msg2trans));
    BOOST_TEST(check_tensor_eq(msg2xy, expected_msg2xy));
}

BOOST_AUTO_TEST_CASE(segment_transition_factor) {
    // Testing whether the transition factor node is working properly at
    // producing an output message to the final segment and for a transition
    // dependency that is not a dependency of the segment duration distribution.

    int state_cardinality = 3;
    int x_cardinality = 2;
    int y_cardinality = 2;

    // P(duration | State, X, Y)
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

    CPD::TableOrderingMap total_ordering_map;
    total_ordering_map["State"] =
        ParentIndexing(0, state_cardinality, x_cardinality * y_cardinality);
    {}
    total_ordering_map["X"] = ParentIndexing(1, x_cardinality, y_cardinality);
    total_ordering_map["Y"] = ParentIndexing(2, y_cardinality, 1);

    SegExpFactorNodePtr expansion_factor =
        make_shared<SegmentExpansionFactorNode>("f:Expansion",
                                                1,
                                                duration_distributions,
                                                duration_ordering_map,
                                                total_ordering_map);

    // P(transition | State, X, Y)
    Eigen::MatrixXd transition_probs(
        state_cardinality * x_cardinality * y_cardinality, state_cardinality);
    transition_probs << 0, 0.4, 0.6, 0, 0.5, 0.5, 0, 0.5, 0.5, 0, 0.5, 0.5, 0.4,
        0, 0.6, 0.5, 0, 0.5, 0.5, 0, 0.5, 0.5, 0, 0.5, 0.4, 0.6, 0, 0.5, 0.5, 0,
        0.5, 0.5, 0, 0.5, 0.5, 0;

    CPD::TableOrderingMap transition_ordering_map;
    transition_ordering_map["State"] =
        ParentIndexing(0, state_cardinality, x_cardinality * y_cardinality);
    transition_ordering_map["X"] =
        ParentIndexing(1, x_cardinality, y_cardinality);
    transition_ordering_map["Y"] = ParentIndexing(2, y_cardinality, 1);

    int time_step = 1;
    SegmentTransitionFactorNode transition_factor("f:Transition",
                                                  time_step,
                                                  transition_probs,
                                                  transition_ordering_map,
                                                  total_ordering_map);
    VarNodePtr segment = make_shared<VariableNode>("Segment", time_step);

    int num_rows = x_cardinality * y_cardinality;
    int num_data_points = 2;
    vector<Eigen::MatrixXd> msg_from_ext_factor_matrices(num_data_points);
    msg_from_ext_factor_matrices[0] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    msg_from_ext_factor_matrices[0] << 0.003310914970543, 0.000448083615311,
        0.000485132183934, 0.005689085029457, 0.008551916384689,
        0.071514867816066, 0.007725468264600, 0.001045528435725,
        0.001131975095846, 0.013274531735400, 0.019954471564275,
        0.166868024904154, 0.002842040947969, 0.000384628416663,
        0.000416430365680, 0.018157959052031, 0.020615371583337,
        0.167583569634320, 0.006631428878594, 0.000897466305548,
        0.000971670853253, 0.042368571121406, 0.048102533694452,
        0.391028329146747;
    msg_from_ext_factor_matrices[1] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    msg_from_ext_factor_matrices[1] << 0.001471517764686, 0.000199148273471,
        0.000215614303971, 0.002528482235314, 0.003800851726529,
        0.031784385696029, 0.005886071058743, 0.000796593093886,
        0.000862457215883, 0.010113928941257, 0.015203406906114,
        0.127137542784117, 0.002165364531786, 0.000293050222220,
        0.000317280278613, 0.013834635468214, 0.015706949777780,
        0.127682719721387, 0.008661458127143, 0.001172200888879,
        0.001269121114453, 0.055338541872857, 0.062827799111121,
        0.510730878885547;
    Tensor3 msg_from_ext_factor(msg_from_ext_factor_matrices);

    transition_factor.set_incoming_message_from(
        expansion_factor,
        time_step,
        time_step,
        msg_from_ext_factor,
        MessageNode::Direction::forward);

    // From expansion to transition factor
    Tensor3 msg2seg = transition_factor.get_outward_message_to(
        segment, time_step, time_step, MessageNode::Direction::forward);

    vector<Eigen::MatrixXd> expected_msg2seg_matrices(num_data_points);
    expected_msg2seg_matrices[0] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    expected_msg2seg_matrices[0] << 0.000373286319698, 0.001615445298578,
        0.002255399151512, 0.005689085029457, 0.008551916384689,
        0.071514867816066, 0.001088751765786, 0.004428721680223,
        0.004385498350163, 0.013274531735400, 0.019954471564275,
        0.166868024904154, 0.000400529391172, 0.001629235656824,
        0.001613334682316, 0.018157959052031, 0.020615371583337,
        0.167583569634320, 0.000934568579401, 0.003801549865924,
        0.003764447592071, 0.042368571121406, 0.048102533694452,
        0.391028329146747;
    expected_msg2seg_matrices[1] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    expected_msg2seg_matrices[1] << 0.000165905030977, 0.000717975688257,
        0.001002399622894, 0.002528482235314, 0.003800851726529,
        0.031784385696029, 0.000829525154884, 0.003374264137313,
        0.003341332076314, 0.010113928941257, 0.015203406906114,
        0.127137542784117, 0.000305165250417, 0.001241322405200,
        0.001229207377003, 0.013834635468214, 0.015706949777780,
        0.127682719721387, 0.001220661001666, 0.004965289620798,
        0.004916829508011, 0.055338541872857, 0.062827799111121,
        0.510730878885547;
    Tensor3 expected_msg2seg(expected_msg2seg_matrices);

    BOOST_TEST(check_tensor_eq(msg2seg, expected_msg2seg));

    // From transition to expansion factor
    vector<Eigen::MatrixXd> msg_from_seg_matrices(num_data_points);
    msg_from_seg_matrices[0] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    msg_from_seg_matrices[0] << 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000;
    msg_from_seg_matrices[1] = msg_from_seg_matrices[0];
    Tensor3 msg_from_seg(msg_from_seg_matrices);

    transition_factor.set_incoming_message_from(
        segment,
        time_step,
        time_step,
        msg_from_seg,
        MessageNode::Direction::backwards);

    Tensor3 msg2exp_factor = transition_factor.get_outward_message_to(
        expansion_factor,
        time_step,
        time_step,
        MessageNode::Direction::backwards);

    vector<Eigen::MatrixXd> expected_msg2ext_factor_matrices(num_data_points);
    expected_msg2ext_factor_matrices[0] =
        Eigen::MatrixXd(num_rows, (time_step + 1) * state_cardinality);
    expected_msg2ext_factor_matrices[0] << 0.100000000000000, 0.380000000000000,
        0.380000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.100000000000000, 0.450000000000000,
        0.450000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.100000000000000, 0.450000000000000,
        0.450000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000, 0.100000000000000, 0.450000000000000,
        0.450000000000000, 0.800000000000000, 0.100000000000000,
        0.100000000000000;
    expected_msg2ext_factor_matrices[1] = expected_msg2ext_factor_matrices[0];
    Tensor3 expected_msg2ext_factor(expected_msg2ext_factor_matrices);

    cout << msg2exp_factor << endl;

    BOOST_TEST(check_tensor_eq(msg2exp_factor, expected_msg2ext_factor));
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
    expected_msg2state_matrix << 0.303797468354430, 0.354430379746835,
        0.341772151898734, 0.390804597701149, 0.298850574712644,
        0.310344827586207, 0.423913043478261, 0.315217391304348,
        0.260869565217391;
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

BOOST_AUTO_TEST_CASE(marginalization_factor) {
    // Testing whether the marginalization factor node is working
    // properly at producing an output message to nodes from a joint
    // distribution.

    int x_cardinality = 2;
    int y_cardinality = 3;
    VarNodePtr x0 = make_shared<VariableNode>("X", 0, x_cardinality);
    VarNodePtr y0 = make_shared<VariableNode>("Y", 0, y_cardinality);
    VarNodePtr xy0 =
        make_shared<VariableNode>("XY", 0, x_cardinality * y_cardinality);
    VarNodePtr x1 = make_shared<VariableNode>("X", 1, x_cardinality);
    VarNodePtr y1 = make_shared<VariableNode>("Y", 1, y_cardinality);
    VarNodePtr xy1 =
        make_shared<VariableNode>("XY", 1, x_cardinality * y_cardinality);

    CPD::TableOrderingMap joint_ordering_map;
    joint_ordering_map["X"] = ParentIndexing(0, x_cardinality, y_cardinality);
    joint_ordering_map["Y"] = ParentIndexing(1, y_cardinality, 1);

    MarginalizationFactorNode marginalization_factor0(
        "XY", 0, joint_ordering_map, "XY");
    MarginalizationFactorNode marginalization_factor1(
        "XY", 1, joint_ordering_map, "XY");

    int num_data_points = 2;
    Eigen::MatrixXd msg_from_x0_matrix(num_data_points, x_cardinality);
    msg_from_x0_matrix << 0.3, 0.7, 0.2, 0.8;
    Tensor3 msg_from_x0(msg_from_x0_matrix);

    Eigen::MatrixXd msg_from_y0_matrix(num_data_points, y_cardinality);
    msg_from_y0_matrix << 0.2, 0.7, 0.1, 0.1, 0.5, 0.4;
    Tensor3 msg_from_y0(msg_from_y0_matrix);

    Eigen::MatrixXd msg_from_xy0_matrix(num_data_points,
                                        x_cardinality * y_cardinality);
    msg_from_xy0_matrix << 0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.2,
        0.1, 0.2;
    Tensor3 msg_from_xy0(msg_from_xy0_matrix);

    marginalization_factor0.set_incoming_message_from(
        x0, 0, 0, msg_from_x0, MessageNode::Direction::forward);
    marginalization_factor0.set_incoming_message_from(
        y0, 0, 0, msg_from_y0, MessageNode::Direction::forward);
    marginalization_factor0.set_incoming_message_from(
        xy0, 0, 0, msg_from_xy0, MessageNode::Direction::backwards);

    Eigen::MatrixXd expected_msg_2_x0_matrix(num_data_points, x_cardinality);
    expected_msg_2_x0_matrix << 0.6, 0.4, 0.5, 0.5;
    Tensor3 expected_msg_2_x0(expected_msg_2_x0_matrix);

    Eigen::MatrixXd expected_msg_2_y0_matrix(num_data_points, y_cardinality);
    expected_msg_2_y0_matrix << 0.3, 0.3, 0.4, 0.5, 0.2, 0.3;
    Tensor3 expected_msg_2_y0(expected_msg_2_y0_matrix);

    Eigen::MatrixXd expected_msg_2_xy0_matrix(num_data_points,
                                              x_cardinality * y_cardinality);
    expected_msg_2_xy0_matrix << 0.06, 0.21, 0.03, 0.14, 0.49, 0.07, 0.02, 0.1,
        0.08, 0.08, 0.4, 0.32;
    Tensor3 expected_msg_2_xy0(expected_msg_2_xy0_matrix);

    Tensor3 msg_2_x0 = marginalization_factor0.get_outward_message_to(
        x0, 0, 0, MessageNode::Direction::backwards);

    Tensor3 msg_2_y0 = marginalization_factor0.get_outward_message_to(
        y0, 0, 0, MessageNode::Direction::backwards);

    Tensor3 msg_2_xy0 = marginalization_factor0.get_outward_message_to(
        xy0, 0, 0, MessageNode::Direction::forward);

    BOOST_TEST(check_tensor_eq(msg_2_x0, expected_msg_2_x0));
    BOOST_TEST(check_tensor_eq(msg_2_y0, expected_msg_2_y0));
    BOOST_TEST(check_tensor_eq(msg_2_xy0, expected_msg_2_xy0));

    // Time step = 1
    Tensor3 msg_from_x1 = msg_2_x0;
    Tensor3 msg_from_y1 = msg_2_y0;

    marginalization_factor1.set_incoming_message_from(
        x1, 1, 1, msg_from_x1, MessageNode::Direction::forward);
    marginalization_factor1.set_incoming_message_from(
        y1, 1, 1, msg_from_y1, MessageNode::Direction::forward);

    Tensor3 expected_msg_2_xy1 =
        Tensor3::ones(1, num_data_points, x_cardinality * y_cardinality);

    Tensor3 msg_2_xy1 = marginalization_factor1.get_outward_message_to(
        xy1, 1, 1, MessageNode::Direction::forward);

    BOOST_TEST(check_tensor_eq(msg_2_xy1, expected_msg_2_xy1));
}

BOOST_AUTO_TEST_CASE(edhmm_exact_inference) {
    // Testing whether exact inference works for a simple edhmm model
    // with duration an transition dependent on a multitime node.
    fs::current_path("../../test");
    EvidenceSet data("data/edhmm_exact");

    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/edhmm_exact_copy.json"));
    model->unroll(3, true);

    SumProductEstimator state_estimator(model, 0, "State");

    state_estimator.set_show_progress(false);
    state_estimator.prepare();
    state_estimator.estimate(data);

    Tensor3 state_estimates(state_estimator.get_estimates().estimates);

    vector<Eigen::MatrixXd> expected_state_estimates_matrices(3);
    expected_state_estimates_matrices[0] =
        Eigen::MatrixXd(data.get_num_data_points(), data.get_time_steps());
    expected_state_estimates_matrices[0] << 0.100000000000000,
        0.417253617739875, 0.049976108391951, 0.100000000000000,
        0.013921719460383, 0.316267785180042, 0.800000000000000,
        0.316748103024800, 0.086619925711114;
    expected_state_estimates_matrices[1] =
        Eigen::MatrixXd(data.get_num_data_points(), data.get_time_steps());
    expected_state_estimates_matrices[1] << 0.100000000000000,
        0.066539088526785, 0.019746364616472, 0.800000000000000,
        0.966689373827991, 0.631479801372433, 0.100000000000000,
        0.574945549782578, 0.220553871813771;
    expected_state_estimates_matrices[2] =
        Eigen::MatrixXd(data.get_num_data_points(), data.get_time_steps());
    expected_state_estimates_matrices[2] << 0.800000000000000,
        0.516207293733340, 0.930277526991577, 0.100000000000000,
        0.019388906711627, 0.052252413447525, 0.100000000000000,
        0.108306347192622, 0.692826202475116;
    Tensor3 expected_state_estimates(expected_state_estimates_matrices);

    BOOST_TEST(check_tensor_eq(state_estimates, expected_state_estimates));

    SumProductEstimator x_estimator(model, 0, "X");

    x_estimator.set_show_progress(false);
    x_estimator.prepare();
    x_estimator.estimate(data);

    Tensor3 x_estimates(x_estimator.get_estimates().estimates);

    vector<Eigen::MatrixXd> expected_x_estimates_matrices(2);
    expected_x_estimates_matrices[0] =
        Eigen::MatrixXd(data.get_num_data_points(), data.get_time_steps());
    expected_x_estimates_matrices[0] << 0.300000000000000, 0.282106049062746,
        0.298877731367809, 0.300000000000000, 0.296934331973128,
        0.332403499904754, 0.300000000000000, 0.365234783011698,
        0.369020255880466;
    expected_x_estimates_matrices[1] =
        Eigen::MatrixXd(data.get_num_data_points(), data.get_time_steps());
    expected_x_estimates_matrices[1] << 0.700000000000000, 0.717893950937254,
        0.701122268632191, 0.700000000000000, 0.703065668026872,
        0.667596500095246, 0.700000000000000, 0.634765216988301,
        0.630979744119535;
    Tensor3 expected_x_estimates(expected_x_estimates_matrices);

    BOOST_TEST(check_tensor_eq(x_estimates, expected_x_estimates));

    SumProductEstimator y_estimator(model, 0, "Y");

    y_estimator.set_show_progress(false);
    y_estimator.prepare();
    y_estimator.estimate(data);

    Tensor3 y_estimates(y_estimator.get_estimates().estimates);

    vector<Eigen::MatrixXd> expected_y_estimates_matrices(2);
    expected_y_estimates_matrices[0] =
        Eigen::MatrixXd(data.get_num_data_points(), data.get_time_steps());
    expected_y_estimates_matrices[0] << 0.200000000000000, 0.200575990969949,
        0.203848346008490, 0.200000000000000, 0.199622438260025,
        0.191910586638231, 0.200000000000000, 0.190792475808668,
        0.206703106597843;
    expected_y_estimates_matrices[1] =
        Eigen::MatrixXd(data.get_num_data_points(), data.get_time_steps());
    expected_y_estimates_matrices[1] << 0.800000000000000, 0.799424009030051,
        0.796151653991510, 0.800000000000000, 0.800377561739975,
        0.808089413361769, 0.800000000000000, 0.809207524191332,
        0.793296893402157;
    Tensor3 expected_y_estimates(expected_y_estimates_matrices);

    BOOST_TEST(check_tensor_eq(y_estimates, expected_y_estimates));
}

BOOST_AUTO_TEST_SUITE_END()