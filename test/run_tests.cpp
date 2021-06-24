#define BOOST_TEST_MODULE TomcatModelTest

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "boost/test/included/unit_test.hpp"
#include "eigen3/Eigen/Dense"
#include <boost/filesystem.hpp>
#include <gsl/gsl_rng.h>

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "distribution/Distribution.h"
#include "distribution/Gamma.h"
#include "distribution/Poisson.h"
#include "pgm/EvidenceSet.h"
#include "pgm/inference/MarginalizationFactorNode.h"
#include "pgm/inference/ParticleFilter.h"
#include "pgm/inference/SegmentExpansionFactorNode.h"
#include "pgm/inference/SegmentMarginalizationFactorNode.h"
#include "pgm/inference/SegmentTransitionFactorNode.h"
#include "pgm/inference/VariableNode.h"
#include "pipeline/estimation/ParticleFilterEstimator.h"
#include "pipeline/estimation/SamplerEstimator.h"
#include "pipeline/estimation/SumProductEstimator.h"
#include "pipeline/estimation/custom_metrics/FinalScore.h"
#include "pipeline/training/DBNSamplingTrainer.h"
#include "sampling/AncestralSampler.h"
#include "sampling/GibbsSampler.h"
#include "test_helpers.h"

using namespace tomcat::model;
using namespace std;
using namespace Eigen;
namespace fs = boost::filesystem;
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

BOOST_GLOBAL_FIXTURE(Fixture);

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

BOOST_AUTO_TEST_CASE(complete) {
    /**
     * This test case checks if data can be generated correctly, following
     * the distributions defined in the model. A deterministic model is used
     * so that the values generated can be known in advance.
     */

    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/deterministic_dbn.json"));

    int time_steps = 4;
    model->unroll(time_steps, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.sample(gen, 1);

    MatrixXd fixed = sampler.get_samples("Fixed")(0, 0);
    MatrixXd expected_fixed(1, time_steps);
    expected_fixed << NO_OBS, 0, 0, 0;
    auto check = check_matrix_eq(fixed, expected_fixed);
    BOOST_TEST(check.first, check.second);

    MatrixXd movable = sampler.get_samples("Movable")(0, 0);
    MatrixXd expected_movable(1, time_steps);
    expected_movable << 1, 0, 1, 0;
    check = check_matrix_eq(movable, expected_movable);
    BOOST_TEST(check.first, check.second);

    MatrixXd states = sampler.get_samples("State")(0, 0);
    MatrixXd expected_states(1, time_steps);
    expected_states << 0, 1, 1, 2;
    check = check_matrix_eq(states, expected_states);
    BOOST_TEST(check.first, check.second);

    MatrixXd obs1 = sampler.get_samples("Obs1")(0, 0);
    MatrixXd expected_obs1(1, time_steps);
    expected_obs1 << NO_OBS, 0, 0, 1;
    check = check_matrix_eq(obs1, expected_obs1);
    BOOST_TEST(check.first, check.second);

    MatrixXd obs2 = sampler.get_samples("Obs2")(0, 0);
    MatrixXd expected_obs2(1, time_steps);
    expected_obs2 << NO_OBS, 1, 1, 0;
    check = check_matrix_eq(obs2, expected_obs2);
    BOOST_TEST(check.first, check.second);
}

BOOST_AUTO_TEST_CASE(heterogeneous) {
    /**
     * This test case checks if samples are correctly generated and vary
     * according to the distributions defined in the DBN. A non-deterministic
     * model is used so samples can have different values.
     */

    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn.json"));

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    int equal_samples_until = 4;
    int num_samples = 10;

    AncestralSampler sampler(model);
    sampler.sample(gen, num_samples);

    MatrixXd fixed = sampler.get_samples("Fixed")(0, 0);
    MatrixXd first_fixed = fixed.row(0);
    MatrixXd equal_samples_fixed = first_fixed.replicate<10, 1>();
    MatrixXd cropped_fixed =
        fixed.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_fixed =
        equal_samples_fixed.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!is_equal(cropped_fixed, cropped_etc_samples_fixed));

    MatrixXd movable = sampler.get_samples("Movable")(0, 0);
    MatrixXd first_movable = movable.row(0);
    MatrixXd equal_samples_movable = first_movable.replicate<10, 1>();
    MatrixXd cropped_movable =
        movable.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_movable =
        equal_samples_movable.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!is_equal(cropped_movable, cropped_etc_samples_movable));

    MatrixXd states = sampler.get_samples("State")(0, 0);
    MatrixXd first_states = states.row(0);
    MatrixXd equal_samples_states = first_states.replicate<10, 1>();
    MatrixXd cropped_states =
        states.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_states =
        equal_samples_states.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!is_equal(cropped_states, cropped_etc_samples_states));

    MatrixXd obs1 = sampler.get_samples("Obs1")(0, 0);
    MatrixXd first_obs1 = obs1.row(0);
    MatrixXd equal_samples_obs1 = first_obs1.replicate<10, 1>();
    MatrixXd cropped_obs1 = obs1.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_obs1 =
        equal_samples_obs1.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!is_equal(cropped_obs1, cropped_etc_samples_obs1));

    MatrixXd obs2 = sampler.get_samples("Obs2")(0, 0);
    MatrixXd first_obs2 = obs2.row(0);
    MatrixXd equal_samples_obs2 = first_obs2.replicate<10, 1>();
    MatrixXd cropped_obs2 = obs2.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_obs2 =
        equal_samples_obs2.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(!is_equal(cropped_obs2, cropped_etc_samples_obs2));
}

BOOST_AUTO_TEST_CASE(homogeneous) {
    /**
     * This test case checks if samples are correctly generated and present
     * the same values up to a certain time t. After this time, values are
     * generated independently and different samples can have different
     * values for the nodes in the DBN. A non-deterministic model is used.
     */

    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn.json"));

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    int equal_samples_until = 4;
    int num_samples = 10;

    AncestralSampler sampler(model);
    sampler.set_equal_samples_time_step_limit(equal_samples_until);
    sampler.sample(gen, num_samples);

    MatrixXd fixed = sampler.get_samples("Fixed")(0, 0);
    MatrixXd first_fixed = fixed.row(0);
    MatrixXd equal_samples_fixed = first_fixed.replicate<10, 1>();
    MatrixXd cropped_fixed =
        fixed.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_fixed =
        equal_samples_fixed.block(0, 0, num_samples, equal_samples_until);
    // Samples equal up to time 4 and above because tc don't change over time
    BOOST_TEST(is_equal(cropped_fixed, cropped_etc_samples_fixed));

    MatrixXd movable = sampler.get_samples("Movable")(0, 0);
    MatrixXd first_movable = movable.row(0);
    MatrixXd equal_samples_movable = first_movable.replicate<10, 1>();
    MatrixXd cropped_movable =
        movable.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_movable =
        equal_samples_movable.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(is_equal(cropped_movable, cropped_etc_samples_movable));
    BOOST_TEST(!is_equal(movable, equal_samples_movable));

    // Samplers differ after time step 4 for the other nodes.
    MatrixXd states = sampler.get_samples("State")(0, 0);
    MatrixXd first_states = states.row(0);
    MatrixXd equal_samples_states = first_states.replicate<10, 1>();
    MatrixXd cropped_states =
        states.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_states =
        equal_samples_states.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(is_equal(cropped_states, cropped_etc_samples_states));
    BOOST_TEST(!is_equal(states, equal_samples_states));

    MatrixXd obs1 = sampler.get_samples("Obs1")(0, 0);
    MatrixXd first_obs1 = obs1.row(0);
    MatrixXd equal_samples_obs1 = first_obs1.replicate<10, 1>();
    MatrixXd cropped_obs1 = obs1.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_obs1 =
        equal_samples_obs1.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(is_equal(cropped_obs1, cropped_etc_samples_obs1));
    BOOST_TEST(!is_equal(obs1, equal_samples_obs1));

    MatrixXd obs2 = sampler.get_samples("Obs2")(0, 0);
    MatrixXd first_obs2 = obs2.row(0);
    MatrixXd equal_samples_obs2 = first_obs2.replicate<10, 1>();
    MatrixXd cropped_obs2 = obs2.block(0, 0, num_samples, equal_samples_until);
    MatrixXd cropped_etc_samples_obs2 =
        equal_samples_obs2.block(0, 0, num_samples, equal_samples_until);
    BOOST_TEST(is_equal(cropped_obs2, cropped_etc_samples_obs2));
    BOOST_TEST(!is_equal(obs2, equal_samples_obs2));
}

BOOST_AUTO_TEST_CASE(semi_markov) {
    /**
     * This test case checks if data can be generated correctly, following
     * the distributions defined in the semi-Markov model. A deterministic model
     * is used so that the values generated can be known in advance.
     */

    DBNPtr model =
        make_shared<DynamicBayesNet>(DynamicBayesNet::create_from_json(
            "models/deterministic_semi_markov_dbn.json"));

    int time_steps = 15;
    model->unroll(time_steps, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    vector<int> deterministic_timer(
        {0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 2, 1, 0, 0});
    for (int t = 0; t < time_steps; t++) {
        auto timer = model->get_node("StateTimer", t);
        timer->set_assignment(
            Eigen::MatrixXd::Constant(1, 1, deterministic_timer[t]));
        timer->freeze();
    }

    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.sample(gen, 1);

    MatrixXd fixed = sampler.get_samples("Fixed")(0, 0);
    MatrixXd expected_fixed = MatrixXd::Zero(1, time_steps);
    auto check = check_matrix_eq(fixed, expected_fixed);
    BOOST_TEST(check.first, check.second);

    MatrixXd movable = sampler.get_samples("Movable")(0, 0);
    MatrixXd expected_movable(1, time_steps);
    expected_movable << 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;
    check = check_matrix_eq(movable, expected_movable);
    BOOST_TEST(check.first, check.second);

    MatrixXd states = sampler.get_samples("State")(0, 0);
    MatrixXd expected_states(1, time_steps);
    expected_states << 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0;
    check = check_matrix_eq(states, expected_states);
    BOOST_TEST(check.first, check.second);

    MatrixXd obs1 = sampler.get_samples("Obs1")(0, 0);
    MatrixXd expected_obs1(1, time_steps);
    expected_obs1 << NO_OBS, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1;
    check = check_matrix_eq(obs1, expected_obs1);
    BOOST_TEST(check.first, check.second);

    MatrixXd obs2 = sampler.get_samples("Obs2")(0, 0);
    MatrixXd expected_obs2(1, time_steps);
    expected_obs2 << NO_OBS, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0;
    check = check_matrix_eq(obs2, expected_obs2);
    BOOST_TEST(check.first, check.second);
}

BOOST_AUTO_TEST_SUITE_END()

// Training

BOOST_AUTO_TEST_SUITE(model_training)

BOOST_AUTO_TEST_CASE(dbn) {
    /**
     * This test case checks if the model can learn the parameters of a
     * non-deterministic model, given data generated from such a model.
     * Observations for node Fixed are not provided to the sampler to capture
     * the ability of the procedure to learn the parameters given that some
     * nodes are hidden.
     */

    DBNPtr oracle = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn.json"));
    oracle->unroll(20, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Generate a bunch of samples to train a model from the scratch.
    AncestralSampler sampler(oracle);
    sampler.sample(gen, 1000);

    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/trainable_dbn.json"));
    model->unroll(20, true);

    shared_ptr<gsl_rng> gen_training(gsl_rng_alloc(gsl_rng_mt19937));
    shared_ptr<GibbsSampler> gibbs_sampler =
        make_shared<GibbsSampler>(model, 200, 1);
    gibbs_sampler->set_show_progress(false);
    DBNSamplingTrainer trainer(gen_training, gibbs_sampler, 200);

    double tolerance = 0.05;

    // Check parameter learning when Fixed is not provided.
    EvidenceSet data;
    data.add_data("Movable", sampler.get_samples("Movable"));
    data.add_data("State", sampler.get_samples("State"));
    data.add_data("Obs1", sampler.get_samples("Obs1"));
    data.add_data("Obs2", sampler.get_samples("Obs2"));

    // Fix some parameters to avoid permutation of Fixed and Movable
    auto theta_fixed_cpd = get_cpd_table(oracle, "Fixed", true);
    const shared_ptr<RandomVariableNode>& theta_fixed =
        dynamic_pointer_cast<RandomVariableNode>(
            model->get_nodes_by_label("ThetaFixed_0")[0]);
    Eigen::MatrixXd fixed_prior(1, 3);
    theta_fixed->set_assignment(theta_fixed_cpd);
    theta_fixed->freeze();

    auto theta_state_given_others_cpd = get_cpd_table(oracle, "State", false);
    for (int i = 0; i < 18; i = i + 2) {
        stringstream label;
        label << "ThetaState.State.Fixed.Movable_" << i;
        const shared_ptr<RandomVariableNode>& theta_state =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        theta_state->set_assignment(theta_state_given_others_cpd.row(i));
        theta_state->freeze();
    }

    trainer.prepare();
    trainer.fit(data);

    // Check trained values
    auto expected_movable_prior = get_cpd_table(oracle, "Movable", true);
    MatrixXd estimated_movable_prior =
        model->get_nodes_by_label("PiMovable_0")[0]->get_assignment();
    auto check = check_matrix_eq(
        estimated_movable_prior, expected_movable_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    auto expected_state_prior = get_cpd_table(oracle, "State", true);
    MatrixXd estimated_state_prior =
        model->get_nodes_by_label("ThetaState_0")[0]->get_assignment();
    check =
        check_matrix_eq(estimated_state_prior, expected_state_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    auto expected_movable_given_movable =
        get_cpd_table(oracle, "Movable", false);
    for (int i = 0; i < 2; i++) {
        stringstream label;
        label << "PiMovable.Movable_" << i;

        MatrixXd estimated_movable_given_movable =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_movable_given_movable,
                                expected_movable_given_movable.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    auto expected_state_given_others = get_cpd_table(oracle, "State", false);
    for (int i = 0; i < 18; i++) {
        stringstream label;
        label << "ThetaState.State.Fixed.Movable_" << i;
        MatrixXd estimated_state_given_others =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_state_given_others,
                                expected_state_given_others.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    auto expected_obs1_given_state = get_cpd_table(oracle, "Obs1", false);
    for (int i = 0; i < 3; i++) {
        stringstream label;
        label << "PiObs1.State_" << i;
        MatrixXd estimated_obs1_given_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_obs1_given_state,
                                expected_obs1_given_state.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    auto expected_obs2_given_state = get_cpd_table(oracle, "Obs2", false);
    for (int i = 0; i < 3; i++) {
        stringstream label;
        label << "PiObs2.State_" << i;
        MatrixXd estimated_obs2_given_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_obs2_given_state,
                                expected_obs2_given_state.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }
}

BOOST_AUTO_TEST_CASE(semi_markov_durations) {
    /**
     * This test case checks if the model can learn the duration parameters of a
     * non-deterministic semi-Markov model, given data generated from such a
     * model. Observations for node Fixed are not provided to the sampler to
     * capture the ability of the procedure to learn the parameters given that
     * some nodes are hidden.
     */

    int time_steps = 120;
    DBNPtr oracle = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/semi_markov_dbn.json"));
    oracle->unroll(time_steps, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Generate a bunch of samples to train a model from the scratch.
    AncestralSampler sampler(oracle);
    sampler.sample(gen, 500);

    DBNPtr model =
        make_shared<DynamicBayesNet>(DynamicBayesNet::create_from_json(
            "models/trainable_semi_markov_dbn.json"));
    model->unroll(time_steps, true);

    shared_ptr<gsl_rng> gen_training(gsl_rng_alloc(gsl_rng_mt19937));
    shared_ptr<GibbsSampler> gibbs_sampler =
        make_shared<GibbsSampler>(model, 50, 1);
    gibbs_sampler->set_show_progress(false);
    DBNSamplingTrainer trainer(gen_training, gibbs_sampler, 100);

    // Given the distributions for state transition and emission of Obs1
    // and Obs2, observations of Movable, Obs1 and Obs2, can we determine the
    // distribution over Fixed and StateTimer?
    EvidenceSet data;
    data.add_data("Movable", sampler.get_samples("Movable"));
    data.add_data("Obs1", sampler.get_samples("Obs1"));
    data.add_data("Obs2", sampler.get_samples("Obs2"));

    // To avoid permutation, we need to provide some answers.
    auto fixed_prior_cpd = get_cpd_table(oracle, "Fixed", true);
    const shared_ptr<RandomVariableNode>& theta_fixed =
        dynamic_pointer_cast<RandomVariableNode>(
            model->get_nodes_by_label("ThetaFixed_0")[0]);
    theta_fixed->set_assignment(fixed_prior_cpd);
    theta_fixed->freeze();

    auto state_timer_given_others_cpd =
        get_cpd_table(oracle, "StateTimer", false);
    for (int i : {0, 3, 5, 6}) {
        stringstream label;
        label << "LambdaStateTimer.State.Fixed_" << i;
        const shared_ptr<RandomVariableNode>& lambda_timer =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        lambda_timer->set_assignment(state_timer_given_others_cpd.row(i));
        lambda_timer->freeze();
    }

    auto state_given_others_cpd = get_cpd_table(oracle, "State", false);
    for (int i = 0; i < 6; i++) {
        stringstream label;
        label << "ThetaState.State.Movable_" << i;
        const shared_ptr<RandomVariableNode>& theta_state =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        theta_state->set_assignment(state_given_others_cpd.row(i));
        theta_state->freeze();
    }

    auto obs1_given_state_cpd = get_cpd_table(oracle, "Obs1", false);
    for (int i = 0; i < 3; i++) {
        stringstream label;
        label << "PiObs1.State_" << i;
        const shared_ptr<RandomVariableNode>& pi_obs1 =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        pi_obs1->set_assignment(obs1_given_state_cpd.row(i));
        pi_obs1->freeze();
    }

    auto obs2_given_state_cpd = get_cpd_table(oracle, "Obs2", false);
    for (int i = 0; i < 3; i++) {
        stringstream label;
        label << "PiObs2.State_" << i;
        const shared_ptr<RandomVariableNode>& pi_obs2 =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        pi_obs2->set_assignment(obs2_given_state_cpd.row(i));
        pi_obs2->freeze();
    }

    trainer.prepare();
    trainer.fit(data);

    double tolerance = 0.1;
    auto expected_movable_prior = get_cpd_table(oracle, "Movable", true);
    MatrixXd estimated_movable_prior =
        model->get_nodes_by_label("PiMovable_0")[0]->get_assignment();
    auto check = check_matrix_eq(
        estimated_movable_prior, expected_movable_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    auto expected_state_prior = get_cpd_table(oracle, "State", true);
    MatrixXd estimated_state_prior =
        model->get_nodes_by_label("ThetaState_0")[0]->get_assignment();
    check =
        check_matrix_eq(estimated_state_prior, expected_state_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    auto expected_movable_given_movable =
        get_cpd_table(oracle, "Movable", false);
    for (int i = 0; i < 2; i++) {
        stringstream label;
        label << "PiMovable.Movable_" << i;
        MatrixXd estimated_movable_given_movable =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_movable_given_movable,
                                expected_movable_given_movable.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    tolerance = 1.5;
    auto expected_timer_given_others =
        get_cpd_table(oracle, "StateTimer", false);
    for (int i = 0; i < 9; i++) {
        stringstream label;
        label << "LambdaStateTimer.State.Fixed_" << i;
        MatrixXd estimated_timer_given_others =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_timer_given_others,
                                expected_timer_given_others.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }
}

BOOST_AUTO_TEST_CASE(semi_markov_transitions) {
    /**
     * This test case checks if the model can learn the transition parameters
     * of a* non-deterministic semi-Markov model, given data generated from such
     * a model. Observations for node Fixed are not provided to the sampler to
     * capture the ability of the procedure to learn the parameters given that
     * some nodes are hidden.
     */

    int time_steps = 120;
    DBNPtr oracle = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/semi_markov_dbn.json"));
    oracle->unroll(time_steps, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Generate a bunch of samples to train a model from the scratch.
    AncestralSampler sampler(oracle);
    sampler.sample(gen, 500);

    DBNPtr model =
        make_shared<DynamicBayesNet>(DynamicBayesNet::create_from_json(
            "models/trainable_semi_markov_dbn.json"));
    model->unroll(time_steps, true);

    shared_ptr<gsl_rng> gen_training(gsl_rng_alloc(gsl_rng_mt19937));
    shared_ptr<GibbsSampler> gibbs_sampler =
        make_shared<GibbsSampler>(model, 50, 1);
    gibbs_sampler->set_show_progress(false);
    DBNSamplingTrainer trainer(gen_training, gibbs_sampler, 100);

    // Given the distributions for state duration and emission of green
    // and yellow, observations of Movable, Obs1 and Obs2, can we determine the
    // distribution over training conditions and state transition?
    EvidenceSet data;
    data.add_data("Movable", sampler.get_samples("Movable"));
    data.add_data("Obs1", sampler.get_samples("Obs1"));
    data.add_data("Obs2", sampler.get_samples("Obs2"));

    // To avoid permutation, we need to provide some answers.
    auto fixed_prior_cpd = get_cpd_table(oracle, "Fixed", true);
    const shared_ptr<RandomVariableNode>& theta_fixed =
        dynamic_pointer_cast<RandomVariableNode>(
            model->get_nodes_by_label("ThetaFixed_0")[0]);
    theta_fixed->set_assignment(fixed_prior_cpd);
    theta_fixed->freeze();

    auto state_timer_given_others_cpd =
        get_cpd_table(oracle, "StateTimer", false);
    for (int i = 0; i < 9; i++) {
        stringstream label;
        label << "LambdaStateTimer.State.Fixed_" << i;
        const shared_ptr<RandomVariableNode>& lambda_timer =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        lambda_timer->set_assignment(state_timer_given_others_cpd.row(i));
        lambda_timer->freeze();
    }

    auto obs1_given_state_cpd = get_cpd_table(oracle, "Obs1", false);
    for (int i = 0; i < 3; i++) {
        stringstream label;
        label << "PiObs1.State_" << i;
        const shared_ptr<RandomVariableNode>& pi_obs1 =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        pi_obs1->set_assignment(obs1_given_state_cpd.row(i));
        pi_obs1->freeze();
    }

    auto obs2_given_state_cpd = get_cpd_table(oracle, "Obs2", false);
    for (int i = 0; i < 3; i++) {
        stringstream label;
        label << "PiObs2.State_" << i;
        const shared_ptr<RandomVariableNode>& pi_obs2 =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        pi_obs2->set_assignment(obs2_given_state_cpd.row(i));
        pi_obs2->freeze();
    }

    trainer.prepare();
    trainer.fit(data);

    double tolerance = 0.1;

    auto expected_movable_prior = get_cpd_table(oracle, "Movable", true);
    MatrixXd estimated_movable_prior =
        model->get_nodes_by_label("PiMovable_0")[0]->get_assignment();
    auto check = check_matrix_eq(
        estimated_movable_prior, expected_movable_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    auto expected_state_prior = get_cpd_table(oracle, "State", true);
    MatrixXd estimated_state_prior =
        model->get_nodes_by_label("ThetaState_0")[0]->get_assignment();
    check =
        check_matrix_eq(estimated_state_prior, expected_state_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    auto expected_movable_given_movable =
        get_cpd_table(oracle, "Movable", false);
    for (int i = 0; i < 2; i++) {
        stringstream label;
        label << "PiMovable.Movable_" << i;
        MatrixXd estimated_movable_given_movable =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_movable_given_movable,
                                expected_movable_given_movable.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    auto expected_state_given_others = get_cpd_table(oracle, "State", false);
    for (int i = 0; i < 6; i++) {
        stringstream label;
        label << "ThetaState.State.Movable_" << i;
        MatrixXd estimated_state_given_others =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_state_given_others,
                                expected_state_given_others.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }
}

BOOST_AUTO_TEST_SUITE_END()

// Inference

BOOST_AUTO_TEST_SUITE(estimation)

BOOST_AUTO_TEST_CASE(hmm_exact) {
    /**
     * This test case uses a HMM with 2 observed nodes per state. It evaluates
     * whether the probabilities of X (state variable) and projections over a
     * horizon of size 3 can be accurately determined at every time step using
     * sum-product.
     */

    int T = 50;
    int D = 2;
    int H = 3;

    // Data
    Eigen::MatrixXd z1(D, T);
    z1 << NO_OBS, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1, NO_OBS, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
        1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
        0, 0, 1, 1, 1, 0, 1, 0;
    Eigen::MatrixXd z2(D, T);
    z2 << NO_OBS, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0,
        0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
        1, 0, 0, 0, NO_OBS, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
        1, 0, 1, 1, 0, 0, 0, 0;

    EvidenceSet data;
    data.add_data("Z1", Tensor3(z1));
    data.add_data("Z2", Tensor3(z2));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/hmm.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Inference
    SumProductEstimator x_estimator(model, 0, "X");
    x_estimator.set_subgraph_window_size(1);
    x_estimator.set_show_progress(false);
    x_estimator.prepare();
    x_estimator.estimate(data);

    Eigen::MatrixXd expected_x_inference1(D, T);
    expected_x_inference1 << 0.3, 0.53057, 0.54798, 0.37506, 0.35156, 0.53596,
        0.13582, 0.31863, 0.34287, 0.53486, 0.37269, 0.12145, 0.31656, 0.53229,
        0.37215, 0.53785, 0.13605, 0.31867, 0.34288, 0.34649, 0.34709, 0.53538,
        0.37281, 0.53797, 0.37336, 0.53807, 0.37339, 0.35126, 0.11956, 0.51733,
        0.3695, 0.53745, 0.13599, 0.31866, 0.11667, 0.31582, 0.53219, 0.54828,
        0.37513, 0.35157, 0.11959, 0.1015, 0.31364, 0.34206, 0.34635, 0.34706,
        0.11916, 0.3162, 0.53224, 0.54829, 0.3, 0.53057, 0.37178, 0.53778,
        0.54929, 0.37539, 0.12173, 0.10167, 0.51552, 0.36914, 0.53738, 0.13599,
        0.51896, 0.36983, 0.35057, 0.5358, 0.54894, 0.55074, 0.55114, 0.37584,
        0.12178, 0.10168, 0.31366, 0.34207, 0.53475, 0.37266, 0.3511, 0.34788,
        0.34734, 0.34724, 0.34723, 0.11917, 0.10147, 0.5155, 0.36913, 0.53738,
        0.37322, 0.35122, 0.3479, 0.5355, 0.37284, 0.35114, 0.11954, 0.31627,
        0.34249, 0.34642, 0.53529, 0.37279, 0.53797, 0.37336;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.12611, 0.10663, 0.3904, 0.42499, 0.12156,
        0.77887, 0.47284, 0.43627, 0.12232, 0.39303, 0.80126, 0.47571, 0.12464,
        0.39354, 0.11978, 0.77859, 0.47279, 0.43627, 0.43142, 0.4307, 0.12195,
        0.39292, 0.11972, 0.39241, 0.11967, 0.39239, 0.42532, 0.80404, 0.14303,
        0.39653, 0.12001, 0.77865, 0.4728, 0.80819, 0.47667, 0.12471, 0.1065,
        0.39034, 0.42497, 0.804, 0.83186, 0.4797, 0.43727, 0.43159, 0.43073,
        0.80458, 0.47617, 0.12467, 0.1065, 0.5, 0.12611, 0.39389, 0.11981,
        0.10606, 0.39014, 0.80091, 0.83162, 0.1448, 0.3969, 0.12004, 0.77866,
        0.14143, 0.39621, 0.42603, 0.12165, 0.10622, 0.10517, 0.10503, 0.38977,
        0.80085, 0.83161, 0.47966, 0.43727, 0.12239, 0.39305, 0.42547, 0.42978,
        0.43042, 0.43053, 0.43055, 0.80455, 0.8319, 0.14482, 0.3969, 0.12004,
        0.39252, 0.42536, 0.42976, 0.12188, 0.3929, 0.42543, 0.80405, 0.47609,
        0.43675, 0.4315, 0.12201, 0.39294, 0.11972, 0.39241;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.34331, 0.34539, 0.23454, 0.22345, 0.34247,
        0.085311, 0.20853, 0.22085, 0.34282, 0.23428, 0.077286, 0.20773,
        0.34307, 0.23431, 0.34237, 0.085359, 0.20854, 0.22085, 0.22209, 0.22221,
        0.34267, 0.23427, 0.34231, 0.23423, 0.34226, 0.23423, 0.22342, 0.076406,
        0.33964, 0.23396, 0.34255, 0.085355, 0.20854, 0.075147, 0.20751, 0.3431,
        0.34522, 0.23452, 0.22345, 0.076415, 0.066639, 0.20666, 0.22067,
        0.22207, 0.22221, 0.076267, 0.20763, 0.34309, 0.34521, 0.2, 0.34331,
        0.23433, 0.34241, 0.34466, 0.23447, 0.077356, 0.06671, 0.33968, 0.23397,
        0.34258, 0.085354, 0.33961, 0.23396, 0.2234, 0.34255, 0.34484, 0.34409,
        0.34383, 0.23438, 0.077364, 0.066711, 0.20667, 0.22067, 0.34286,
        0.23429, 0.22343, 0.22234, 0.22223, 0.22222, 0.22222, 0.076271,
        0.066626, 0.33968, 0.23397, 0.34258, 0.23426, 0.22343, 0.22234, 0.34262,
        0.23426, 0.22343, 0.076404, 0.20764, 0.22076, 0.22208, 0.3427, 0.23427,
        0.34231, 0.23423;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator.get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(estimated_x_inference, expected_x_inference));

    // Prediction
    SumProductEstimator z1_predictor(model, H, "Z1", VectorXd::Constant(1, 0));
    z1_predictor.set_subgraph_window_size(1);
    z1_predictor.set_show_progress(false);
    z1_predictor.prepare();
    z1_predictor.estimate(data);

    Eigen::MatrixXd expected_z1_prediction1(D, T);
    expected_z1_prediction1 << 0.84887, 0.84287, 0.84238, 0.84687, 0.8475,
        0.84271, 0.8531, 0.84838, 0.84773, 0.84275, 0.84694, 0.85348, 0.84843,
        0.84282, 0.84695, 0.84266, 0.85309, 0.84838, 0.84773, 0.84764, 0.84762,
        0.84273, 0.84693, 0.84265, 0.84692, 0.84265, 0.84692, 0.84751, 0.85353,
        0.84323, 0.84703, 0.84267, 0.85309, 0.84838, 0.8536, 0.84845, 0.84282,
        0.84237, 0.84687, 0.8475, 0.85353, 0.854, 0.84851, 0.84776, 0.84764,
        0.84762, 0.85354, 0.84844, 0.84282, 0.84237, 0.84887, 0.84287, 0.84696,
        0.84266, 0.84234, 0.84686, 0.85347, 0.854, 0.84328, 0.84704, 0.84267,
        0.85309, 0.84319, 0.84702, 0.84752, 0.84272, 0.84235, 0.8423, 0.84229,
        0.84685, 0.85347, 0.854, 0.84851, 0.84776, 0.84275, 0.84694, 0.84751,
        0.8476, 0.84761, 0.84761, 0.84762, 0.85354, 0.854, 0.84328, 0.84704,
        0.84267, 0.84692, 0.84751, 0.8476, 0.84273, 0.84693, 0.84751, 0.85353,
        0.84844, 0.84774, 0.84764, 0.84273, 0.84693, 0.84265, 0.84692;
    Tensor3 expected_z1_prediction(expected_z1_prediction1);
    Tensor3 estimated_z1_prediction =
        Tensor3(z1_predictor.get_estimates().estimates);
    BOOST_TEST(
        check_tensor_eq(estimated_z1_prediction, expected_z1_prediction));
}

BOOST_AUTO_TEST_CASE(hmm_particle_filter) {
    /**
     * This test case uses a HMM with 2 observed nodes per state. It evaluates
     * whether the probabilities of X (state variable) and projections over a
     * horizon of size 3 can be approximated at every time step using particle
     * filter.
     */

    int T = 50;
    int D = 2;
    int H = 3;
    double tolerance = 0.1;
    int num_particles = 1000;

    // Data
    Eigen::MatrixXd z1(D, T);
    z1 << NO_OBS, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1, NO_OBS, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
        1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
        0, 0, 1, 1, 1, 0, 1, 0;
    Eigen::MatrixXd z2(D, T);
    z2 << NO_OBS, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0,
        0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
        1, 0, 0, 0, NO_OBS, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
        1, 0, 1, 1, 0, 0, 0, 0;

    EvidenceSet data;
    data.add_data("Z1", Tensor3(z1));
    data.add_data("Z2", Tensor3(z2));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/hmm.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Inference
    auto x_estimator = make_shared<SamplerEstimator>(model, 0, "X");
    auto z1_predictor =
        make_shared<SamplerEstimator>(model, H, "Z1", VectorXd::Constant(1, 0));

    ParticleFilterEstimator particle_estimator(model, num_particles, gen, 1);
    particle_estimator.add_base_estimator(x_estimator);
    particle_estimator.add_base_estimator(z1_predictor);
    particle_estimator.set_show_progress(false);

    particle_estimator.prepare();
    particle_estimator.estimate(data);

    Eigen::MatrixXd expected_x_inference1(D, T);
    expected_x_inference1 << 0.3, 0.53057, 0.54798, 0.37506, 0.35156, 0.53596,
        0.13582, 0.31863, 0.34287, 0.53486, 0.37269, 0.12145, 0.31656, 0.53229,
        0.37215, 0.53785, 0.13605, 0.31867, 0.34288, 0.34649, 0.34709, 0.53538,
        0.37281, 0.53797, 0.37336, 0.53807, 0.37339, 0.35126, 0.11956, 0.51733,
        0.3695, 0.53745, 0.13599, 0.31866, 0.11667, 0.31582, 0.53219, 0.54828,
        0.37513, 0.35157, 0.11959, 0.1015, 0.31364, 0.34206, 0.34635, 0.34706,
        0.11916, 0.3162, 0.53224, 0.54829, 0.3, 0.53057, 0.37178, 0.53778,
        0.54929, 0.37539, 0.12173, 0.10167, 0.51552, 0.36914, 0.53738, 0.13599,
        0.51896, 0.36983, 0.35057, 0.5358, 0.54894, 0.55074, 0.55114, 0.37584,
        0.12178, 0.10168, 0.31366, 0.34207, 0.53475, 0.37266, 0.3511, 0.34788,
        0.34734, 0.34724, 0.34723, 0.11917, 0.10147, 0.5155, 0.36913, 0.53738,
        0.37322, 0.35122, 0.3479, 0.5355, 0.37284, 0.35114, 0.11954, 0.31627,
        0.34249, 0.34642, 0.53529, 0.37279, 0.53797, 0.37336;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.12611, 0.10663, 0.3904, 0.42499, 0.12156,
        0.77887, 0.47284, 0.43627, 0.12232, 0.39303, 0.80126, 0.47571, 0.12464,
        0.39354, 0.11978, 0.77859, 0.47279, 0.43627, 0.43142, 0.4307, 0.12195,
        0.39292, 0.11972, 0.39241, 0.11967, 0.39239, 0.42532, 0.80404, 0.14303,
        0.39653, 0.12001, 0.77865, 0.4728, 0.80819, 0.47667, 0.12471, 0.1065,
        0.39034, 0.42497, 0.804, 0.83186, 0.4797, 0.43727, 0.43159, 0.43073,
        0.80458, 0.47617, 0.12467, 0.1065, 0.5, 0.12611, 0.39389, 0.11981,
        0.10606, 0.39014, 0.80091, 0.83162, 0.1448, 0.3969, 0.12004, 0.77866,
        0.14143, 0.39621, 0.42603, 0.12165, 0.10622, 0.10517, 0.10503, 0.38977,
        0.80085, 0.83161, 0.47966, 0.43727, 0.12239, 0.39305, 0.42547, 0.42978,
        0.43042, 0.43053, 0.43055, 0.80455, 0.8319, 0.14482, 0.3969, 0.12004,
        0.39252, 0.42536, 0.42976, 0.12188, 0.3929, 0.42543, 0.80405, 0.47609,
        0.43675, 0.4315, 0.12201, 0.39294, 0.11972, 0.39241;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.34331, 0.34539, 0.23454, 0.22345, 0.34247,
        0.085311, 0.20853, 0.22085, 0.34282, 0.23428, 0.077286, 0.20773,
        0.34307, 0.23431, 0.34237, 0.085359, 0.20854, 0.22085, 0.22209, 0.22221,
        0.34267, 0.23427, 0.34231, 0.23423, 0.34226, 0.23423, 0.22342, 0.076406,
        0.33964, 0.23396, 0.34255, 0.085355, 0.20854, 0.075147, 0.20751, 0.3431,
        0.34522, 0.23452, 0.22345, 0.076415, 0.066639, 0.20666, 0.22067,
        0.22207, 0.22221, 0.076267, 0.20763, 0.34309, 0.34521, 0.2, 0.34331,
        0.23433, 0.34241, 0.34466, 0.23447, 0.077356, 0.06671, 0.33968, 0.23397,
        0.34258, 0.085354, 0.33961, 0.23396, 0.2234, 0.34255, 0.34484, 0.34409,
        0.34383, 0.23438, 0.077364, 0.066711, 0.20667, 0.22067, 0.34286,
        0.23429, 0.22343, 0.22234, 0.22223, 0.22222, 0.22222, 0.076271,
        0.066626, 0.33968, 0.23397, 0.34258, 0.23426, 0.22343, 0.22234, 0.34262,
        0.23426, 0.22343, 0.076404, 0.20764, 0.22076, 0.22208, 0.3427, 0.23427,
        0.34231, 0.23423;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    // Prediction
    Eigen::MatrixXd expected_z1_prediction1(D, T);
    expected_z1_prediction1 << 0.84887, 0.84287, 0.84238, 0.84687, 0.8475,
        0.84271, 0.8531, 0.84838, 0.84773, 0.84275, 0.84694, 0.85348, 0.84843,
        0.84282, 0.84695, 0.84266, 0.85309, 0.84838, 0.84773, 0.84764, 0.84762,
        0.84273, 0.84693, 0.84265, 0.84692, 0.84265, 0.84692, 0.84751, 0.85353,
        0.84323, 0.84703, 0.84267, 0.85309, 0.84838, 0.8536, 0.84845, 0.84282,
        0.84237, 0.84687, 0.8475, 0.85353, 0.854, 0.84851, 0.84776, 0.84764,
        0.84762, 0.85354, 0.84844, 0.84282, 0.84237, 0.84887, 0.84287, 0.84696,
        0.84266, 0.84234, 0.84686, 0.85347, 0.854, 0.84328, 0.84704, 0.84267,
        0.85309, 0.84319, 0.84702, 0.84752, 0.84272, 0.84235, 0.8423, 0.84229,
        0.84685, 0.85347, 0.854, 0.84851, 0.84776, 0.84275, 0.84694, 0.84751,
        0.8476, 0.84761, 0.84761, 0.84762, 0.85354, 0.854, 0.84328, 0.84704,
        0.84267, 0.84692, 0.84751, 0.8476, 0.84273, 0.84693, 0.84751, 0.85353,
        0.84844, 0.84774, 0.84764, 0.84273, 0.84693, 0.84265, 0.84692;
    Tensor3 expected_z1_prediction(expected_z1_prediction1);
    Tensor3 estimated_z1_prediction =
        Tensor3(z1_predictor->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_z1_prediction, expected_z1_prediction, tolerance));
}

BOOST_AUTO_TEST_CASE(dbn1_particle_filter) {
    /**
     * This test case uses a DBN comprised of a single time node
     * that affects state transitions in a hidden markov chain. It
     * evaluates whether the probabilities of X (state variable) and A (single
     * time variable) can be approximated at every time step using particle
     * filter.
     */

    int T = 50;
    int D = 1;
    double tolerance = 0.1;
    int num_particles = 1000;

    // Data
    Eigen::MatrixXd z1(D, T);
    z1 << NO_OBS, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1;
    Eigen::MatrixXd z2(D, T);
    z2 << NO_OBS, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 1, 0;

    EvidenceSet data;
    data.add_data("Z1", Tensor3(z1));
    data.add_data("Z2", Tensor3(z2));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn1.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Inference
    auto x_estimator = make_shared<SamplerEstimator>(model, 0, "X");
    auto a_estimator = make_shared<SamplerEstimator>(model, 0, "A");

    ParticleFilterEstimator particle_estimator(model, num_particles, gen, 1);
    particle_estimator.add_base_estimator(x_estimator);
    particle_estimator.add_base_estimator(a_estimator);
    particle_estimator.set_show_progress(false);

    particle_estimator.prepare();
    particle_estimator.estimate(data);

    Eigen::MatrixXd expected_x_inference1(D, T);
    expected_x_inference1 << 0.3, 0.20686, 0.26081, 0.091587, 0.19559, 0.26166,
        0.36193, 0.35009, 0.19779, 0.258, 0.18594, 0.31891, 0.18081, 0.25179,
        0.074907, 0.18759, 0.07951, 0.086416, 0.17332, 0.24435, 0.32286,
        0.31139, 0.068757, 0.22885, 0.13958, 0.24029, 0.13531, 0.23973, 0.29928,
        0.27142, 0.063082, 0.2283, 0.30547, 0.078863, 0.086586, 0.083142,
        0.17114, 0.24167, 0.047666, 0.17586, 0.27465, 0.053374, 0.28866,
        0.25438, 0.10599, 0.07738, 0.078206, 0.28315, 0.10687, 0.23234;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.07642, 0.65228, 0.011904, 0.15051, 0.63111,
        0.1876, 0.30701, 0.087423, 0.64793, 0.052523, 0.37282, 0.07606, 0.65172,
        0.0094266, 0.14562, 0.022716, 0.029398, 0.13167, 0.6329, 0.44025,
        0.50717, 0.011232, 0.68078, 0.04122, 0.66067, 0.041901, 0.66023,
        0.16168, 0.6025, 0.0090601, 0.68072, 0.15611, 0.020166, 0.029589,
        0.028105, 0.1298, 0.63275, 0.0077845, 0.13425, 0.32285, 0.013389,
        0.37491, 0.54284, 0.048282, 0.024964, 0.026172, 0.3644, 0.061933,
        0.65189;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.71672, 0.086906, 0.89651, 0.6539, 0.10722,
        0.45046, 0.3429, 0.71479, 0.094072, 0.76154, 0.30827, 0.74313, 0.096488,
        0.91567, 0.6668, 0.89777, 0.88419, 0.69501, 0.12275, 0.23689, 0.18144,
        0.92001, 0.090368, 0.8192, 0.099034, 0.82279, 0.10004, 0.53904, 0.12608,
        0.92786, 0.090979, 0.53842, 0.90097, 0.88382, 0.88875, 0.69906, 0.12558,
        0.94455, 0.68989, 0.40249, 0.93324, 0.33643, 0.20278, 0.84573, 0.89766,
        0.89562, 0.35246, 0.8312, 0.11577;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.2, 0.14534, 0.16197, 0.081908, 0.074817,
        0.089479, 0.097408, 0.10606, 0.080739, 0.092024, 0.060798, 0.065108,
        0.047312, 0.054312, 0.025165, 0.022623, 0.015074, 0.011749, 0.010109,
        0.012671, 0.022735, 0.035409, 0.01669, 0.0185, 0.01132, 0.01309,
        0.0080522, 0.0093372, 0.010566, 0.013637, 0.00609, 0.0067637, 0.0075936,
        0.0044672, 0.0034598, 0.0026411, 0.0022588, 0.0028547, 0.0011801,
        0.0010251, 0.0011114, 0.00055054, 0.00057938, 0.0009271, 0.00057049,
        0.00038458, 0.00028113, 0.00029329, 0.00019232, 0.00023453;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.3, 0.36249, 0.33177, 0.44336, 0.47745, 0.43075,
        0.39272, 0.37262, 0.42114, 0.38625, 0.44455, 0.43442, 0.48459, 0.45085,
        0.55042, 0.58342, 0.64943, 0.70245, 0.7333, 0.69908, 0.61362, 0.5375,
        0.62833, 0.59867, 0.64665, 0.61959, 0.66663, 0.64044, 0.60314, 0.56599,
        0.65442, 0.6262, 0.58778, 0.64359, 0.69651, 0.74679, 0.77413, 0.74387,
        0.80557, 0.82559, 0.81686, 0.85769, 0.85502, 0.81914, 0.84604, 0.87115,
        0.8966, 0.8941, 0.90907, 0.89785;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.5, 0.49217, 0.50626, 0.47473, 0.44774, 0.47977,
        0.50987, 0.52132, 0.49813, 0.52173, 0.49465, 0.50047, 0.4681, 0.49483,
        0.42441, 0.39396, 0.3355, 0.2858, 0.25659, 0.28824, 0.36365, 0.42709,
        0.35498, 0.38283, 0.34203, 0.36732, 0.32531, 0.35022, 0.38629, 0.42037,
        0.33949, 0.36704, 0.40463, 0.35194, 0.30003, 0.25057, 0.22361, 0.25328,
        0.19325, 0.17339, 0.18203, 0.14176, 0.1444, 0.17994, 0.15339, 0.12847,
        0.10312, 0.1056, 0.090742, 0.10191;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(dbn2_particle_filter) {
    /**
     * This test case introduces a new single node in DBN1 that also affects
     * state transitions. It evaluates whether the probabilities of X (state
     * variable), A and B (single time variables) can be approximated at every
     * time step using particle filter.
     */

    //  This test needs more particles because the model contains a v-structure.
    //  Estimates degenerate after a couple of time steps due to entanglement
    //  between variables if not enough particles are used. A solution, besides
    //  increasing the number of particles, is to remove this kinds of
    //  structures from the model by merging them into a single node.

    int T = 50;
    int D = 1;
    double tolerance = 0.1;
    int num_particles = 4000;

    // Data
    Eigen::MatrixXd z1(D, T);
    z1 << NO_OBS, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1;
    Eigen::MatrixXd z2(D, T);
    z2 << NO_OBS, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 1, 0;

    EvidenceSet data;
    data.add_data("Z1", Tensor3(z1));
    data.add_data("Z2", Tensor3(z2));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn2.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Inference
    auto x_estimator = make_shared<SamplerEstimator>(model, 0, "X");
    auto a_estimator = make_shared<SamplerEstimator>(model, 0, "A");
    auto b_estimator = make_shared<SamplerEstimator>(model, 0, "B");

    ParticleFilterEstimator particle_estimator(model, num_particles, gen, 1);
    particle_estimator.add_base_estimator(x_estimator);
    particle_estimator.add_base_estimator(a_estimator);
    particle_estimator.add_base_estimator(b_estimator);
    particle_estimator.set_show_progress(false);

    particle_estimator.prepare();
    particle_estimator.estimate(data);

    Eigen::MatrixXd expected_x_inference1(D, T);
    expected_x_inference1 << 0.3, 0.32866, 0.42311, 0.17569, 0.27169, 0.40484,
        0.49671, 0.51176, 0.31878, 0.43453, 0.3258, 0.48392, 0.30055, 0.43597,
        0.14911, 0.27025, 0.12418, 0.12714, 0.25439, 0.41052, 0.4201, 0.44018,
        0.14358, 0.41127, 0.23818, 0.41218, 0.22053, 0.39522, 0.39235, 0.41756,
        0.11198, 0.36783, 0.37738, 0.11933, 0.11947, 0.11423, 0.23667, 0.37299,
        0.083375, 0.23944, 0.37545, 0.084059, 0.39555, 0.33502, 0.16771,
        0.10932, 0.10742, 0.38767, 0.16136, 0.34795;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.093835, 0.48605, 0.018578, 0.097304, 0.4901,
        0.21166, 0.21087, 0.072536, 0.46935, 0.073485, 0.23301, 0.068245,
        0.46528, 0.01513, 0.092493, 0.016736, 0.01914, 0.084627, 0.4701,
        0.44436, 0.44685, 0.015576, 0.48664, 0.064364, 0.48313, 0.063035,
        0.49901, 0.20432, 0.46733, 0.013469, 0.53098, 0.20219, 0.016637,
        0.020555, 0.019524, 0.089562, 0.50093, 0.012427, 0.096591, 0.2463,
        0.013126, 0.27692, 0.49205, 0.066976, 0.01897, 0.019334, 0.26927,
        0.062024, 0.53363;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.5775, 0.090842, 0.80573, 0.631, 0.10506,
        0.29163, 0.27737, 0.60869, 0.096118, 0.60072, 0.28307, 0.6312, 0.098751,
        0.83576, 0.63726, 0.85909, 0.85372, 0.66099, 0.11938, 0.13554, 0.11297,
        0.84085, 0.10209, 0.69746, 0.10469, 0.71644, 0.10576, 0.40333, 0.11511,
        0.87455, 0.10119, 0.42042, 0.86403, 0.85998, 0.86625, 0.67377, 0.12608,
        0.9042, 0.66397, 0.37825, 0.90282, 0.32754, 0.17294, 0.76531, 0.87171,
        0.87325, 0.34305, 0.77662, 0.11841;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.2, 0.23329, 0.21854, 0.30255, 0.31311, 0.28257,
        0.24645, 0.22324, 0.24688, 0.23044, 0.26215, 0.24902, 0.27876, 0.26318,
        0.33981, 0.3489, 0.39623, 0.41945, 0.43096, 0.40694, 0.37135, 0.34291,
        0.41324, 0.40614, 0.43655, 0.4223, 0.44449, 0.42913, 0.39235, 0.36802,
        0.42013, 0.40757, 0.375, 0.41939, 0.44385, 0.47062, 0.48412, 0.46162,
        0.49723, 0.50843, 0.49906, 0.53275, 0.53286, 0.51295, 0.51172, 0.53503,
        0.55393, 0.55307, 0.56395, 0.55052;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.3, 0.26901, 0.29987, 0.23976, 0.21334, 0.24359,
        0.24254, 0.23316, 0.21216, 0.23827, 0.21427, 0.20137, 0.18571, 0.20444,
        0.19692, 0.18179, 0.17637, 0.16722, 0.15567, 0.16104, 0.13412, 0.12252,
        0.13644, 0.14824, 0.15901, 0.1668, 0.18478, 0.19237, 0.17823, 0.17638,
        0.21557, 0.22543, 0.20911, 0.22423, 0.22212, 0.21878, 0.20985, 0.2118,
        0.26552, 0.25587, 0.24301, 0.27151, 0.25932, 0.23533, 0.26764, 0.26869,
        0.26495, 0.25339, 0.26733, 0.2742;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.5, 0.4977, 0.48158, 0.4577, 0.47355, 0.47385,
        0.51101, 0.54361, 0.54096, 0.53128, 0.52358, 0.54961, 0.53553, 0.53238,
        0.46326, 0.46931, 0.4274, 0.41333, 0.41337, 0.43202, 0.49453, 0.53458,
        0.45033, 0.44562, 0.40444, 0.4109, 0.37073, 0.3785, 0.42942, 0.4556,
        0.3643, 0.367, 0.41589, 0.35638, 0.33403, 0.31059, 0.30603, 0.32658,
        0.23726, 0.2357, 0.25793, 0.19573, 0.20782, 0.25173, 0.22064, 0.19628,
        0.18112, 0.19353, 0.16872, 0.17528;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));

    Eigen::MatrixXd expected_b_inference1(D, T);
    expected_b_inference1 << 0.1, 0.11283, 0.11946, 0.15812, 0.1449, 0.1492,
        0.13826, 0.12833, 0.13292, 0.14019, 0.16307, 0.15212, 0.16077, 0.16996,
        0.22234, 0.2043, 0.20512, 0.19242, 0.17887, 0.18427, 0.14461, 0.12233,
        0.15644, 0.1708, 0.19657, 0.20938, 0.23749, 0.25193, 0.23877, 0.23905,
        0.29266, 0.31445, 0.30118, 0.31128, 0.2926, 0.27538, 0.25809, 0.26676,
        0.32179, 0.30355, 0.29111, 0.31318, 0.30008, 0.28223, 0.31285, 0.30583,
        0.29368, 0.28169, 0.29293, 0.30399;
    Eigen::MatrixXd expected_b_inference2(D, T);
    expected_b_inference2 << 0.9, 0.88717, 0.88054, 0.84188, 0.8551, 0.8508,
        0.86174, 0.87167, 0.86708, 0.85981, 0.83693, 0.84788, 0.83923, 0.83004,
        0.77766, 0.7957, 0.79488, 0.80758, 0.82113, 0.81573, 0.85539, 0.87767,
        0.84356, 0.8292, 0.80343, 0.79062, 0.76251, 0.74807, 0.76123, 0.76095,
        0.70734, 0.68555, 0.69882, 0.68872, 0.7074, 0.72462, 0.74191, 0.73324,
        0.67821, 0.69645, 0.70889, 0.68682, 0.69992, 0.71777, 0.68715, 0.69417,
        0.70632, 0.71831, 0.70707, 0.69601;
    Tensor3 expected_b_inference((vector<Eigen::MatrixXd>){
        expected_b_inference1, expected_b_inference2});
    Tensor3 estimated_b_inference =
        Tensor3(b_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_b_inference, expected_b_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(dbn3_particle_filter) {
    /**
     * This test case uses a DBN comprised of 2 single time variables. One
     * affects the state transition in a hidden markov chain. The other affects
     * the former single time variable. It evaluates whether the probabilities
     * of X (state variable), A and B (single time variables) can be
     * approximated at every time step using particle filter.
     */

    int T = 50;
    int D = 1;
    double tolerance = 0.1;
    int num_particles = 1000;

    // Data
    Eigen::MatrixXd z1(D, T);
    z1 << NO_OBS, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1;
    Eigen::MatrixXd z2(D, T);
    z2 << NO_OBS, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 1, 0;

    EvidenceSet data;
    data.add_data("Z1", Tensor3(z1));
    data.add_data("Z2", Tensor3(z2));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn3.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Inference
    auto x_estimator = make_shared<SamplerEstimator>(model, 0, "X");
    auto a_estimator = make_shared<SamplerEstimator>(model, 0, "A");
    auto b_estimator = make_shared<SamplerEstimator>(model, 0, "B");

    ParticleFilterEstimator particle_estimator(model, num_particles, gen, 1);
    particle_estimator.add_base_estimator(x_estimator);
    particle_estimator.add_base_estimator(a_estimator);
    particle_estimator.add_base_estimator(b_estimator);
    particle_estimator.set_show_progress(false);

    particle_estimator.prepare();
    particle_estimator.estimate(data);

    Eigen::MatrixXd expected_x_inference1(D, T);
    expected_x_inference1 << 0.3, 0.20451, 0.27264, 0.086607, 0.19531, 0.2527,
        0.30436, 0.34265, 0.1883, 0.26181, 0.15367, 0.31847, 0.15374, 0.25012,
        0.054729, 0.17998, 0.068459, 0.080521, 0.16434, 0.23334, 0.25945,
        0.26208, 0.050818, 0.23203, 0.095906, 0.2357, 0.092796, 0.23465, 0.2212,
        0.24531, 0.041735, 0.23065, 0.22106, 0.064646, 0.079876, 0.076826,
        0.16255, 0.23115, 0.033862, 0.17051, 0.26048, 0.044936, 0.28603,
        0.23034, 0.085056, 0.073442, 0.075472, 0.28045, 0.095693, 0.22971;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.095408, 0.64228, 0.015878, 0.14129, 0.63363,
        0.22151, 0.31413, 0.09083, 0.64035, 0.061932, 0.36236, 0.07658, 0.64649,
        0.0098406, 0.13444, 0.020752, 0.026812, 0.12298, 0.63389, 0.46839,
        0.53101, 0.010744, 0.66981, 0.040534, 0.65748, 0.040027, 0.65753,
        0.1654, 0.61515, 0.0081501, 0.67025, 0.16084, 0.018768, 0.02668,
        0.025579, 0.1217, 0.63342, 0.0071984, 0.12823, 0.31796, 0.012709,
        0.3677, 0.55228, 0.046029, 0.024046, 0.02506, 0.35886, 0.060557, 0.6514;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.70008, 0.08508, 0.89751, 0.6634, 0.11367,
        0.47413, 0.34322, 0.72087, 0.09785, 0.7844, 0.31916, 0.76968, 0.1034,
        0.93543, 0.68558, 0.91079, 0.89267, 0.71269, 0.13277, 0.27216, 0.20691,
        0.93844, 0.098168, 0.86356, 0.10682, 0.86718, 0.10782, 0.6134, 0.13954,
        0.95012, 0.099096, 0.6181, 0.91659, 0.89344, 0.8976, 0.71575, 0.13543,
        0.95894, 0.70126, 0.42155, 0.94236, 0.34628, 0.21737, 0.86891, 0.90251,
        0.89947, 0.3607, 0.84375, 0.11889;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.47, 0.3593, 0.39911, 0.21345, 0.19072, 0.23245,
        0.25947, 0.28335, 0.21642, 0.25016, 0.16627, 0.17835, 0.12747, 0.14975,
        0.066184, 0.057667, 0.036385, 0.027048, 0.022649, 0.029212, 0.056066,
        0.092298, 0.041051, 0.046801, 0.027624, 0.032757, 0.019375, 0.023035,
        0.027074, 0.03619, 0.014906, 0.017041, 0.019933, 0.011101, 0.0081501,
        0.0059269, 0.004941, 0.0064218, 0.0025117, 0.0021428, 0.0023414,
        0.0011192, 0.0011805, 0.0019497, 0.0011718, 0.00077286, 0.0005528,
        0.00057792, 0.00037424, 0.00046072;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.39, 0.49573, 0.45225, 0.63915, 0.67329, 0.61904,
        0.5787, 0.55071, 0.62448, 0.58085, 0.67255, 0.6583, 0.72226, 0.68768,
        0.80082, 0.82268, 0.86713, 0.89456, 0.90885, 0.89161, 0.83709, 0.77506,
        0.85492, 0.83781, 0.87293, 0.85772, 0.88736, 0.87402, 0.85498, 0.83089,
        0.88609, 0.87278, 0.85351, 0.8847, 0.90764, 0.92708, 0.93678, 0.92569,
        0.94848, 0.95467, 0.95197, 0.96454, 0.96376, 0.95296, 0.96129, 0.96847,
        0.97529, 0.97463, 0.97859, 0.97569;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.14, 0.14497, 0.14864, 0.1474, 0.13599, 0.1485,
        0.16183, 0.16595, 0.15909, 0.16899, 0.16118, 0.16334, 0.15027, 0.16257,
        0.133, 0.11965, 0.096485, 0.078391, 0.068497, 0.079181, 0.10685,
        0.13264, 0.10403, 0.11539, 0.099447, 0.10952, 0.093267, 0.10294,
        0.11794, 0.13292, 0.099007, 0.11018, 0.12655, 0.1042, 0.084212,
        0.066997, 0.058281, 0.067886, 0.049006, 0.043185, 0.04569, 0.034337,
        0.035058, 0.045088, 0.037539, 0.03076, 0.024159, 0.024794, 0.021039,
        0.023853;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));

    Eigen::MatrixXd expected_b_inference1(D, T);
    expected_b_inference1 << 0.1, 0.1052, 0.10486, 0.11089, 0.10848, 0.11055,
        0.11335, 0.11369, 0.11407, 0.11568, 0.11637, 0.11657, 0.11465, 0.11733,
        0.11192, 0.10847, 0.10271, 0.09796, 0.095339, 0.098107, 0.10494,
        0.11092, 0.10466, 0.10765, 0.10384, 0.10649, 0.10239, 0.10498, 0.10904,
        0.11293, 0.10415, 0.10721, 0.1117, 0.10574, 0.10024, 0.095493, 0.093085,
        0.095725, 0.090569, 0.088951, 0.089646, 0.086507, 0.086706, 0.089491,
        0.087402, 0.085516, 0.083674, 0.083851, 0.082806, 0.083591;
    Eigen::MatrixXd expected_b_inference2(D, T);
    expected_b_inference2 << 0.9, 0.8948, 0.89514, 0.88911, 0.89152, 0.88945,
        0.88665, 0.88631, 0.88593, 0.88432, 0.88363, 0.88343, 0.88535, 0.88267,
        0.88808, 0.89153, 0.89729, 0.90204, 0.90466, 0.90189, 0.89506, 0.88908,
        0.89534, 0.89235, 0.89616, 0.89351, 0.89761, 0.89502, 0.89096, 0.88707,
        0.89585, 0.89279, 0.8883, 0.89426, 0.89976, 0.90451, 0.90692, 0.90427,
        0.90943, 0.91105, 0.91035, 0.91349, 0.91329, 0.91051, 0.9126, 0.91448,
        0.91633, 0.91615, 0.91719, 0.91641;
    Tensor3 expected_b_inference((vector<Eigen::MatrixXd>){
        expected_b_inference1, expected_b_inference2});
    Tensor3 estimated_b_inference =
        Tensor3(b_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_b_inference, expected_b_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(dbn4_particle_filter) {
    /**
     * This test case uses a DBN comprised of 2 single time variables. Both
     * affect the state transition in a hidden markov chain and one is dependent
     * of the other. It evaluates whether the probabilities of X (state
     * variable), A and B (single time variables) can be approximated at every
     * time step using particle filter.
     */

    int T = 50;
    int D = 1;
    double tolerance = 0.1;
    int num_particles = 2000;

    // Data
    Eigen::MatrixXd z1(D, T);
    z1 << NO_OBS, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1;
    Eigen::MatrixXd z2(D, T);
    z2 << NO_OBS, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 1, 0;

    EvidenceSet data;
    data.add_data("Z1", Tensor3(z1));
    data.add_data("Z2", Tensor3(z2));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn4.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Inference
    auto x_estimator = make_shared<SamplerEstimator>(model, 0, "X");
    auto a_estimator = make_shared<SamplerEstimator>(model, 0, "A");
    auto b_estimator = make_shared<SamplerEstimator>(model, 0, "B");

    ParticleFilterEstimator particle_estimator(model, num_particles, gen, 1);
    particle_estimator.add_base_estimator(x_estimator);
    particle_estimator.add_base_estimator(a_estimator);
    particle_estimator.add_base_estimator(b_estimator);
    particle_estimator.set_show_progress(false);

    particle_estimator.prepare();
    particle_estimator.estimate(data);

    Eigen::MatrixXd expected_x_inference1(D, T);
    expected_x_inference1 << 0.3, 0.2198, 0.32229, 0.10679, 0.2288, 0.31207,
        0.34424, 0.38002, 0.20714, 0.32552, 0.20042, 0.38847, 0.18503, 0.33327,
        0.078545, 0.23087, 0.087641, 0.1065, 0.21623, 0.32886, 0.25236, 0.25712,
        0.067037, 0.34573, 0.12427, 0.33818, 0.11988, 0.33293, 0.24824, 0.30271,
        0.058044, 0.33314, 0.24889, 0.080358, 0.10561, 0.10058, 0.21357,
        0.32119, 0.047237, 0.22607, 0.33507, 0.059946, 0.38321, 0.26764,
        0.11084, 0.095848, 0.10051, 0.37429, 0.1233, 0.327;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.10075, 0.57933, 0.019543, 0.11064, 0.56677,
        0.26816, 0.27403, 0.086687, 0.56832, 0.084671, 0.28952, 0.077946,
        0.55726, 0.015785, 0.1031, 0.017259, 0.020357, 0.091194, 0.53468,
        0.57065, 0.59522, 0.017384, 0.54774, 0.071357, 0.54735, 0.071063,
        0.55218, 0.25563, 0.56076, 0.014677, 0.56144, 0.24747, 0.017388,
        0.020801, 0.019529, 0.091447, 0.53954, 0.013314, 0.097271, 0.25656,
        0.013335, 0.2796, 0.54059, 0.075863, 0.018976, 0.019011, 0.27117,
        0.064424, 0.54889;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.67945, 0.098379, 0.87367, 0.66057, 0.12116,
        0.38761, 0.34595, 0.70617, 0.10616, 0.71491, 0.322, 0.73702, 0.10947,
        0.90567, 0.66603, 0.8951, 0.87315, 0.69258, 0.13646, 0.17698, 0.14765,
        0.91558, 0.10652, 0.80438, 0.11447, 0.80906, 0.11489, 0.49613, 0.13653,
        0.92728, 0.10542, 0.50364, 0.90225, 0.87359, 0.87989, 0.69499, 0.13927,
        0.93945, 0.67666, 0.40838, 0.92672, 0.33719, 0.19177, 0.81329, 0.88518,
        0.88048, 0.35454, 0.81227, 0.12412;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.47, 0.53322, 0.49333, 0.62197, 0.65262, 0.60116,
        0.56229, 0.54046, 0.58559, 0.54521, 0.59788, 0.59289, 0.63799, 0.60485,
        0.68327, 0.70567, 0.7485, 0.77436, 0.79142, 0.77126, 0.76087, 0.74569,
        0.78394, 0.76817, 0.78143, 0.76511, 0.77081, 0.75393, 0.73665, 0.71679,
        0.73593, 0.71712, 0.69991, 0.72792, 0.75209, 0.7754, 0.79048, 0.77452,
        0.77303, 0.78555, 0.78641, 0.79321, 0.79832, 0.79643, 0.7846, 0.79724,
        0.80946, 0.81376, 0.81375, 0.80448;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.39, 0.32936, 0.36868, 0.25009, 0.22256, 0.26569,
        0.28806, 0.29646, 0.25741, 0.29344, 0.24413, 0.24045, 0.20629, 0.23199,
        0.18015, 0.16527, 0.14482, 0.1318, 0.12066, 0.13068, 0.12297, 0.12397,
        0.11361, 0.12433, 0.12184, 0.13034, 0.13464, 0.14293, 0.14297, 0.14887,
        0.15706, 0.16587, 0.16471, 0.16153, 0.1549, 0.14721, 0.13934, 0.14528,
        0.1665, 0.15927, 0.15433, 0.16231, 0.15598, 0.14706, 0.16478, 0.16062,
        0.1552, 0.14945, 0.1545, 0.16054;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.14, 0.13742, 0.13799, 0.12794, 0.12483, 0.13315,
        0.14965, 0.16308, 0.157, 0.16135, 0.15799, 0.16666, 0.15572, 0.16317,
        0.13659, 0.12906, 0.10669, 0.093837, 0.087922, 0.098057, 0.11616,
        0.13035, 0.10245, 0.1075, 0.096724, 0.10455, 0.094549, 0.10314, 0.12038,
        0.13433, 0.10702, 0.11701, 0.13538, 0.11055, 0.093011, 0.077391,
        0.070179, 0.080193, 0.060473, 0.055176, 0.059258, 0.044475, 0.045691,
        0.056516, 0.05062, 0.042137, 0.035338, 0.036789, 0.031752, 0.03498;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));

    Eigen::MatrixXd expected_b_inference1(D, T);
    expected_b_inference1 << 0.1, 0.1077, 0.11392, 0.13345, 0.12337, 0.13067,
        0.13046, 0.12899, 0.12949, 0.13729, 0.15222, 0.14841, 0.14963, 0.1596,
        0.18061, 0.16661, 0.15572, 0.14255, 0.13172, 0.14018, 0.11916, 0.10722,
        0.11916, 0.12982, 0.14118, 0.15233, 0.16516, 0.17763, 0.18001, 0.18723,
        0.20558, 0.22201, 0.22568, 0.21654, 0.19859, 0.18166, 0.16869, 0.17921,
        0.2002, 0.18767, 0.18356, 0.18655, 0.17986, 0.17533, 0.19191, 0.18231,
        0.17168, 0.1658, 0.16908, 0.1777;
    Eigen::MatrixXd expected_b_inference2(D, T);
    expected_b_inference2 << 0.9, 0.8923, 0.88608, 0.86655, 0.87663, 0.86933,
        0.86954, 0.87101, 0.87051, 0.86271, 0.84778, 0.85159, 0.85037, 0.8404,
        0.81939, 0.83339, 0.84428, 0.85745, 0.86828, 0.85982, 0.88084, 0.89278,
        0.88084, 0.87018, 0.85882, 0.84767, 0.83484, 0.82237, 0.81999, 0.81277,
        0.79442, 0.77799, 0.77432, 0.78346, 0.80141, 0.81834, 0.83131, 0.82079,
        0.7998, 0.81233, 0.81644, 0.81345, 0.82014, 0.82467, 0.80809, 0.81769,
        0.82832, 0.8342, 0.83092, 0.8223;
    Tensor3 expected_b_inference((vector<Eigen::MatrixXd>){
        expected_b_inference1, expected_b_inference2});
    Tensor3 estimated_b_inference =
        Tensor3(b_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_b_inference, expected_b_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(dbn5_particle_filter) {
    /**
     * This test case uses a DBN comprised of a pair of semi-coupled hidden
     * markov chains. One chain affects the next state of the other. It
     * evaluates whether the probabilities of X and Y (state variables from both
     * chains) can be approximated at every time step using particle filter.
     */

    int T = 50;
    int D = 1;
    double tolerance = 0.1;
    int num_particles = 1000;

    // Data
    Eigen::MatrixXd z1(D, T);
    z1 << NO_OBS, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1;
    Eigen::MatrixXd z2(D, T);
    z2 << NO_OBS, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 1, 0;

    EvidenceSet data;
    data.add_data("Z1", Tensor3(z1));
    data.add_data("Z2", Tensor3(z2));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn5.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Inference
    auto x_estimator = make_shared<SamplerEstimator>(model, 0, "X");
    auto y_estimator = make_shared<SamplerEstimator>(model, 0, "Y");

    ParticleFilterEstimator particle_estimator(model, num_particles, gen, 1);
    particle_estimator.add_base_estimator(x_estimator);
    particle_estimator.add_base_estimator(y_estimator);
    particle_estimator.set_show_progress(false);

    particle_estimator.prepare();
    particle_estimator.estimate(data);

    Eigen::MatrixXd expected_x_inference1(D, T);
    expected_x_inference1 << 0.3, 0.84438, 0.23817, 0.87134, 0.6108, 0.38379,
        0.53684, 0.4575, 0.79409, 0.27663, 0.85754, 0.24, 0.86427, 0.23636,
        0.87183, 0.61064, 0.70376, 0.69021, 0.69411, 0.33475, 0.57009, 0.41782,
        0.80313, 0.32185, 0.84052, 0.25475, 0.86508, 0.24054, 0.63508, 0.40927,
        0.80568, 0.32118, 0.57779, 0.74877, 0.66176, 0.70359, 0.68884, 0.33776,
        0.83524, 0.63018, 0.36925, 0.82497, 0.29903, 0.58803, 0.72371, 0.64285,
        0.7168, 0.35255, 0.82958, 0.25633;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.7, 0.15562, 0.76183, 0.12866, 0.3892, 0.61621,
        0.46316, 0.5425, 0.20591, 0.72337, 0.14246, 0.76, 0.13573, 0.76364,
        0.12817, 0.38936, 0.29624, 0.30979, 0.30589, 0.66525, 0.42991, 0.58218,
        0.19687, 0.67815, 0.15948, 0.74525, 0.13492, 0.75946, 0.36492, 0.59073,
        0.19432, 0.67882, 0.42221, 0.25123, 0.33824, 0.29641, 0.31116, 0.66224,
        0.16476, 0.36982, 0.63075, 0.17503, 0.70097, 0.41197, 0.27629, 0.35715,
        0.2832, 0.64745, 0.17042, 0.74367;
    Tensor3 expected_x_inference((vector<Eigen::MatrixXd>){
        expected_x_inference1, expected_x_inference2});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_y_inference1(D, T);
    expected_y_inference1 << 0.8, 0.20072, 0.25792, 0.43138, 0.22622, 0.25514,
        0.43095, 0.40975, 0.23532, 0.25468, 0.25211, 0.43284, 0.23575, 0.25478,
        0.43175, 0.22617, 0.43017, 0.40124, 0.23238, 0.25474, 0.25174, 0.25262,
        0.42986, 0.23676, 0.25333, 0.25313, 0.25243, 0.25327, 0.43069, 0.23612,
        0.43199, 0.23657, 0.43305, 0.40395, 0.40353, 0.40548, 0.23177, 0.25479,
        0.43059, 0.22703, 0.43521, 0.40744, 0.41393, 0.23664, 0.25071, 0.42499,
        0.40323, 0.41407, 0.23617, 0.25467;
    Eigen::MatrixXd expected_y_inference2(D, T);
    expected_y_inference2 << 0.2, 0.79928, 0.74208, 0.56862, 0.77378, 0.74486,
        0.56905, 0.59025, 0.76468, 0.74532, 0.74789, 0.56716, 0.76425, 0.74522,
        0.56825, 0.77383, 0.56983, 0.59876, 0.76762, 0.74526, 0.74826, 0.74738,
        0.57014, 0.76324, 0.74667, 0.74687, 0.74757, 0.74673, 0.56931, 0.76388,
        0.56801, 0.76343, 0.56695, 0.59605, 0.59647, 0.59452, 0.76823, 0.74521,
        0.56941, 0.77297, 0.56479, 0.59256, 0.58607, 0.76336, 0.74929, 0.57501,
        0.59677, 0.58593, 0.76383, 0.74533;
    Tensor3 expected_y_inference((vector<Eigen::MatrixXd>){
        expected_y_inference1, expected_y_inference2});
    Tensor3 estimated_y_inference =
        Tensor3(y_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_y_inference, expected_y_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(dbn6_particle_filter) {
    /**
     * This test case uses a extends DBN 5 by introducing a aingle time node
     * that affects the state transition of both chains. It evaluates whether
     * the probabilities of X and Y (state variables from both chains) nd A
     * (single time node) can be approximated at every time step using particle
     * filter.
     */

    int T = 50;
    int D = 1;
    double tolerance = 0.1;
    int num_particles = 2000;

    // Data
    Eigen::MatrixXd z1(D, T);
    z1 << NO_OBS, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1;
    Eigen::MatrixXd z2(D, T);
    z2 << NO_OBS, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 1, 0;

    EvidenceSet data;
    data.add_data("Z1", Tensor3(z1));
    data.add_data("Z2", Tensor3(z2));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn6.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Inference
    auto x_estimator = make_shared<SamplerEstimator>(model, 0, "X");
    auto y_estimator = make_shared<SamplerEstimator>(model, 0, "Y");
    auto a_estimator = make_shared<SamplerEstimator>(model, 0, "A");

    ParticleFilterEstimator particle_estimator(model, num_particles, gen, 1);
    particle_estimator.add_base_estimator(x_estimator);
    particle_estimator.add_base_estimator(y_estimator);
    particle_estimator.add_base_estimator(a_estimator);
    particle_estimator.set_show_progress(false);

    particle_estimator.prepare();
    particle_estimator.estimate(data);

    Eigen::MatrixXd expected_x_inference1(D, T);
    expected_x_inference1 << 0.3, 0.74662, 0.46833, 0.81207, 0.7127, 0.5161,
        0.56809, 0.49093, 0.79051, 0.47045, 0.83539, 0.48511, 0.81495, 0.48091,
        0.86255, 0.73857, 0.84811, 0.78645, 0.81031, 0.56393, 0.62185, 0.59448,
        0.87444, 0.47154, 0.89074, 0.51547, 0.87994, 0.52409, 0.68589, 0.52007,
        0.89815, 0.46145, 0.7253, 0.81956, 0.84271, 0.82732, 0.81486, 0.60342,
        0.88664, 0.78881, 0.66797, 0.84202, 0.5873, 0.58917, 0.88082, 0.85568,
        0.82904, 0.59018, 0.84827, 0.60209;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.7, 0.25338, 0.53167, 0.18793, 0.2873, 0.4839,
        0.43191, 0.50907, 0.20949, 0.52955, 0.16461, 0.51489, 0.18505, 0.51909,
        0.13745, 0.26143, 0.15189, 0.21355, 0.18969, 0.43607, 0.37815, 0.40552,
        0.12556, 0.52846, 0.10926, 0.48453, 0.12006, 0.47591, 0.31411, 0.47993,
        0.10185, 0.53855, 0.2747, 0.18044, 0.15729, 0.17268, 0.18514, 0.39658,
        0.11336, 0.21119, 0.33203, 0.15798, 0.4127, 0.41083, 0.11918, 0.14432,
        0.17096, 0.40982, 0.15173, 0.39791;
    Tensor3 expected_x_inference((vector<Eigen::MatrixXd>){
        expected_x_inference1, expected_x_inference2});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_y_inference1(D, T);
    expected_y_inference1 << 0.8, 0.25853, 0.43536, 0.5737, 0.32552, 0.38588,
        0.55921, 0.49795, 0.36935, 0.37107, 0.39827, 0.54022, 0.32865, 0.36605,
        0.57729, 0.29984, 0.61764, 0.4766, 0.35997, 0.36015, 0.36057, 0.35941,
        0.60244, 0.25488, 0.44948, 0.31746, 0.42142, 0.32833, 0.57248, 0.26639,
        0.64227, 0.23666, 0.62444, 0.4806, 0.56385, 0.52273, 0.34931, 0.37756,
        0.61171, 0.30723, 0.60046, 0.50495, 0.49779, 0.30957, 0.44553, 0.58788,
        0.51681, 0.49296, 0.36632, 0.3741;
    Eigen::MatrixXd expected_y_inference2(D, T);
    expected_y_inference2 << 0.2, 0.74147, 0.56464, 0.4263, 0.67448, 0.61412,
        0.44079, 0.50205, 0.63065, 0.62893, 0.60173, 0.45978, 0.67135, 0.63395,
        0.42271, 0.70016, 0.38236, 0.5234, 0.64003, 0.63985, 0.63943, 0.64059,
        0.39756, 0.74512, 0.55052, 0.68254, 0.57858, 0.67167, 0.42752, 0.73361,
        0.35773, 0.76334, 0.37556, 0.5194, 0.43615, 0.47727, 0.65069, 0.62244,
        0.38829, 0.69277, 0.39954, 0.49505, 0.50221, 0.69043, 0.55447, 0.41212,
        0.48319, 0.50704, 0.63368, 0.6259;
    Tensor3 expected_y_inference((vector<Eigen::MatrixXd>){
        expected_y_inference1, expected_y_inference2});
    Tensor3 estimated_y_inference =
        Tensor3(y_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_y_inference, expected_y_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.2, 0.23162, 0.2514, 0.25229, 0.22524, 0.23683,
        0.21812, 0.20674, 0.21062, 0.22478, 0.23612, 0.23263, 0.25287, 0.2734,
        0.25554, 0.21976, 0.16247, 0.14066, 0.12161, 0.13071, 0.13428, 0.14234,
        0.11658, 0.12158, 0.11462, 0.12372, 0.12286, 0.13358, 0.12215, 0.12624,
        0.096779, 0.10047, 0.09083, 0.080851, 0.059135, 0.047982, 0.040872,
        0.044418, 0.037228, 0.030554, 0.029188, 0.027572, 0.027793, 0.028087,
        0.02283, 0.015937, 0.013164, 0.013139, 0.012911, 0.014325;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.3, 0.2088, 0.21939, 0.17617, 0.12609, 0.13357,
        0.16427, 0.2001, 0.1382, 0.14515, 0.096624, 0.11841, 0.07769, 0.081584,
        0.061494, 0.042326, 0.032202, 0.025073, 0.016236, 0.017423, 0.018903,
        0.020356, 0.015004, 0.015903, 0.009998, 0.010657, 0.0066938, 0.0071419,
        0.008996, 0.0095811, 0.0068639, 0.0072593, 0.0090315, 0.0070226,
        0.005067, 0.0037921, 0.002412, 0.0026159, 0.001875, 0.001204, 0.0014991,
        0.0011236, 0.0014145, 0.0015315, 0.00096635, 0.00069672, 0.00051348,
        0.00064776, 0.00040944, 0.00044445;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.5, 0.55958, 0.52921, 0.57154, 0.64867, 0.6296,
        0.61761, 0.59316, 0.65118, 0.63006, 0.66726, 0.64896, 0.66944, 0.64502,
        0.68297, 0.73792, 0.80533, 0.83427, 0.86215, 0.85186, 0.84681, 0.8373,
        0.86841, 0.86252, 0.87538, 0.86562, 0.87044, 0.85927, 0.86885, 0.86417,
        0.89636, 0.89227, 0.90014, 0.91213, 0.9358, 0.94823, 0.95672, 0.95297,
        0.9609, 0.96824, 0.96931, 0.9713, 0.97079, 0.97038, 0.9762, 0.98337,
        0.98632, 0.98621, 0.98668, 0.98523;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));
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

BOOST_AUTO_TEST_CASE(semi_markov_extended_hmm_exact) {
    /**
     * Test exact inference with sum product in a semi-markov HMM model with a
     * single time dependency on the state and another on the timer node.
     */

    // Data
    Eigen::MatrixXd obs(3, 3);
    obs << 2, 0, 2, 1, 1, 0, 0, 1, 2;

    EvidenceSet data;
    data.add_data("O", Tensor3(obs));

    DBNPtr model =
        make_shared<DynamicBayesNet>(DynamicBayesNet::create_from_json(
            "models/semi_markov_extended_hmm.json"));
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

BOOST_AUTO_TEST_CASE(semi_markov_extended_hmm_particle_filter) {
    /**
     * Test approximate inference with particle filter in a semi-markov HMM
     * model with a single time dependency on the state and another on the timer
     * node.
     */

    // Data
    Eigen::MatrixXd obs(3, 3);
    obs << 2, 0, 2, 1, 1, 0, 0, 1, 2;

    EvidenceSet data;
    data.add_data("O", Tensor3(obs));

    DBNPtr model =
        make_shared<DynamicBayesNet>(DynamicBayesNet::create_from_json(
            "models/semi_markov_extended_hmm.json"));
    model->unroll(3, true);

    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    auto state_estimator = make_shared<SamplerEstimator>(model, 0, "State");
    auto x_estimator = make_shared<SamplerEstimator>(model, 0, "X");
    auto y_estimator = make_shared<SamplerEstimator>(model, 0, "Y");

    ParticleFilterEstimator particle_estimator(model, 1000, gen, 1);
    particle_estimator.add_base_estimator(state_estimator);
    particle_estimator.add_base_estimator(x_estimator);
    particle_estimator.add_base_estimator(y_estimator);
    particle_estimator.set_show_progress(false);

    particle_estimator.prepare();
    particle_estimator.estimate(data);

    double tolerance = 0.1;

    Tensor3 state_estimates(state_estimator->get_estimates().estimates);

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

    BOOST_TEST(
        check_tensor_eq(state_estimates, expected_state_estimates, tolerance));

    Tensor3 x_estimates(x_estimator->get_estimates().estimates);

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

    BOOST_TEST(check_tensor_eq(x_estimates, expected_x_estimates, tolerance));

    Tensor3 y_estimates(y_estimator->get_estimates().estimates);

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

    BOOST_TEST(check_tensor_eq(y_estimates, expected_y_estimates, tolerance));
}

BOOST_AUTO_TEST_SUITE_END()