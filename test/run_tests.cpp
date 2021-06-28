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
     * This test case uses a extends DBN 5 by introducing a single time node
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

BOOST_AUTO_TEST_CASE(semi_markov_hmm_particle_filter) {
    /**
     * This test case uses a HSMM with 2 observed nodes per state. It evaluates
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
        DynamicBayesNet::create_from_json("models/semi_markov_hmm.json"));

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
    expected_x_inference1 << 0.3, 0.36433, 0.23522, 0.12947, 0.13496, 0.22404,
        0.17614, 0.20896, 0.22884, 0.33986, 0.26649, 0.10186, 0.13689, 0.29436,
        0.24518, 0.25782, 0.14063, 0.15936, 0.19156, 0.21935, 0.23016, 0.29955,
        0.23801, 0.24983, 0.23191, 0.27839, 0.27329, 0.2635, 0.11198, 0.30358,
        0.25954, 0.28197, 0.13453, 0.1492, 0.062435, 0.16117, 0.34264, 0.28531,
        0.18667, 0.16275, 0.093763, 0.047717, 0.15893, 0.22262, 0.22705,
        0.20238, 0.083828, 0.15474, 0.2971, 0.27517, 0.3, 0.36433, 0.18186,
        0.1585, 0.16969, 0.20429, 0.14787, 0.05994, 0.31639, 0.26837, 0.27411,
        0.11865, 0.2157, 0.21649, 0.22709, 0.30121, 0.31812, 0.3175, 0.3139,
        0.27631, 0.12445, 0.041322, 0.13005, 0.20007, 0.31941, 0.21401, 0.15898,
        0.16398, 0.19942, 0.23447, 0.25195, 0.10883, 0.047876, 0.33602, 0.26265,
        0.23508, 0.16769, 0.16734, 0.20197, 0.29733, 0.29537, 0.27495, 0.10906,
        0.15032, 0.19072, 0.21141, 0.28695, 0.21443, 0.2194, 0.21446;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.17823, 0.066024, 0.15012, 0.18512, 0.045029,
        0.40955, 0.4178, 0.40722, 0.10151, 0.18376, 0.64094, 0.57087, 0.14296,
        0.18901, 0.052544, 0.45712, 0.45373, 0.42057, 0.37244, 0.33085,
        0.075006, 0.15346, 0.048235, 0.13068, 0.043711, 0.1353, 0.21712,
        0.67877, 0.21507, 0.25193, 0.066767, 0.49024, 0.47368, 0.80335, 0.60719,
        0.12596, 0.038117, 0.13145, 0.19096, 0.61334, 0.85999, 0.64132, 0.45574,
        0.3395, 0.28407, 0.66058, 0.51504, 0.11127, 0.035096, 0.5, 0.17823,
        0.27791, 0.074615, 0.024482, 0.083712, 0.48763, 0.83348, 0.28141, 0.282,
        0.069769, 0.48583, 0.13628, 0.18107, 0.22084, 0.059137, 0.031257,
        0.02871, 0.028713, 0.1378, 0.61304, 0.88296, 0.70405, 0.51462, 0.098706,
        0.15767, 0.20593, 0.23173, 0.24451, 0.25622, 0.27274, 0.69206, 0.87747,
        0.24705, 0.2441, 0.060699, 0.13149, 0.17814, 0.2116, 0.055063, 0.1438,
        0.22813, 0.69169, 0.60812, 0.4975, 0.39548, 0.08118, 0.14745, 0.044218,
        0.11749;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.45743, 0.69875, 0.72041, 0.67992, 0.73093,
        0.4143, 0.37324, 0.36395, 0.55863, 0.54975, 0.25721, 0.29224, 0.56267,
        0.56582, 0.68964, 0.40225, 0.38691, 0.38788, 0.4082, 0.43898, 0.62544,
        0.60853, 0.70194, 0.6374, 0.6779, 0.59141, 0.51939, 0.20925, 0.48135,
        0.48852, 0.65127, 0.37523, 0.37712, 0.13421, 0.23165, 0.5314, 0.67658,
        0.68187, 0.64629, 0.29289, 0.092292, 0.19975, 0.32164, 0.43346, 0.51355,
        0.25559, 0.33022, 0.59163, 0.68973, 0.2, 0.45743, 0.54023, 0.76689,
        0.80583, 0.712, 0.36451, 0.10658, 0.4022, 0.44963, 0.65612, 0.39551,
        0.64802, 0.60244, 0.55207, 0.63965, 0.65062, 0.65379, 0.65738, 0.58589,
        0.26252, 0.075722, 0.1659, 0.28531, 0.58189, 0.62832, 0.63509, 0.60429,
        0.55607, 0.50931, 0.47531, 0.19911, 0.074653, 0.41692, 0.49325, 0.70422,
        0.70082, 0.65452, 0.58643, 0.64761, 0.56083, 0.49692, 0.19925, 0.24157,
        0.31178, 0.39311, 0.63187, 0.63811, 0.73638, 0.66805;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    // Prediction
    Eigen::MatrixXd expected_z1_prediction1(D, T);
    expected_z1_prediction1 << 0.8421, 0.76581, 0.72569, 0.73646, 0.74761,
        0.73425, 0.80866, 0.80413, 0.79545, 0.75062, 0.76228, 0.8365, 0.81145,
        0.74779, 0.75666, 0.73391, 0.81196, 0.80232, 0.79051, 0.7807, 0.77451,
        0.73954, 0.75372, 0.73448, 0.75354, 0.7414, 0.76236, 0.77643, 0.84486,
        0.76583, 0.76931, 0.73759, 0.81524, 0.80256, 0.83939, 0.80088, 0.74343,
        0.72789, 0.74214, 0.75394, 0.82678, 0.84704, 0.80428, 0.77728, 0.76356,
        0.75836, 0.81879, 0.79367, 0.741, 0.73054, 0.8421, 0.76581, 0.76458,
        0.72047, 0.71879, 0.74223, 0.82164, 0.86009, 0.77182, 0.76897, 0.73289,
        0.81, 0.74432, 0.75674, 0.76694, 0.74516, 0.74327, 0.74301, 0.74255,
        0.76297, 0.84282, 0.86316, 0.81714, 0.78427, 0.73517, 0.74409, 0.75178,
        0.7589, 0.76577, 0.77204, 0.7768, 0.83557, 0.84659, 0.76003, 0.75851,
        0.72509, 0.73892, 0.75191, 0.7635, 0.74599, 0.76645, 0.77941, 0.84663,
        0.81704, 0.79171, 0.77468, 0.73408, 0.74657, 0.72877, 0.74858;
    Tensor3 expected_z1_prediction(expected_z1_prediction1);
    Tensor3 estimated_z1_prediction =
        Tensor3(z1_predictor->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_z1_prediction, expected_z1_prediction, tolerance));
}

BOOST_AUTO_TEST_CASE(semi_markov_dbn1_particle_filter) {
    /**
     * This test case uses a DBN comprised of a single time node
     * that affects state transitions in a hidden semi-markov chain. The
     * duration of a segment only depends on the state variable. It evaluates
     * whether the probabilities of X (state variable) and A (single time
     * variable) can be approximated at every time step using particle filter.
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
        DynamicBayesNet::create_from_json("models/semi_markov_dbn1.json"));

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
    expected_x_inference1 << 0.3, 0.20758, 0.18265, 0.052451, 0.025652, 0.12116,
        0.16016, 0.19462, 0.13466, 0.19956, 0.14557, 0.17197, 0.084786, 0.17895,
        0.059318, 0.046883, 0.020284, 0.019871, 0.059754, 0.2142, 0.15159,
        0.10481, 0.10468, 0.19631, 0.090602, 0.17102, 0.10054, 0.18025, 0.21266,
        0.17687, 0.096911, 0.18351, 0.18587, 0.039183, 0.012289, 0.013481,
        0.04592, 0.19088, 0.093976, 0.07477, 0.17013, 0.04769, 0.15603, 0.23636,
        0.17984, 0.040457, 0.016847, 0.12663, 0.094254, 0.26967;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.20106, 0.56703, 0.036844, 0.01819, 0.39939,
        0.4449, 0.45197, 0.14362, 0.54171, 0.18066, 0.22225, 0.066756, 0.46978,
        0.035347, 0.032339, 0.0064902, 0.0064294, 0.042617, 0.55694, 0.80164,
        0.86481, 0.082101, 0.30636, 0.068041, 0.38936, 0.11742, 0.52796,
        0.48702, 0.72845, 0.05697, 0.32692, 0.31685, 0.016626, 0.0036045,
        0.0044649, 0.034884, 0.549, 0.062156, 0.054043, 0.20879, 0.016871,
        0.1552, 0.57603, 0.23748, 0.01355, 0.0032719, 0.092493, 0.044562,
        0.44677;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.59136, 0.25032, 0.91071, 0.95616, 0.47945,
        0.39494, 0.35341, 0.72173, 0.25873, 0.67377, 0.60578, 0.84846, 0.35127,
        0.90533, 0.92078, 0.97323, 0.9737, 0.89763, 0.22886, 0.046773, 0.030387,
        0.81321, 0.49733, 0.84136, 0.43962, 0.78203, 0.29178, 0.30032, 0.094677,
        0.84612, 0.48957, 0.49728, 0.94419, 0.98411, 0.98205, 0.9192, 0.26012,
        0.84387, 0.87119, 0.62108, 0.93544, 0.68877, 0.18762, 0.58268, 0.94599,
        0.97988, 0.78087, 0.86118, 0.28355;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.2, 0.19099, 0.21328, 0.17579, 0.17102, 0.1812,
        0.18, 0.17916, 0.16232, 0.18236, 0.16246, 0.16324, 0.15002, 0.16553,
        0.14067, 0.13662, 0.13304, 0.13023, 0.12687, 0.13546, 0.14672, 0.1542,
        0.13899, 0.15162, 0.13819, 0.15709, 0.13822, 0.15661, 0.15578, 0.17135,
        0.14764, 0.1618, 0.16254, 0.1407, 0.13793, 0.13665, 0.13465, 0.14073,
        0.11813, 0.11041, 0.10986, 0.092277, 0.09139, 0.10717, 0.087117,
        0.07187, 0.069217, 0.067707, 0.064215, 0.070599;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.3, 0.30492, 0.29066, 0.32364, 0.33086, 0.30754,
        0.32099, 0.32952, 0.3803, 0.32328, 0.37284, 0.37263, 0.39648, 0.37215,
        0.42585, 0.43701, 0.4472, 0.45915, 0.4773, 0.4342, 0.37759, 0.3419,
        0.41684, 0.36274, 0.39177, 0.3554, 0.38923, 0.36211, 0.37043, 0.3316,
        0.39705, 0.35454, 0.35137, 0.39143, 0.39407, 0.3936, 0.40001, 0.37993,
        0.48175, 0.51604, 0.52585, 0.60376, 0.61251, 0.53993, 0.63199, 0.69253,
        0.70202, 0.71112, 0.72559, 0.70017;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.5, 0.50409, 0.49606, 0.50057, 0.49812, 0.51126,
        0.49901, 0.49131, 0.45738, 0.49436, 0.4647, 0.46413, 0.4535, 0.46232,
        0.43348, 0.42637, 0.41976, 0.41062, 0.39583, 0.43034, 0.47569, 0.5039,
        0.44417, 0.48564, 0.47005, 0.48751, 0.47255, 0.48128, 0.47379, 0.49705,
        0.4553, 0.48366, 0.48609, 0.46787, 0.468, 0.46976, 0.46534, 0.47933,
        0.40012, 0.37355, 0.36429, 0.30396, 0.2961, 0.3529, 0.28089, 0.2356,
        0.22877, 0.22117, 0.21019, 0.22923;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(semi_markov_dbn1_1_particle_filter) {
    /**
     * This test case uses a DBN comprised of a single time node
     * that affects state transitions in a hidden semi-markov chain. The
     * duration of a segment on the state and single time variable. It
     * evaluates whether the probabilities of X (state variable) and A (single
     * time variable) can be approximated at every time step using particle
     * filter.
     */

    int T = 50;
    int D = 1;
    double tolerance = 0.1;
    int num_particles = 3000;

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
        DynamicBayesNet::create_from_json("models/semi_markov_dbn1_1.json"));

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
    expected_x_inference1 << 0.3, 0.29688, 0.36101, 0.19383, 0.10297, 0.28973,
        0.36522, 0.41504, 0.26345, 0.35455, 0.23838, 0.25888, 0.12729, 0.23666,
        0.073472, 0.045433, 0.015661, 0.012752, 0.036624, 0.18436, 0.17259,
        0.11088, 0.13735, 0.18901, 0.14696, 0.23123, 0.20415, 0.32391, 0.40297,
        0.38169, 0.22674, 0.4162, 0.45915, 0.14176, 0.028064, 0.010691,
        0.026657, 0.17593, 0.07006, 0.060382, 0.15504, 0.050707, 0.14396,
        0.20831, 0.23946, 0.088275, 0.026993, 0.1237, 0.10458, 0.29123;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.21135, 0.47628, 0.04776, 0.016473, 0.2391,
        0.25503, 0.26272, 0.093917, 0.40823, 0.15323, 0.19655, 0.063139,
        0.38689, 0.02799, 0.019915, 0.0036303, 0.0036499, 0.025711, 0.47786,
        0.76627, 0.87156, 0.19343, 0.53525, 0.18109, 0.50507, 0.16785, 0.45672,
        0.36975, 0.54478, 0.052024, 0.23129, 0.2093, 0.014589, 0.0020712,
        0.0018986, 0.014685, 0.37478, 0.029738, 0.033289, 0.17243, 0.015432,
        0.1916, 0.64259, 0.34752, 0.030527, 0.0062393, 0.12923, 0.055391,
        0.43078;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.49177, 0.16271, 0.75841, 0.88056, 0.47117,
        0.37975, 0.32224, 0.64263, 0.23721, 0.60839, 0.54457, 0.80957, 0.37645,
        0.89854, 0.93465, 0.98071, 0.9836, 0.93766, 0.33779, 0.061143, 0.017558,
        0.66922, 0.27574, 0.67196, 0.2637, 0.628, 0.21937, 0.22728, 0.073526,
        0.72123, 0.35251, 0.33155, 0.84366, 0.96986, 0.98741, 0.95866, 0.44929,
        0.9002, 0.90633, 0.67253, 0.93386, 0.66444, 0.1491, 0.41302, 0.8812,
        0.96677, 0.74707, 0.84003, 0.27799;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.2, 0.20329, 0.18882, 0.2181, 0.2256, 0.24076,
        0.22106, 0.20138, 0.17345, 0.19075, 0.16764, 0.16201, 0.1518, 0.17671,
        0.13953, 0.12921, 0.11521, 0.10251, 0.09213, 0.12337, 0.14282, 0.14137,
        0.24192, 0.1785, 0.22682, 0.1874, 0.19789, 0.19559, 0.17896, 0.19319,
        0.18784, 0.1697, 0.15023, 0.15947, 0.15912, 0.14557, 0.13065, 0.19375,
        0.10855, 0.093376, 0.093991, 0.077506, 0.076856, 0.080468, 0.0813,
        0.097908, 0.10521, 0.1026, 0.10164, 0.11084;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.3, 0.3056, 0.28505, 0.36865, 0.3969, 0.33542,
        0.32142, 0.31028, 0.33514, 0.30706, 0.32861, 0.32973, 0.34163, 0.33799,
        0.35764, 0.35529, 0.34366, 0.33191, 0.32682, 0.34909, 0.32737, 0.29016,
        0.53131, 0.3749, 0.50902, 0.34636, 0.45535, 0.34012, 0.32518, 0.28953,
        0.34635, 0.28645, 0.26004, 0.32309, 0.34009, 0.33274, 0.32351, 0.377,
        0.32588, 0.30816, 0.33475, 0.3094, 0.32733, 0.28333, 0.39367, 0.56644,
        0.64391, 0.64074, 0.71348, 0.62014;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.5, 0.49111, 0.52613, 0.41325, 0.3775, 0.42382,
        0.45752, 0.48834, 0.49141, 0.50218, 0.50375, 0.50826, 0.50657, 0.4853,
        0.50283, 0.5155, 0.54113, 0.56558, 0.58105, 0.52754, 0.5298, 0.56847,
        0.22677, 0.4466, 0.26415, 0.46624, 0.34676, 0.46429, 0.49586, 0.51728,
        0.46581, 0.54384, 0.58972, 0.51744, 0.5008, 0.52169, 0.54584, 0.42925,
        0.56557, 0.59847, 0.57126, 0.61309, 0.59582, 0.6362, 0.52503, 0.33566,
        0.25088, 0.25665, 0.18488, 0.26902;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(semi_markov_dbn2_particle_filter) {
    /**
     * This test case introduces a new single node in the semi-Markov DBN1 that
     * also affects state transitions. The duration of a segment only depends on
     * the state variable. It evaluates whether the probabilities of X (state
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
        DynamicBayesNet::create_from_json("models/semi_markov_dbn2.json"));

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
    expected_x_inference1 << 0.3, 0.21421, 0.20264, 0.072096, 0.041009, 0.20573,
        0.25391, 0.27683, 0.17049, 0.26396, 0.18397, 0.22024, 0.11219, 0.25033,
        0.081268, 0.065918, 0.028852, 0.028252, 0.083525, 0.31557, 0.23304,
        0.14914, 0.1466, 0.24952, 0.1255, 0.2257, 0.14048, 0.24918, 0.28514,
        0.2398, 0.13014, 0.23989, 0.2417, 0.054461, 0.01772, 0.019667, 0.067179,
        0.30042, 0.12408, 0.099003, 0.2288, 0.060194, 0.19391, 0.30805, 0.21272,
        0.04625, 0.019527, 0.1454, 0.10727, 0.30669;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.20172, 0.55366, 0.037532, 0.016438, 0.32235,
        0.35752, 0.36988, 0.11963, 0.46849, 0.15651, 0.19424, 0.060301, 0.40538,
        0.030532, 0.025553, 0.0048834, 0.0046517, 0.030704, 0.44106, 0.71031,
        0.81764, 0.086559, 0.31851, 0.07874, 0.38008, 0.11556, 0.46784, 0.42107,
        0.66314, 0.055089, 0.30793, 0.30131, 0.016904, 0.0029301, 0.003247,
        0.024837, 0.42339, 0.043242, 0.037792, 0.14557, 0.011933, 0.11469,
        0.48622, 0.19487, 0.01166, 0.0029123, 0.081849, 0.04016, 0.40834;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.58408, 0.24369, 0.89037, 0.94255, 0.47192,
        0.38857, 0.35329, 0.70988, 0.26755, 0.65952, 0.58552, 0.82751, 0.34429,
        0.8882, 0.90853, 0.96626, 0.9671, 0.88577, 0.24337, 0.056643, 0.03322,
        0.76684, 0.43197, 0.79576, 0.39422, 0.74396, 0.28297, 0.29379, 0.097062,
        0.81477, 0.45218, 0.45699, 0.92864, 0.97935, 0.97709, 0.90798, 0.27619,
        0.83268, 0.8632, 0.62564, 0.92787, 0.6914, 0.20573, 0.59241, 0.94209,
        0.97756, 0.77275, 0.85257, 0.28497;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.2, 0.20828, 0.18945, 0.23073, 0.23762, 0.22329,
        0.22172, 0.2188, 0.24482, 0.21173, 0.24003, 0.23475, 0.25326, 0.23467,
        0.26498, 0.26971, 0.27418, 0.27703, 0.27931, 0.27151, 0.25889, 0.24433,
        0.31736, 0.25709, 0.297, 0.25161, 0.28305, 0.2574, 0.25415, 0.22986,
        0.28942, 0.24787, 0.24097, 0.28473, 0.28845, 0.28651, 0.28492, 0.28647,
        0.30864, 0.3197, 0.31523, 0.34564, 0.34217, 0.32032, 0.34812, 0.38178,
        0.38851, 0.38671, 0.39216, 0.38032;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.3, 0.29516, 0.29915, 0.28218, 0.27772, 0.29035,
        0.28206, 0.27694, 0.24941, 0.27744, 0.25123, 0.25066, 0.23765, 0.2473,
        0.22325, 0.21821, 0.21372, 0.20882, 0.2019, 0.21826, 0.24167, 0.25753,
        0.22014, 0.24476, 0.22888, 0.24282, 0.22769, 0.23748, 0.23359, 0.24675,
        0.21826, 0.23415, 0.23301, 0.21474, 0.21269, 0.2122, 0.20933, 0.21804,
        0.18021, 0.16907, 0.16603, 0.14366, 0.14132, 0.16065, 0.13452, 0.11757,
        0.11447, 0.11228, 0.10811, 0.11457;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.5, 0.49656, 0.5114, 0.48708, 0.48466, 0.48636,
        0.49622, 0.50426, 0.50578, 0.51083, 0.50874, 0.51459, 0.50909, 0.51803,
        0.51177, 0.51208, 0.5121, 0.51415, 0.51879, 0.51023, 0.49945, 0.49815,
        0.4625, 0.49815, 0.47411, 0.50557, 0.48927, 0.50513, 0.51226, 0.52339,
        0.49232, 0.51797, 0.52602, 0.50053, 0.49886, 0.50129, 0.50576, 0.49549,
        0.51115, 0.51123, 0.51873, 0.5107, 0.51651, 0.51902, 0.51736, 0.50064,
        0.49702, 0.50101, 0.49972, 0.50511;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));

    Eigen::MatrixXd expected_b_inference1(D, T);
    expected_b_inference1 << 0.1, 0.099852, 0.10115, 0.10243, 0.10297, 0.10276,
        0.10079, 0.098923, 0.096421, 0.098911, 0.095795, 0.094436, 0.093543,
        0.095279, 0.091215, 0.09029, 0.089258, 0.088117, 0.086582, 0.090467,
        0.096016, 0.098649, 0.10725, 0.097664, 0.10219, 0.097998, 0.097784,
        0.10005, 0.098057, 0.10187, 0.1037, 0.099244, 0.09714, 0.099338,
        0.099139, 0.097953, 0.096231, 0.10124, 0.089965, 0.087547, 0.086412,
        0.082676, 0.081622, 0.086752, 0.07956, 0.076561, 0.076008, 0.075028,
        0.07345, 0.076373;
    Eigen::MatrixXd expected_b_inference2(D, T);
    expected_b_inference2 << 0.9, 0.90015, 0.89885, 0.89757, 0.89703, 0.89724,
        0.89921, 0.90108, 0.90358, 0.90109, 0.90421, 0.90556, 0.90646, 0.90472,
        0.90879, 0.90971, 0.91074, 0.91188, 0.91342, 0.90953, 0.90398, 0.90135,
        0.89275, 0.90234, 0.89781, 0.902, 0.90222, 0.89995, 0.90194, 0.89813,
        0.8963, 0.90076, 0.90286, 0.90066, 0.90086, 0.90205, 0.90377, 0.89876,
        0.91003, 0.91245, 0.91359, 0.91732, 0.91838, 0.91325, 0.92044, 0.92344,
        0.92399, 0.92497, 0.92655, 0.92363;
    Tensor3 expected_b_inference((vector<Eigen::MatrixXd>){
        expected_b_inference1, expected_b_inference2});
    Tensor3 estimated_b_inference =
        Tensor3(b_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_b_inference, expected_b_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(semi_markov_dbn3_particle_filter) {
    /**
     * This test case uses a semi-Markov DBN comprised of 2 single time
     * variables. One affects the state transition in a hidden markov chain. The
     * duration of a segment only depends on the state variable. The other
     * affects the former single time variable. It evaluates whether the
     * probabilities of X (state variable), A and B (single time variables) can
     * be approximated at every time step using particle filter.
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
        DynamicBayesNet::create_from_json("models/semi_markov_dbn3.json"));

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
    expected_x_inference1 << 0.3, 0.20914, 0.17724, 0.051294, 0.028136, 0.15013,
        0.19201, 0.21571, 0.13546, 0.20906, 0.1453, 0.17574, 0.089168, 0.20328,
        0.065913, 0.055219, 0.02463, 0.024324, 0.072543, 0.26666, 0.18824,
        0.11189, 0.087781, 0.16861, 0.079338, 0.17324, 0.10675, 0.20419,
        0.23045, 0.18171, 0.088104, 0.1763, 0.18221, 0.039257, 0.014679,
        0.017347, 0.059101, 0.24642, 0.10969, 0.086022, 0.19705, 0.05261,
        0.17646, 0.27826, 0.19054, 0.040353, 0.018234, 0.141, 0.10415, 0.30129;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.20733, 0.58563, 0.040455, 0.018279, 0.37082,
        0.41306, 0.42567, 0.1366, 0.52538, 0.17655, 0.21902, 0.066714, 0.45126,
        0.034175, 0.029599, 0.0057604, 0.0055473, 0.036442, 0.49761, 0.75957,
        0.85358, 0.078623, 0.31123, 0.070046, 0.38862, 0.11774, 0.51101, 0.4716,
        0.72185, 0.056366, 0.3238, 0.31719, 0.016859, 0.003389, 0.0040499,
        0.031154, 0.49408, 0.053918, 0.045901, 0.1741, 0.013972, 0.12981,
        0.52244, 0.21083, 0.012196, 0.0029388, 0.082334, 0.040337, 0.41227;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.58353, 0.23713, 0.90825, 0.95359, 0.47906,
        0.39493, 0.35861, 0.72794, 0.26556, 0.67816, 0.60524, 0.84412, 0.34546,
        0.89991, 0.91518, 0.96961, 0.97013, 0.89102, 0.23573, 0.052189,
        0.034527, 0.8336, 0.52015, 0.85062, 0.43814, 0.7755, 0.2848, 0.29795,
        0.096442, 0.85553, 0.4999, 0.5006, 0.94388, 0.98193, 0.9786, 0.90974,
        0.2595, 0.83639, 0.86808, 0.62885, 0.93342, 0.69373, 0.1993, 0.59862,
        0.94745, 0.97883, 0.77666, 0.85551, 0.28644;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.47, 0.45503, 0.49237, 0.42414, 0.41369, 0.43955,
        0.43163, 0.42658, 0.37996, 0.43409, 0.38309, 0.38438, 0.35433, 0.38813,
        0.32875, 0.31834, 0.30907, 0.30066, 0.28961, 0.31728, 0.35588, 0.38227,
        0.32897, 0.36967, 0.33629, 0.38148, 0.33725, 0.37804, 0.37344, 0.41389,
        0.35025, 0.38935, 0.39183, 0.34069, 0.33503, 0.333, 0.32731, 0.34492,
        0.27326, 0.25072, 0.24735, 0.19952, 0.19632, 0.23926, 0.18528, 0.14879,
        0.14277, 0.1389, 0.13088, 0.14549;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.39, 0.40187, 0.37119, 0.43196, 0.44275, 0.41268,
        0.4258, 0.43403, 0.49247, 0.4257, 0.48635, 0.4854, 0.51805, 0.48271,
        0.55055, 0.56329, 0.57473, 0.58639, 0.60273, 0.56262, 0.50665, 0.46889,
        0.54577, 0.48925, 0.52741, 0.47745, 0.52537, 0.48354, 0.49124, 0.44307,
        0.52106, 0.47197, 0.46856, 0.52432, 0.52953, 0.5306, 0.53791, 0.51511,
        0.61647, 0.64822, 0.65493, 0.72217, 0.72789, 0.66686, 0.74355, 0.7931,
        0.80101, 0.80703, 0.81808, 0.79822;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.14, 0.1431, 0.13645, 0.1439, 0.14357, 0.14777,
        0.14257, 0.13938, 0.12757, 0.14021, 0.13056, 0.13022, 0.12762, 0.12916,
        0.1207, 0.11837, 0.11619, 0.11295, 0.10766, 0.1201, 0.13748, 0.14884,
        0.12526, 0.14108, 0.1363, 0.14106, 0.13738, 0.13842, 0.13533, 0.14305,
        0.12869, 0.13868, 0.13962, 0.13498, 0.13545, 0.1364, 0.13478, 0.13997,
        0.11028, 0.10107, 0.097722, 0.078308, 0.075788, 0.093878, 0.071178,
        0.058114, 0.056221, 0.054063, 0.051043, 0.056288;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));

    Eigen::MatrixXd expected_b_inference1(D, T);
    expected_b_inference1 << 0.1, 0.10138, 0.098235, 0.10267, 0.10294, 0.10322,
        0.10204, 0.10132, 0.099612, 0.10129, 0.10034, 0.1002, 0.10051, 0.099777,
        0.099447, 0.099151, 0.09886, 0.098241, 0.097138, 0.099673, 0.10322,
        0.10549, 0.10072, 0.10375, 0.10356, 0.10334, 0.10383, 0.10272, 0.10201,
        0.10278, 0.10095, 0.1024, 0.10258, 0.10304, 0.10336, 0.1037, 0.10344,
        0.10429, 0.098433, 0.096627, 0.095805, 0.092009, 0.091413, 0.095006,
        0.090501, 0.088094, 0.08777, 0.087298, 0.086728, 0.087695;
    Eigen::MatrixXd expected_b_inference2(D, T);
    expected_b_inference2 << 0.9, 0.89862, 0.90176, 0.89733, 0.89706, 0.89678,
        0.89796, 0.89868, 0.90039, 0.89871, 0.89966, 0.8998, 0.89949, 0.90022,
        0.90055, 0.90085, 0.90114, 0.90176, 0.90286, 0.90033, 0.89678, 0.89451,
        0.89928, 0.89625, 0.89644, 0.89666, 0.89617, 0.89728, 0.89799, 0.89722,
        0.89905, 0.8976, 0.89742, 0.89696, 0.89664, 0.8963, 0.89656, 0.89571,
        0.90157, 0.90337, 0.90419, 0.90799, 0.90859, 0.90499, 0.9095, 0.91191,
        0.91223, 0.9127, 0.91327, 0.9123;
    Tensor3 expected_b_inference((vector<Eigen::MatrixXd>){
        expected_b_inference1, expected_b_inference2});
    Tensor3 estimated_b_inference =
        Tensor3(b_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_b_inference, expected_b_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(semi_markov_dbn3_1_particle_filter) {
    /**
     * This test case uses a semi-Markov DBN comprised of 2 single time
     * variables. One affects the state transition in a hidden markov chain. The
     * duration of a segment depends on the state variable and the single
     * node variable that affects the state transition. The other affects the
     * former single time variable. It evaluates whether the probabilities of X
     * (state variable), A and B (single time variables) can be approximated at
     * every time step using particle filter.
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
        DynamicBayesNet::create_from_json("models/semi_markov_dbn3_1.json"));

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
    expected_x_inference1 << 0.3, 0.25995, 0.26954, 0.10538, 0.049384, 0.17732,
        0.24113, 0.29431, 0.19689, 0.28065, 0.20176, 0.23155, 0.11773, 0.22588,
        0.075558, 0.053335, 0.021307, 0.01942, 0.057426, 0.25408, 0.22484,
        0.148, 0.12427, 0.19629, 0.10597, 0.18498, 0.12987, 0.235, 0.29441,
        0.26452, 0.14478, 0.27235, 0.29491, 0.077044, 0.017705, 0.012584,
        0.040851, 0.22292, 0.11033, 0.096561, 0.22785, 0.076068, 0.19937,
        0.28607, 0.26306, 0.070119, 0.020578, 0.11811, 0.096572, 0.30142;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.21355, 0.54134, 0.046015, 0.018407, 0.32413,
        0.35449, 0.36142, 0.12203, 0.47958, 0.17348, 0.2158, 0.068434, 0.42431,
        0.031999, 0.024913, 0.0046305, 0.0044541, 0.029701, 0.46511, 0.72303,
        0.82788, 0.11118, 0.39768, 0.10733, 0.4373, 0.13367, 0.48913, 0.43063,
        0.65089, 0.058161, 0.30429, 0.29149, 0.017716, 0.0028549, 0.0030099,
        0.022928, 0.44212, 0.043705, 0.042407, 0.17798, 0.016233, 0.15869,
        0.55362, 0.25689, 0.018145, 0.0036895, 0.087647, 0.038847, 0.38973;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.52649, 0.18912, 0.84861, 0.93221, 0.49856,
        0.40438, 0.34427, 0.68108, 0.23977, 0.62476, 0.55265, 0.81383, 0.34981,
        0.89244, 0.92175, 0.97406, 0.97613, 0.91287, 0.28081, 0.052137,
        0.024113, 0.76455, 0.40603, 0.7867, 0.37771, 0.73646, 0.27587, 0.27496,
        0.084586, 0.79706, 0.42336, 0.4136, 0.90524, 0.97944, 0.98441, 0.93622,
        0.33497, 0.84597, 0.86103, 0.59417, 0.9077, 0.64193, 0.16031, 0.48005,
        0.91174, 0.97573, 0.79425, 0.86458, 0.30885;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.47, 0.47182, 0.46145, 0.46279, 0.46027, 0.50495,
        0.48758, 0.46701, 0.41556, 0.45369, 0.40943, 0.40006, 0.37842, 0.41924,
        0.35121, 0.33372, 0.31154, 0.28998, 0.26926, 0.32522, 0.36901, 0.38248,
        0.4298, 0.40651, 0.42013, 0.43125, 0.40296, 0.44547, 0.4282, 0.46553,
        0.43188, 0.43185, 0.41232, 0.39882, 0.39103, 0.37155, 0.34872, 0.42728,
        0.30474, 0.27861, 0.27068, 0.24091, 0.23366, 0.25708, 0.22482, 0.21697,
        0.21414, 0.21041, 0.19608, 0.22808;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.39, 0.39237, 0.38535, 0.43273, 0.44796, 0.38915,
        0.39218, 0.39805, 0.44417, 0.404, 0.44398, 0.4504, 0.47112, 0.44358,
        0.49798, 0.50764, 0.5141, 0.5194, 0.52839, 0.50908, 0.4679, 0.43427,
        0.52219, 0.4723, 0.52157, 0.44092, 0.51291, 0.42853, 0.43043, 0.38595,
        0.44051, 0.40325, 0.39483, 0.44699, 0.46233, 0.4698, 0.47769, 0.45993,
        0.50608, 0.50863, 0.5333, 0.53202, 0.55051, 0.50074, 0.6022, 0.6944,
        0.72502, 0.72688, 0.76143, 0.70596;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.14, 0.13581, 0.1532, 0.10448, 0.091768, 0.10591,
        0.12024, 0.13494, 0.14027, 0.14231, 0.14659, 0.14954, 0.15046, 0.13718,
        0.1508, 0.15864, 0.17436, 0.19063, 0.20234, 0.1657, 0.16309, 0.18325,
        0.048005, 0.12118, 0.058297, 0.12784, 0.084129, 0.12599, 0.14137,
        0.14852, 0.1276, 0.1649, 0.19285, 0.15419, 0.14664, 0.15865, 0.17359,
        0.11279, 0.18918, 0.21276, 0.19602, 0.22706, 0.21583, 0.24218, 0.17299,
        0.088628, 0.060842, 0.062711, 0.042495, 0.065961;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));

    Eigen::MatrixXd expected_b_inference1(D, T);
    expected_b_inference1 << 0.1, 0.098763, 0.10399, 0.090294, 0.086819,
        0.089245, 0.093858, 0.098684, 0.10195, 0.10121, 0.10393, 0.10508,
        0.10608, 0.10095, 0.10711, 0.10991, 0.11507, 0.12037, 0.12437, 0.11218,
        0.10994, 0.11513, 0.075603, 0.09691, 0.078819, 0.097923, 0.086648,
        0.096918, 0.10182, 0.10254, 0.097836, 0.10829, 0.11679, 0.10642,
        0.10457, 0.10861, 0.11358, 0.093844, 0.11946, 0.12697, 0.12255, 0.13227,
        0.12937, 0.13595, 0.11767, 0.094301, 0.086612, 0.087264, 0.082092,
        0.087567;
    Eigen::MatrixXd expected_b_inference2(D, T);
    expected_b_inference2 << 0.9, 0.90124, 0.89601, 0.90971, 0.91318, 0.91075,
        0.90614, 0.90132, 0.89805, 0.89879, 0.89607, 0.89492, 0.89392, 0.89905,
        0.89289, 0.89009, 0.88493, 0.87963, 0.87563, 0.88782, 0.89006, 0.88487,
        0.9244, 0.90309, 0.92118, 0.90208, 0.91335, 0.90308, 0.89818, 0.89746,
        0.90216, 0.89171, 0.88321, 0.89358, 0.89543, 0.89139, 0.88642, 0.90616,
        0.88054, 0.87303, 0.87745, 0.86773, 0.87063, 0.86405, 0.88233, 0.9057,
        0.91339, 0.91274, 0.91791, 0.91243;
    Tensor3 expected_b_inference((vector<Eigen::MatrixXd>){
        expected_b_inference1, expected_b_inference2});
    Tensor3 estimated_b_inference =
        Tensor3(b_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_b_inference, expected_b_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(semi_markov_dbn4_particle_filter) {
    /**
     * This test case uses a semi-Markov DBN comprised of 2 single time
     * variables. Both affect the state transition in a hidden markov chain and
     * one is dependent of the other. The duration of a segment depends only on
     * the state variable. It evaluates whether the probabilities of X (state
     * variable), A and B (single time variables) can be approximated at every
     * time step using particle filter.
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
        DynamicBayesNet::create_from_json("models/semi_markov_dbn4.json"));

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
    expected_x_inference1 << 0.3, 0.20891, 0.19886, 0.060805, 0.033983, 0.17954,
        0.22674, 0.25307, 0.15246, 0.24875, 0.16454, 0.19869, 0.098254, 0.23145,
        0.071363, 0.058936, 0.025965, 0.025464, 0.075467, 0.28512, 0.21074,
        0.13497, 0.11502, 0.21595, 0.10045, 0.20732, 0.12126, 0.23562, 0.26544,
        0.22478, 0.10478, 0.21146, 0.21618, 0.044766, 0.015985, 0.018547,
        0.063216, 0.27595, 0.11245, 0.087409, 0.20336, 0.051633, 0.17386,
        0.28749, 0.1907, 0.039321, 0.01723, 0.13245, 0.097671, 0.28661;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.5, 0.19594, 0.54017, 0.033192, 0.015341, 0.33458,
        0.36797, 0.37268, 0.11271, 0.4622, 0.14575, 0.18278, 0.054764, 0.40596,
        0.028879, 0.025798, 0.0050851, 0.0050048, 0.033269, 0.46955, 0.73049,
        0.82764, 0.076217, 0.28527, 0.062363, 0.34789, 0.10079, 0.45875,
        0.41563, 0.66542, 0.048633, 0.2861, 0.27595, 0.014052, 0.0029503,
        0.0035744, 0.027458, 0.45174, 0.044236, 0.037197, 0.14626, 0.011127,
        0.11693, 0.49523, 0.18605, 0.010229, 0.0028661, 0.086149, 0.040521,
        0.41903;
    Eigen::MatrixXd expected_x_inference3(D, T);
    expected_x_inference3 << 0.2, 0.59516, 0.26097, 0.906, 0.95068, 0.48588,
        0.40529, 0.37425, 0.73484, 0.28904, 0.68971, 0.61853, 0.84698, 0.36258,
        0.89976, 0.91527, 0.96895, 0.96953, 0.89126, 0.24533, 0.058776, 0.03739,
        0.80876, 0.49878, 0.83719, 0.44478, 0.77795, 0.30562, 0.31893, 0.10979,
        0.84659, 0.50244, 0.50786, 0.94118, 0.98106, 0.97788, 0.90933, 0.27231,
        0.84331, 0.87539, 0.65038, 0.93724, 0.70921, 0.21728, 0.62325, 0.95045,
        0.9799, 0.7814, 0.86181, 0.29437;
    Tensor3 expected_x_inference(
        {expected_x_inference1, expected_x_inference2, expected_x_inference3});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.47, 0.48499, 0.4534, 0.52281, 0.53423, 0.50807,
        0.51106, 0.51037, 0.5615, 0.49993, 0.55439, 0.54826, 0.57957, 0.54994,
        0.60272, 0.61145, 0.61943, 0.62595, 0.63333, 0.61327, 0.58293, 0.55526,
        0.65425, 0.57825, 0.63019, 0.57266, 0.61747, 0.58245, 0.5815, 0.54204,
        0.62894, 0.57302, 0.56556, 0.62714, 0.63234, 0.63088, 0.63148, 0.62639,
        0.67561, 0.69379, 0.69229, 0.73419, 0.73345, 0.70062, 0.74321, 0.77947,
        0.78606, 0.78675, 0.79339, 0.78091;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.39, 0.37801, 0.39951, 0.3469, 0.33738, 0.36029,
        0.35443, 0.35245, 0.30691, 0.35905, 0.31222, 0.31579, 0.29024, 0.31259,
        0.26918, 0.26152, 0.25468, 0.24842, 0.24053, 0.2605, 0.28998, 0.3144,
        0.23568, 0.29403, 0.2547, 0.29601, 0.26197, 0.28684, 0.28554, 0.31495,
        0.24897, 0.29031, 0.29427, 0.24867, 0.24467, 0.24539, 0.2437, 0.25058,
        0.20422, 0.18868, 0.18762, 0.1542, 0.15313, 0.18003, 0.14462, 0.11858,
        0.11396, 0.11232, 0.10715, 0.11591;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.14, 0.137, 0.14709, 0.1303, 0.12839, 0.13164,
        0.13451, 0.13717, 0.1316, 0.14103, 0.13338, 0.13595, 0.13019, 0.13747,
        0.1281, 0.12703, 0.12589, 0.12563, 0.12614, 0.12622, 0.12709, 0.13034,
        0.11007, 0.12772, 0.11511, 0.13133, 0.12056, 0.13071, 0.13297, 0.14301,
        0.12209, 0.13667, 0.14017, 0.12419, 0.123, 0.12373, 0.12483, 0.12303,
        0.12016, 0.11753, 0.12009, 0.11161, 0.11342, 0.11935, 0.11217, 0.10195,
        0.099985, 0.10093, 0.099461, 0.10318;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));

    Eigen::MatrixXd expected_b_inference1(D, T);
    expected_b_inference1 << 0.1, 0.098499, 0.1038, 0.097447, 0.096955,
        0.098508, 0.097854, 0.097233, 0.092164, 0.098759, 0.092353, 0.09219,
        0.088871, 0.093385, 0.085796, 0.084514, 0.083147, 0.082013, 0.080789,
        0.084344, 0.08969, 0.093474, 0.090895, 0.091287, 0.089355, 0.092905,
        0.088065, 0.094217, 0.093399, 0.10098, 0.093129, 0.095802, 0.09535,
        0.090366, 0.089665, 0.08899, 0.087976, 0.091427, 0.080714, 0.077674,
        0.07761, 0.071441, 0.071185, 0.077534, 0.069039, 0.063282, 0.062219,
        0.061752, 0.060076, 0.063481;
    Eigen::MatrixXd expected_b_inference2(D, T);
    expected_b_inference2 << 0.9, 0.9015, 0.8962, 0.90255, 0.90304, 0.90149,
        0.90215, 0.90277, 0.90784, 0.90124, 0.90765, 0.90781, 0.91113, 0.90661,
        0.9142, 0.91549, 0.91685, 0.91799, 0.91921, 0.91566, 0.91031, 0.90653,
        0.9091, 0.90871, 0.91065, 0.9071, 0.91193, 0.90578, 0.9066, 0.89902,
        0.90687, 0.9042, 0.90465, 0.90963, 0.91034, 0.91101, 0.91202, 0.90857,
        0.91929, 0.92233, 0.92239, 0.92856, 0.92881, 0.92247, 0.93096, 0.93672,
        0.93778, 0.93825, 0.93992, 0.93652;
    Tensor3 expected_b_inference((vector<Eigen::MatrixXd>){
        expected_b_inference1, expected_b_inference2});
    Tensor3 estimated_b_inference =
        Tensor3(b_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_b_inference, expected_b_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(semi_markov_dbn5_particle_filter) {
    /**
     * This test case uses a DBN comprised of a pair of semi-coupled hidden
     * chains. One chain affects the next state of the other. The uppe-level
     * chain follows a semi-Markov process. The duration only depends on the
     * state of that chain. It evaluates whether the probabilities of X and Y
     * (state variables from both chains) can be approximated at every time step
     * using particle filter.
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
        DynamicBayesNet::create_from_json("models/semi_markov_dbn5.json"));

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
    expected_x_inference1 << 0.3, 0.84453, 0.31307, 0.8557, 0.58697, 0.42344,
        0.5234, 0.47788, 0.7991, 0.30166, 0.85443, 0.25428, 0.86514, 0.27564,
        0.86544, 0.63096, 0.74062, 0.70454, 0.71339, 0.34583, 0.57221, 0.41537,
        0.81184, 0.34721, 0.83788, 0.26947, 0.86441, 0.25088, 0.63919, 0.42725,
        0.81199, 0.33335, 0.58869, 0.75304, 0.70367, 0.732, 0.70014, 0.34617,
        0.83791, 0.62238, 0.43085, 0.81612, 0.34981, 0.57605, 0.73745, 0.64604,
        0.74067, 0.39803, 0.82483, 0.27428;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.7, 0.15547, 0.68693, 0.1443, 0.41303, 0.57656,
        0.4766, 0.52212, 0.2009, 0.69834, 0.14557, 0.74572, 0.13486, 0.72436,
        0.13456, 0.36904, 0.25938, 0.29546, 0.28661, 0.65417, 0.42779, 0.58463,
        0.18816, 0.65279, 0.16212, 0.73053, 0.13559, 0.74912, 0.36081, 0.57275,
        0.18801, 0.66665, 0.41131, 0.24696, 0.29633, 0.268, 0.29986, 0.65383,
        0.16209, 0.37762, 0.56915, 0.18388, 0.65019, 0.42395, 0.26255, 0.35396,
        0.25933, 0.60197, 0.17517, 0.72572;
    Tensor3 expected_x_inference((vector<Eigen::MatrixXd>){
        expected_x_inference1, expected_x_inference2});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_y_inference1(D, T);
    expected_y_inference1 << 0.8, 0.48459, 0.22489, 0.3369, 0.3108, 0.27656,
        0.47869, 0.55287, 0.34082, 0.23839, 0.2473, 0.48046, 0.38044, 0.30901,
        0.48361, 0.36763, 0.46774, 0.50402, 0.3019, 0.21799, 0.24179, 0.29388,
        0.5382, 0.38519, 0.29813, 0.26809, 0.28995, 0.31617, 0.52866, 0.37125,
        0.47972, 0.30849, 0.43255, 0.50788, 0.52212, 0.48445, 0.2701, 0.20141,
        0.41824, 0.37285, 0.47665, 0.51141, 0.48936, 0.28132, 0.22793, 0.44611,
        0.54908, 0.5211, 0.29321, 0.22099;
    Eigen::MatrixXd expected_y_inference2(D, T);
    expected_y_inference2 << 0.2, 0.51541, 0.77511, 0.6631, 0.6892, 0.72344,
        0.52131, 0.44713, 0.65918, 0.76161, 0.7527, 0.51954, 0.61956, 0.69099,
        0.51639, 0.63237, 0.53226, 0.49598, 0.6981, 0.78201, 0.75821, 0.70612,
        0.4618, 0.61481, 0.70187, 0.73191, 0.71005, 0.68383, 0.47134, 0.62875,
        0.52028, 0.69151, 0.56745, 0.49212, 0.47788, 0.51555, 0.7299, 0.79859,
        0.58176, 0.62715, 0.52335, 0.48859, 0.51064, 0.71868, 0.77207, 0.55389,
        0.45092, 0.4789, 0.70679, 0.77901;
    Tensor3 expected_y_inference((vector<Eigen::MatrixXd>){
        expected_y_inference1, expected_y_inference2});
    Tensor3 estimated_y_inference =
        Tensor3(y_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_y_inference, expected_y_inference, tolerance));
}

BOOST_AUTO_TEST_CASE(semi_markov_dbn6_particle_filter) {
    /**
     * This test case uses a decouples DBN 6 and make both chains semi-markov.
     * The single time node affects the state transition of both chains and the
     * duration of segments. States of the chain also affects the segment
     * duration.It evaluates whether the probabilities of X and Y (state
     * variables from both chains) nd A (single time node) can be approximated
     * at every time step using particle filter.
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
        DynamicBayesNet::create_from_json("models/semi_markov_dbn6.json"));

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
    expected_x_inference1 << 0.3, 0.5172, 0.33648, 0.50938, 0.62017, 0.33278,
        0.19435, 0.15093, 0.42009, 0.32863, 0.59576, 0.41669, 0.61784, 0.38873,
        0.56135, 0.65763, 0.68658, 0.67642, 0.6482, 0.29229, 0.13861, 0.097792,
        0.33596, 0.28888, 0.59308, 0.45079, 0.66524, 0.43746, 0.2803, 0.19639,
        0.44076, 0.32334, 0.25959, 0.54704, 0.72689, 0.78124, 0.76646, 0.40204,
        0.48125, 0.53082, 0.25148, 0.3933, 0.2389, 0.18324, 0.46191, 0.69249,
        0.78776, 0.49431, 0.60112, 0.31688;
    Eigen::MatrixXd expected_x_inference2(D, T);
    expected_x_inference2 << 0.7, 0.4828, 0.66352, 0.49062, 0.37983, 0.66722,
        0.80565, 0.84907, 0.57991, 0.67137, 0.40424, 0.58331, 0.38216, 0.61127,
        0.43865, 0.34237, 0.31342, 0.32358, 0.3518, 0.70771, 0.86139, 0.90221,
        0.66404, 0.71112, 0.40692, 0.54921, 0.33476, 0.56254, 0.7197, 0.80361,
        0.55924, 0.67666, 0.74041, 0.45296, 0.27311, 0.21876, 0.23354, 0.59796,
        0.51875, 0.46918, 0.74852, 0.6067, 0.7611, 0.81676, 0.53809, 0.30751,
        0.21224, 0.50569, 0.39888, 0.68312;
    Tensor3 expected_x_inference((vector<Eigen::MatrixXd>){
        expected_x_inference1, expected_x_inference2});
    Tensor3 estimated_x_inference =
        Tensor3(x_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_x_inference, expected_x_inference, tolerance));

    Eigen::MatrixXd expected_y_inference1(D, T);
    expected_y_inference1 << 0.8, 0.7096, 0.60218, 0.70283, 0.55005, 0.42068,
        0.53159, 0.5922, 0.41743, 0.32554, 0.29541, 0.49092, 0.42529, 0.39098,
        0.58753, 0.50988, 0.65033, 0.71393, 0.5397, 0.41432, 0.3392, 0.30365,
        0.48804, 0.41672, 0.38161, 0.37769, 0.38923, 0.40381, 0.61668, 0.53977,
        0.67242, 0.5449, 0.64306, 0.68125, 0.67964, 0.66041, 0.43742, 0.3062,
        0.42598, 0.3346, 0.49105, 0.6167, 0.69317, 0.5412, 0.43228, 0.56251,
        0.63913, 0.67038, 0.48209, 0.36241;
    Eigen::MatrixXd expected_y_inference2(D, T);
    expected_y_inference2 << 0.2, 0.2904, 0.39782, 0.29717, 0.44995, 0.57932,
        0.46841, 0.4078, 0.58257, 0.67446, 0.70459, 0.50908, 0.57471, 0.60902,
        0.41247, 0.49012, 0.34967, 0.28607, 0.4603, 0.58568, 0.6608, 0.69635,
        0.51196, 0.58328, 0.61839, 0.62231, 0.61077, 0.59619, 0.38332, 0.46023,
        0.32758, 0.4551, 0.35694, 0.31875, 0.32036, 0.33959, 0.56258, 0.6938,
        0.57402, 0.6654, 0.50895, 0.3833, 0.30683, 0.4588, 0.56772, 0.43749,
        0.36087, 0.32962, 0.51791, 0.63759;
    Tensor3 expected_y_inference((vector<Eigen::MatrixXd>){
        expected_y_inference1, expected_y_inference2});
    Tensor3 estimated_y_inference =
        Tensor3(y_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_y_inference, expected_y_inference, tolerance));

    Eigen::MatrixXd expected_a_inference1(D, T);
    expected_a_inference1 << 0.2, 0.18866, 0.20389, 0.14861, 0.13762, 0.14082,
        0.14393, 0.14487, 0.15598, 0.15685, 0.14527, 0.16138, 0.13583, 0.14212,
        0.12081, 0.111, 0.089611, 0.068217, 0.058873, 0.061707, 0.062126,
        0.060573, 0.077738, 0.076482, 0.073617, 0.077277, 0.061152, 0.064683,
        0.065043, 0.066604, 0.062938, 0.064323, 0.0585, 0.0504, 0.039416,
        0.027929, 0.021256, 0.022566, 0.021322, 0.020011, 0.02113, 0.021963,
        0.019509, 0.020065, 0.020623, 0.01835, 0.014344, 0.014215, 0.011189,
        0.011721;
    Eigen::MatrixXd expected_a_inference2(D, T);
    expected_a_inference2 << 0.3, 0.29943, 0.30058, 0.2744, 0.26188, 0.26785,
        0.25244, 0.25013, 0.25914, 0.25383, 0.25816, 0.27202, 0.25862, 0.26168,
        0.24252, 0.22947, 0.20958, 0.19125, 0.18648, 0.18991, 0.1916, 0.19087,
        0.21195, 0.20642, 0.21279, 0.21237, 0.19845, 0.20372, 0.20172, 0.20549,
        0.19508, 0.19584, 0.18378, 0.18124, 0.17359, 0.15835, 0.14296, 0.14741,
        0.13454, 0.12254, 0.13269, 0.14826, 0.14786, 0.14709, 0.15716, 0.14544,
        0.12738, 0.12688, 0.10968, 0.11264;
    Eigen::MatrixXd expected_a_inference3(D, T);
    expected_a_inference3 << 0.5, 0.51191, 0.49553, 0.577, 0.60051, 0.59133,
        0.60363, 0.605, 0.58488, 0.58932, 0.59657, 0.5666, 0.60555, 0.5962,
        0.63667, 0.65952, 0.70081, 0.74054, 0.75465, 0.74838, 0.74628, 0.74856,
        0.71031, 0.7171, 0.7136, 0.71035, 0.7404, 0.73159, 0.73323, 0.72791,
        0.74198, 0.73983, 0.75772, 0.76836, 0.78699, 0.81373, 0.83579, 0.83002,
        0.84414, 0.85745, 0.84618, 0.82978, 0.83263, 0.83285, 0.82222, 0.83621,
        0.85827, 0.85891, 0.87913, 0.87564;
    Tensor3 expected_a_inference(
        {expected_a_inference1, expected_a_inference2, expected_a_inference3});
    Tensor3 estimated_a_inference =
        Tensor3(a_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_a_inference, expected_a_inference, tolerance));
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