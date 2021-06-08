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
#include "pgm/EvidenceSet.h"
#include "pgm/inference/MarginalizationFactorNode.h"
#include "pgm/inference/ParticleFilter.h"
#include "pgm/inference/SegmentExpansionFactorNode.h"
#include "pgm/inference/SegmentMarginalizationFactorNode.h"
#include "pgm/inference/SegmentTransitionFactorNode.h"
#include "pgm/inference/VariableNode.h"
#include "pipeline/estimation/CompoundSamplerEstimator.h"
#include "pipeline/estimation/ParticleFilterEstimator.h"
#include "pipeline/estimation/SamplerEstimator.h"
#include "pipeline/estimation/SumProductEstimator.h"
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

BOOST_AUTO_TEST_CASE(truncated) {
    /**
     * This test case checks if data can be generated correctly up to a time
     * t smaller than the final time T that the DBN was unrolled into. A
     * deterministic model is used.
     */

    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/deterministic_dbn.json"));

    model->unroll(10, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    int time_steps = 4;
    AncestralSampler sampler(model);
    sampler.set_num_in_plate_samples(1);
    sampler.set_max_time_step_to_sample(time_steps - 1);
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
     * Test exact inference with sum product in an HMM model.
     */

    // Data
    Eigen::MatrixXd obs1(2, 4);
    obs1 << NO_OBS, 1, 1, 0, NO_OBS, 0, 0, 1;
    Eigen::MatrixXd obs2(2, 4);
    obs2 << NO_OBS, 0, 1, 0, NO_OBS, 0, 1, 1;

    EvidenceSet data;
    data.add_data("Obs1", Tensor3(obs1));
    data.add_data("Obs2", Tensor3(obs2));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/hmm.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Inference
    SumProductEstimator state_estimator(model, 0, "State");
    state_estimator.set_subgraph_window_size(1);
    state_estimator.set_show_progress(false);
    state_estimator.prepare();
    state_estimator.estimate(data);

    Eigen::MatrixXd expected_state_inference1(2, 4);
    expected_state_inference1 << 0.300000000000000, 0.530573248407643,
        0.371783439490446, 0.350923566878981, 0.300000000000000,
        0.340000000000000, 0.118493150684932, 0.316095890410959;
    Eigen::MatrixXd expected_state_inference2(2, 4);
    expected_state_inference2 << 0.500000000000000, 0.126114649681529,
        0.393885350318471, 0.425643312101911, 0.500000000000000,
        0.440000000000000, 0.805479452054795, 0.476301369863014;
    Eigen::MatrixXd expected_state_inference3(2, 4);
    expected_state_inference3 << 0.200000000000000, 0.343312101910828,
        0.234331210191083, 0.223433121019108, 0.200000000000000,
        0.220000000000000, 0.076027397260274, 0.207602739726027;
    Tensor3 expected_state_inference({expected_state_inference1,
                                      expected_state_inference2,
                                      expected_state_inference3});
    Tensor3 estimated_state_inference =
        Tensor3(state_estimator.get_estimates().estimates);
    BOOST_TEST(
        check_tensor_eq(estimated_state_inference, expected_state_inference));

    // Prediction
    SumProductEstimator obs_predictor(
        model, 3, "Obs1", VectorXd::Constant(1, 0));
    obs_predictor.set_subgraph_window_size(1);
    obs_predictor.set_show_progress(false);
    obs_predictor.prepare();
    obs_predictor.estimate(data);

    Eigen::MatrixXd expected_obs_prediction1(2, 4);
    expected_obs_prediction1 << 0.848873600000000, 0.842869304458599,
        0.846963411464968, 0.847514837350319, 0.848873600000000,
        0.847812640000000, 0.853555808219178, 0.848444902465753;
    Tensor3 expected_obs_prediction(expected_obs_prediction1);
    Tensor3 estimated_obs_prediction =
        Tensor3(obs_predictor.get_estimates().estimates);
    BOOST_TEST(
        check_tensor_eq(estimated_obs_prediction, expected_obs_prediction));
}

BOOST_AUTO_TEST_CASE(hmm_particle_filter) {
    /**
     * Test particle filter estimation in an HMM model.
     */

    // Data
    Eigen::MatrixXd obs1(2, 4);
    obs1 << NO_OBS, 1, 1, 0, NO_OBS, 0, 0, 1;
    Eigen::MatrixXd obs2(2, 4);
    obs2 << NO_OBS, 0, 1, 0, NO_OBS, 0, 1, 1;

    EvidenceSet data;
    data.add_data("Obs1", Tensor3(obs1));
    data.add_data("Obs2", Tensor3(obs2));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/hmm.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Inference
    auto state_estimator = make_shared<SamplerEstimator>(model, 0, "State");
    auto obs_predictor = make_shared<SamplerEstimator>(
        model, 3, "Obs1", VectorXd::Constant(1, 0));

    ParticleFilterEstimator particle_estimator(model, 1000, gen, 1);
    particle_estimator.add_base_estimator(state_estimator);
    particle_estimator.add_base_estimator(obs_predictor);
    particle_estimator.set_show_progress(false);

    particle_estimator.prepare();
    particle_estimator.estimate(data);

    double tolerance = 0.1;
    Eigen::MatrixXd expected_state_inference1(2, 4);
    expected_state_inference1 << 0.300000000000000, 0.530573248407643,
        0.371783439490446, 0.350923566878981, 0.300000000000000,
        0.340000000000000, 0.118493150684932, 0.316095890410959;
    Eigen::MatrixXd expected_state_inference2(2, 4);
    expected_state_inference2 << 0.500000000000000, 0.126114649681529,
        0.393885350318471, 0.425643312101911, 0.500000000000000,
        0.440000000000000, 0.805479452054795, 0.476301369863014;
    Eigen::MatrixXd expected_state_inference3(2, 4);
    expected_state_inference3 << 0.200000000000000, 0.343312101910828,
        0.234331210191083, 0.223433121019108, 0.200000000000000,
        0.220000000000000, 0.076027397260274, 0.207602739726027;
    Tensor3 expected_state_inference({expected_state_inference1,
                                      expected_state_inference2,
                                      expected_state_inference3});
    Tensor3 estimated_state_inference =
        Tensor3(state_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_state_inference, expected_state_inference, tolerance));

    // Prediction
    Eigen::MatrixXd expected_obs_prediction1(2, 4);
    expected_obs_prediction1 << 0.848873600000000, 0.842869304458599,
        0.846963411464968, 0.847514837350319, 0.848873600000000,
        0.847812640000000, 0.853555808219178, 0.848444902465753;
    Tensor3 expected_obs_prediction(expected_obs_prediction1);
    Tensor3 estimated_obs_prediction =
        Tensor3(obs_predictor->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(
        estimated_obs_prediction, expected_obs_prediction, tolerance));
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

    BOOST_TEST(check_tensor_eq(state_estimates, expected_state_estimates, tolerance));

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