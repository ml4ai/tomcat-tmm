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
    return estimated.equals(expected, tolerance);
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

BOOST_AUTO_TEST_CASE(complete) {
    /**
     * This test case checks if data can be generated correctly, following
     * the distributions defined in the model. A deterministic model is used
     * so that the values generated can be known in advance.
     */

    fs::current_path("../../test");
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

    fs::current_path("../../test");
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

    fs::current_path("../../test");
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

    fs::current_path("../../test");
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

    fs::current_path("../../test");
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

BOOST_AUTO_TEST_CASE(dbn_training) {
    /**
     * This test case checks if the model can learn the parameters of a
     * non-deterministic model, given data generated from such a model.
     * Observations for node TC are not provided to the sampler to capture
     * the ability of the procedure to learn the parameters given that some
     * nodes are hidden.
     */

    fs::current_path("../../test");
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
    const shared_ptr<RandomVariableNode>& theta_fixed =
        dynamic_pointer_cast<RandomVariableNode>(
            model->get_nodes_by_label("ThetaFixed_0")[0]);
    Eigen::MatrixXd fixed_prior(1, 3);
    fixed_prior << 0.5, 0.3, 0.2;
    theta_fixed->set_assignment(fixed_prior);
    theta_fixed->freeze();

    Eigen::MatrixXd theta_state_given_others(18, 3);
    theta_state_given_others << 0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.3, 0.5, 0.2,
        0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.3, 0.2,
        0.5, 0.3, 0.2, 0.5, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.3,
        0.2, 0.5, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.3, 0.5, 0.2, 0.3, 0.5, 0.2,
        0.3, 0.2, 0.5;
    for (int i = 0; i < 18; i = i + 2) {
        stringstream label;
        label << "ThetaState.State.Fixed.Movable_" << i;
        const shared_ptr<RandomVariableNode>& theta_state =
            dynamic_pointer_cast<RandomVariableNode>(
                model->get_nodes_by_label(label.str())[0]);
        theta_state->set_assignment(theta_state_given_others.row(i));
        theta_state->freeze();
    }

    trainer.prepare();
    trainer.fit(data);

    // Check trained values
    Eigen::MatrixXd movable_prior(1, 2);
    movable_prior << 0.3, 0.7;
    MatrixXd estimated_pi_pbae =
        model->get_nodes_by_label("PiMovable_0")[0]->get_assignment();
    auto check = check_matrix_eq(estimated_pi_pbae, movable_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    Eigen::MatrixXd state_prior(1, 3);
    state_prior << 0.3, 0.5, 0.2;
    MatrixXd estimated_theta_state =
        model->get_nodes_by_label("ThetaState_0")[0]->get_assignment();
    check = check_matrix_eq(estimated_theta_state, state_prior, tolerance);
    BOOST_TEST(check.first, check.second);

    Eigen::MatrixXd pi_movable_given_movable(2, 2);
    pi_movable_given_movable << 0.3, 0.7, 0.7, 0.3;
    for (int i = 0; i < 2; i++) {
        stringstream label;
        label << "PiMovable.Movable_" << i;
        MatrixXd estimated_pi_movable_given_movable =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_pi_movable_given_movable,
                                pi_movable_given_movable.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    for (int i = 0; i < 18; i++) {
        stringstream label;
        label << "ThetaState.State.Fixed.Movable_" << i;
        MatrixXd estimated_theta_state_given_others =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_theta_state_given_others,
                                theta_state_given_others.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    Eigen::MatrixXd pi_obs1_given_state(3, 2);
    pi_obs1_given_state << 0.3, 0.7, 0.7, 0.3, 0.3, 0.7;
    for (int i = 0; i < 3; i++) {
        stringstream label;
        label << "PiObs1.State_" << i;
        MatrixXd estimated_pi_obs1_given_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_pi_obs1_given_state,
                                pi_obs1_given_state.row(i),
                                tolerance);
        BOOST_TEST(check.first, check.second);
    }

    Eigen::MatrixXd pi_obs2_given_state(3, 2);
    pi_obs2_given_state << 0.7, 0.3, 0.3, 0.7, 0.7, 0.3;
    for (int i = 0; i < 3; i++) {
        stringstream label;
        label << "PiObs2.State_" << i;
        MatrixXd estimated_pi_obs2_given_state =
            model->get_nodes_by_label(label.str())[0]->get_assignment();
        check = check_matrix_eq(estimated_pi_obs2_given_state,
                                pi_obs2_given_state.row(i),
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

BOOST_AUTO_TEST_CASE(sum_product_small) {
    /**
     * Test exact inference with sum product in a small DBN with an state, an
     * observation and a non repeatable node linked to the states.
     */

    fs::current_path("../../test");

    // Data
    Eigen::MatrixXd data_matrix(1, 10);
    data_matrix << NO_OBS, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    EvidenceSet data;
    data.add_data("Obs", Tensor3(data_matrix));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn_small.json"));

    model->unroll(data.get_time_steps(), true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    SumProductEstimator obs_estimator_h1(
        model, 1, "Obs", VectorXd::Constant(1, 1));
    SumProductEstimator obs_estimator_h3(
        model, 3, "Obs", VectorXd::Constant(1, 1));
    SumProductEstimator fixed_estimator(model, 0, "Fixed");
    SumProductEstimator state_estimator(model, 0, "State");

    //    obs_estimator_h1.set_variable_window(true);
    //    obs_estimator_h3.set_variable_window(true);
    fixed_estimator.set_variable_window(true);
    state_estimator.set_variable_window(true);

    //    obs_estimator_h1.set_show_progress(false);
    //    obs_estimator_h1.prepare();
    //    obs_estimator_h1.estimate(data);
    //    obs_estimator_h3.set_show_progress(false);
    //    obs_estimator_h3.prepare();
    //    obs_estimator_h3.estimate(data);
    fixed_estimator.set_show_progress(false);
    fixed_estimator.prepare();
    fixed_estimator.estimate(data);
    state_estimator.set_show_progress(false);
    state_estimator.prepare();
    state_estimator.estimate(data);

    //    cout << Tensor3(obs_estimator_h1.get_estimates().estimates) << endl;
    //    cout << Tensor3(obs_estimator_h3.get_estimates().estimates) << endl;
    cout << Tensor3(fixed_estimator.get_estimates().estimates) << endl;
    cout << Tensor3(state_estimator.get_estimates().estimates) << endl;

    //    MatrixXd obs1_estimates_h1 =
    //    obs1_estimator_h1.get_estimates().estimates[0]; MatrixXd
    //    obs2_estimates_h1 = obs2_estimator_h1.get_estimates().estimates[0];
    //    MatrixXd obs1_estimates_h3 =
    //    obs1_estimator_h3.get_estimates().estimates[0]; MatrixXd
    //    obs2_estimates_h3 = obs2_estimator_h3.get_estimates().estimates[0];
    //    vector<MatrixXd> fixed_estimates =
    //        fixed_estimator.get_estimates().estimates;
    //
    //    MatrixXd expected_obs1_h1(1, 4);
    //    expected_obs1_h1 << 0.56788, 0.56589, 0.566867, 0.565993;
    //    auto check = check_matrix_eq(obs1_estimates_h1, expected_obs1_h1);
    //    BOOST_TEST(check.first, check.second);
    //
    //    MatrixXd expected_obs2_h1(1, 4);
    //    expected_obs2_h1 << 0.43212, 0.43411, 0.433133, 0.434007;
    //    check = check_matrix_eq(obs2_estimates_h1, expected_obs2_h1);
    //    BOOST_TEST(check.first, check.second);
    //
    //    MatrixXd expected_obs1_h3(1, 4);
    //    expected_obs1_h3 << 0.91875, 0.918454, 0.918545, 0.918483;
    //    check = check_matrix_eq(obs1_estimates_h3, expected_obs1_h3);
    //    BOOST_TEST(check.first, check.second);
    //
    //    MatrixXd expected_obs2_h3(1, 4);
    //    expected_obs2_h3 << 0.818174, 0.818918, 0.818499, 0.818842;
    //    check = check_matrix_eq(obs2_estimates_h3, expected_obs2_h3);
    //    BOOST_TEST(check.first, check.second);
    //
    //    MatrixXd expected_fixed(3, 4);
    //    expected_fixed << -1, 0.500324, 0.506335, 0.504282, -1, 0.294363,
    //    0.286289,
    //        0.291487, -1, 0.205313, 0.207376, 0.204232;
    //    check = check_matrix_eq(fixed_estimates[0], expected_fixed.row(0));
    //    BOOST_TEST(check.first, check.second);
    //    check = check_matrix_eq(fixed_estimates[1], expected_fixed.row(1));
    //    BOOST_TEST(check.first, check.second);
    //    check = check_matrix_eq(fixed_estimates[2], expected_fixed.row(2));
    //    BOOST_TEST(check.first, check.second);
}

BOOST_FIXTURE_TEST_CASE(sum_product, HMM) {
    /**
     * This test case checks if the sum-product procedure can estimate
     * correctly the marginal probabilities over time of the nodes Green and
     * TC from a non-deterministic model.
     */

    fs::current_path("../../test");
    DBNPtr deterministic_model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn_deterministic.json"));

    //    DBNPtr deterministic_model = create_model(true, false);
    deterministic_model->unroll(4, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    //    DBNPtr pre_trained_model = create_model(false, false);
    DBNPtr pre_trained_model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn.json"));

    pre_trained_model->unroll(4, true);
    SumProductEstimator obs1_estimator_h1(
        pre_trained_model, 1, "Obs1", VectorXd::Constant(1, 1));
    SumProductEstimator obs2_estimator_h1(
        pre_trained_model, 1, "Obs2", VectorXd::Constant(1, 1));
    SumProductEstimator obs1_estimator_h3(
        pre_trained_model, 3, "Obs1", VectorXd::Constant(1, 1));
    SumProductEstimator obs2_estimator_h3(
        pre_trained_model, 3, "Obs2", VectorXd::Constant(1, 1));
    SumProductEstimator fixed_estimator(pre_trained_model, 0, "Fixed");

    SumProductEstimator state_estimator(pre_trained_model, 0, "State");

    fixed_estimator.set_variable_window(true);
    state_estimator.set_variable_window(true);

    // Green node is observed
    // One sample from the deterministic model will be used for estimation.
    AncestralSampler sampler(deterministic_model);
    sampler.sample(gen, 1);
    EvidenceSet data;
    data.add_data("Obs1", sampler.get_samples("Obs1"));
    cout << data << endl;

    obs1_estimator_h1.set_show_progress(false);
    obs1_estimator_h1.prepare();
    obs1_estimator_h1.estimate(data);
    obs2_estimator_h1.set_show_progress(false);
    obs2_estimator_h1.prepare();
    obs2_estimator_h1.estimate(data);
    obs1_estimator_h3.set_show_progress(false);
    obs1_estimator_h3.prepare();
    obs1_estimator_h3.estimate(data);
    obs2_estimator_h3.set_show_progress(false);
    obs2_estimator_h3.prepare();
    obs2_estimator_h3.estimate(data);
    fixed_estimator.set_show_progress(false);
    fixed_estimator.prepare();
    fixed_estimator.estimate(data);

    state_estimator.set_show_progress(false);
    state_estimator.prepare();
    state_estimator.estimate(data);

    cout << Tensor3(state_estimator.get_estimates().estimates) << endl;

    MatrixXd obs1_estimates_h1 = obs1_estimator_h1.get_estimates().estimates[0];
    MatrixXd obs2_estimates_h1 = obs2_estimator_h1.get_estimates().estimates[0];
    MatrixXd obs1_estimates_h3 = obs1_estimator_h3.get_estimates().estimates[0];
    MatrixXd obs2_estimates_h3 = obs2_estimator_h3.get_estimates().estimates[0];
    vector<MatrixXd> fixed_estimates =
        fixed_estimator.get_estimates().estimates;

    MatrixXd expected_obs1_h1(1, 4);
    expected_obs1_h1 << 0.56788, 0.56589, 0.566867, 0.565993;
    auto check = check_matrix_eq(obs1_estimates_h1, expected_obs1_h1);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_obs2_h1(1, 4);
    expected_obs2_h1 << 0.43212, 0.43411, 0.433133, 0.434007;
    check = check_matrix_eq(obs2_estimates_h1, expected_obs2_h1);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_obs1_h3(1, 4);
    expected_obs1_h3 << 0.91875, 0.918454, 0.918545, 0.918483;
    check = check_matrix_eq(obs1_estimates_h3, expected_obs1_h3);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_obs2_h3(1, 4);
    expected_obs2_h3 << 0.818174, 0.818918, 0.818499, 0.818842;
    check = check_matrix_eq(obs2_estimates_h3, expected_obs2_h3);
    BOOST_TEST(check.first, check.second);

    MatrixXd expected_fixed(3, 4);
    expected_fixed << -1, 0.500324, 0.506335, 0.504282, -1, 0.294363, 0.286289,
        0.291487, -1, 0.205313, 0.207376, 0.204232;
    check = check_matrix_eq(fixed_estimates[0], expected_fixed.row(0));
    BOOST_TEST(check.first, check.second);
    check = check_matrix_eq(fixed_estimates[1], expected_fixed.row(1));
    BOOST_TEST(check.first, check.second);
    check = check_matrix_eq(fixed_estimates[2], expected_fixed.row(2));
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
    FactorGraph::create_from_unrolled_dbn(*model).print_graph(cout);
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

BOOST_AUTO_TEST_CASE(particle_filter) {
    /**
     * Test sampling generation
     */

    fs::current_path("../../test");

    // Data
    Eigen::MatrixXd data_matrix(1, 4);
    data_matrix << NO_OBS, 1, 0, 1;
    EvidenceSet data;
    data.add_data("Obs", Tensor3(data_matrix));

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/dbn_small.json"));

    model->unroll(data.get_time_steps(), true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    shared_ptr<SamplerEstimator> fixed_estimator =
        make_shared<SamplerEstimator>(model, 0, "Fixed");
    shared_ptr<SamplerEstimator> state_estimator =
        make_shared<SamplerEstimator>(model, 0, "State");
    shared_ptr<SamplerEstimator> state_estimator_h3 =
        make_shared<SamplerEstimator>(
            model, 3, "State", Eigen::VectorXd::Constant(1, 1));
    ParticleFilterEstimator estimator(model, 5000, gen, 4);
    estimator.add_base_estimator(fixed_estimator);
    estimator.add_base_estimator(state_estimator);
    estimator.add_base_estimator(state_estimator_h3);

    estimator.prepare();
    estimator.estimate(data);

    double tolerance = 0.1;

    vector<Eigen::MatrixXd> tmp(3);

    // Fixed
    tmp[0] = Eigen::MatrixXd(1, 4);
    tmp[0] << 0, 0.487544483985765, 0.478224303688923, 0.465036546679178;
    tmp[1] = Eigen::MatrixXd(1, 4);
    tmp[1] << 0, 0.307473309608541, 0.318151085898979, 0.328913085804547;
    tmp[2] = Eigen::MatrixXd(1, 4);
    tmp[2] << 0, 0.204982206405694, 0.203624610412097, 0.206050367516275;
    Tensor3 expected_fixed(tmp);
    Tensor3 fixed_estimates =
        Tensor3(fixed_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(fixed_estimates, expected_fixed, tolerance));

    // State
    tmp[0] << 0.300000000000000, 0.450889679715303, 0.262470935505681,
        0.445661170635589;
    tmp[1] << 0.500000000000000, 0.184163701067616, 0.533619993073993,
        0.182120380793999;
    tmp[2] << 0.200000000000000, 0.364946619217082, 0.203909071420326,
        0.372218448570412;
    Tensor3 expected_state(tmp);
    Tensor3 state_estimates =
        Tensor3(state_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(state_estimates, expected_state, tolerance));

    cout << state_estimates << endl;
}

BOOST_AUTO_TEST_CASE(particle_filter_edhmm) {
    /**
     * Test sampling generation
     */

    fs::current_path("../../test");

    // Data
    EvidenceSet data("data/edhmm_exact");

    // Model
    DBNPtr model = make_shared<DynamicBayesNet>(
        DynamicBayesNet::create_from_json("models/edhmm_exact.json"));

    model->unroll(3, true);
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    shared_ptr<SamplerEstimator> fixed_estimator =
        make_shared<SamplerEstimator>(model, 0, "X");
    shared_ptr<SamplerEstimator> state_estimator =
        make_shared<SamplerEstimator>(model, 0, "State");
    ParticleFilterEstimator estimator(model, 1000, gen, 4);
    estimator.add_base_estimator(fixed_estimator);
    estimator.add_base_estimator(state_estimator);

    estimator.prepare();
    estimator.estimate(data);

    double tolerance = 0.1;

    vector<Eigen::MatrixXd> tmp(2);

    // Fixed
    tmp[0] = Eigen::MatrixXd(3, 3);
    tmp[0] << 0.300000000000000, 0.280193102611091, 0.299770796337916,
        0.300000000000000, 0.296558942602415, 0.326256122382083,
        0.300000000000000, 0.357164701839202, 0.369581233225729;
    tmp[1] = Eigen::MatrixXd(3, 3);
    tmp[1] << 0.700000000000000, 0.719806897388909, 0.700229203662084,
        0.700000000000000, 0.703441057397585, 0.673743877617917,
        0.700000000000000, 0.642835298160798, 0.630418766774271;
    Tensor3 expected_fixed(tmp);
    Tensor3 fixed_estimates =
        Tensor3(fixed_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(fixed_estimates, expected_fixed, tolerance));

    // State
    tmp.resize(3);
    tmp[0] << 0.100000000000000, 0.419337322156391, 0.050132215740921,
        0.100000000000000, 0.013690494759586, 0.283051616180588,
        0.800000000000000, 0.331975148485154, 0.082254553733401;
    tmp[1] << 0.100000000000000, 0.064758439123537, 0.016177740115953,
        0.800000000000000, 0.966356173762999, 0.656128456484349,
        0.100000000000000, 0.547538495549392, 0.190278854229972;
    tmp[2] = Eigen::MatrixXd(3, 3);
    tmp[2] << 0.800000000000000, 0.515904238720073, 0.933690044143126,
        0.100000000000000, 0.019953331477415, 0.060819927335064,
        0.100000000000000, 0.120486355965454, 0.727466592036626;
    Tensor3 expected_state(tmp);
    Tensor3 state_estimates =
        Tensor3(state_estimator->get_estimates().estimates);
    BOOST_TEST(check_tensor_eq(state_estimates, expected_state, tolerance));
}

BOOST_AUTO_TEST_SUITE_END()