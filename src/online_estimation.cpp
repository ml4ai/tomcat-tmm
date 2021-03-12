#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <gsl/gsl_rng.h>

#include "converter/ASISTSinglePlayerMessageConverter.h"
#include "experiments/Experimentation.h"
#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/ASISTAgent.h"

/**
 * This program is responsible for starting an agent's real-time inference
 * engine. It connects to an informed message bus and calculates inferences
 * and predictions as new messages come arrive. It only finishes running when
 * the program is explicitly killed.
 */

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

void start_agent(const string& agent_id,
                 const string& model_json,
                 const string& params_dir,
                 const string& broker_json,
                 int num_connection_trials,
                 int milliseconds_before_retrial,
                 const string& map_json,
                 int num_seconds,
                 int time_step_size,
                 const string& inference_json,
                 int burn_in,
                 int num_samples,
                 int num_jobs) {

    shared_ptr<gsl_rng> random_generator(gsl_rng_alloc(gsl_rng_mt19937));

    shared_ptr<DynamicBayesNet> model = make_shared<DynamicBayesNet>(
        DynamicBayesNet ::create_from_json(model_json));
    int num_time_steps = num_seconds / time_step_size;
    model->unroll(num_time_steps, true);

    string broker_address = "localhost";
    int broker_port = 1883;
    string estimates_topic = "uaz/estimates";
    string log_topic = "uaz/log";
    fstream file;
    file.open(broker_json);
    if (file.is_open()) {
        nlohmann::json broker = nlohmann::json::parse(file);
        if (broker.contains("address")) {
            broker_address = broker["address"];
        }
        if (broker.contains("port")) {
            broker_port = broker["port"];
        }
        if (broker.contains("estimates_topic")) {
            estimates_topic = broker["estimates_topic"];
        }
        if (broker.contains("log_topic")) {
            log_topic = broker["log_topic"];
        }
    }

    shared_ptr<ASISTMessageConverter> message_converter =
        make_shared<ASISTSinglePlayerMessageConverter>(
            num_seconds, time_step_size, map_json);
    shared_ptr<ASISTAgent> agent = make_shared<ASISTAgent>(
        agent_id, estimates_topic, log_topic, message_converter);

    Experimentation experimentation(random_generator,
                                    model,
                                    agent,
                                    broker_address,
                                    broker_port,
                                    num_connection_trials,
                                    milliseconds_before_retrial);
    experimentation.add_estimators_from_json(
        inference_json, burn_in, num_samples, num_jobs, false);
    experimentation.start_real_time_estimation(params_dir);
}

int main(int argc, char* argv[]) {
    string agent_id;
    string model_json;
    string params_dir;
    string broker_json;
    string map_json;
    string inference_json;
    unsigned int num_connection_trials;
    unsigned int milliseconds_before_retrial;
    unsigned int num_seconds;
    unsigned int time_step_size;
    unsigned int burn_in;
    unsigned int num_samples;
    unsigned int num_jobs;

    po::options_description desc("Allowed options");
    desc.add_options()(
        "help,h",
        "This program starts a new ToMCAT agent to make inferences and "
        "predictions in real time, as new relevant messages are published to "
        "a message bus. If the model does not support exact inference (e.g. "
        "the model has at least one variable under a semi-Markov assumption "
        "or that follows a continuous distribution), approximate inference "
        "will be used.")(
        "agent_id", po::value<string>(&agent_id), "Agent identifier.")(
        "model_json",
        po::value<string>(&model_json)->required(),
        "Filepath of the json file containing the model definition.")(
        "params_dir",
        po::value<string>(&params_dir)->required(),
        "Directory where the pre-trained model's parameters are saved.")(
        "broker_json",
        po::value<string>(&broker_json),
        "Json containing the address and port of the message broker to"
        " connect to.\n")(
        "conn_trials",
        po::value<unsigned int>(&num_connection_trials)
            ->default_value(5)
            ->required(),
        "Number of trials to establish a connection with the message broker "
        "before ending the program.")(
        "conn_wait",
        po::value<unsigned int>(&milliseconds_before_retrial)
            ->default_value(3000)
            ->required(),
        "Number of milliseconds to wait before reattempting to establish a "
        "connection with the message broker in case of fail to "
        "successfully connect previously.")(
        "map_json",
        po::value<string>(&map_json)->required(),
        "Path to the json file containing the map configuration.")(
        "seconds",
        po::value<unsigned int>(&num_seconds)->required(),
        "Number of seconds in the mission. The agent won't close after "
        "messages beyond this number of seconds have arrived, but no more "
        "inference will be made.")(
        "step_size",
        po::value<unsigned int>(&time_step_size)->default_value(1)->required(),
        "Size of a time step in seconds.")(
        "inference_json",
        po::value<string>(&inference_json)->required(),
        "Filepath of the json file containing the variables and inference "
        "horizons to be evaluated by the pre-trained model.")(
        "burn_in",
        po::value<unsigned int>(&burn_in)->default_value(100)->required(),
        "Number of samples to generate until posterior convergence if "
        "approximate inference is used.")(
        "samples",
        po::value<unsigned int>(&num_samples)->default_value(100)->required(),
        "Number of samples used to estimate the parameters of the model "
        "after the burn-in period if approximate inference is used.")(
        "jobs",
        po::value<unsigned int>(&num_jobs)->default_value(4),
        "Number of jobs used for multi-thread inference.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    start_agent(agent_id,
                model_json,
                params_dir,
                broker_json,
                num_connection_trials,
                milliseconds_before_retrial,
                map_json,
                num_seconds,
                time_step_size,
                inference_json,
                burn_in,
                num_samples,
                num_jobs);
}
