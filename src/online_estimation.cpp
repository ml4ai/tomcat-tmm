#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <gsl/gsl_rng.h>

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "converter/ASISTSinglePlayerMessageConverter.h"
#include "experiments/Experimentation.h"
#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/ASISTStudy2EstimateReporter.h"
#include "pipeline/estimation/EstimateReporter.h"

/**
 * This program is responsible for starting an agent's real-time inference
 * engine. It connects to an informed message bus and calculates inferences
 * and predictions as new messages come arrive. It only finishes running when
 * the program is explicitly killed.
 */

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

struct ConverterTypes {
    const static int ASIST_SINGLE_PLAYER = 0;
    const static int ASIST_MULTI_PLAYER = 1;
};

struct ReporterTypes {
    const static int NONE = 0;
    const static int ASIST_STUDY2 = 1;
};

void start_agent(const string& model_json,
                 const string& agents_json,
                 const string& params_dir,
                 const string& broker_json,
                 const string& map_json,
                 int converter_type,
                 int reporter_type,
                 int num_seconds,
                 int time_step_size,
                 int num_particles,
                 int num_jobs,
                 int num_players,
                 bool exact_inference) {

    shared_ptr<gsl_rng> random_generator(gsl_rng_alloc(gsl_rng_mt19937));

    shared_ptr<DynamicBayesNet> model = make_shared<DynamicBayesNet>(
        DynamicBayesNet ::create_from_json(model_json));
    model->unroll(3, true);

    MsgConverterPtr converter;
    if (converter_type == ConverterTypes::ASIST_SINGLE_PLAYER) {
        converter = make_shared<ASISTSinglePlayerMessageConverter>(
            num_seconds, time_step_size, map_json);
    }
    else {
        converter = make_shared<ASISTMultiPlayerMessageConverter>(
            num_seconds, time_step_size, map_json, num_players);
    }

    EstimateReporterPtr reporter;
    if (reporter_type == ReporterTypes::ASIST_STUDY2) {
        reporter = make_shared<ASISTStudy2EstimateReporter>();
    }

    Experimentation experimentation(
        random_generator, model, broker_json, converter, reporter);
    int num_time_steps = num_seconds / time_step_size;
    experimentation.create_agents(agents_json,
                                  num_particles,
                                  num_jobs,
                                  false,
                                  exact_inference,
                                  num_time_steps - 1);
    experimentation.start_real_time_estimation(params_dir);
}

int main(int argc, char* argv[]) {
    string model_json;
    string agents_json;
    string params_dir;
    string broker_json;
    string map_json;
    unsigned int num_seconds;
    unsigned int time_step_size;
    unsigned int num_particles;
    unsigned int num_jobs;
    unsigned int num_players;
    unsigned int converter_type;
    unsigned int reporter_type;
    bool exact_inference;

    po::options_description desc("Allowed options");
    desc.add_options()(
        "help,h",
        "This program starts a new ToMCAT agent to make inferences and "
        "predictions in real time, as new relevant messages are published to "
        "a message bus. If the model does not support exact inference (e.g. "
        "the model has at least one variable under a semi-Markov assumption "
        "or that follows a continuous distribution), approximate inference "
        "will be used.")(
        "model_json",
        po::value<string>(&model_json)->required(),
        "Filepath of the json file containing the model definition.")(
        "agents_json",
        po::value<string>(&agents_json)->required(),
        "Filepath of the json file containing definitions about the agents' "
        "reasoning.")(
        "params_dir",
        po::value<string>(&params_dir)->required(),
        "Directory where the pre-trained model's parameters are saved.")(
        "broker_json",
        po::value<string>(&broker_json),
        "Json containing the address and port of the message broker to"
        " connect to.\n")(
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
        "particles",
        po::value<unsigned int>(&num_particles)
            ->default_value(1000)
            ->required(),
        "Number of particles used to estimate the parameters of the model "
        "if approximate inference is used.")(
        "jobs",
        po::value<unsigned int>(&num_jobs)->default_value(4),
        "Number of jobs used for multi-thread inference.")(
        "players",
        po::value<unsigned int>(&num_players)->default_value(3),
        "Number of players in the multiplayer mission.")(
        "mission_type",
        po::value<unsigned int>(&converter_type)->default_value(1)->required(),
        "0 - ASIST Singleplayer\n."
        "1 - ASIST Multiplayer")(
        "reporter",
        po::value<unsigned int>(&reporter_type)->default_value(1)->required(),
        "0 - None\n"
        "1 - ASIST Study 2\n")(
        "exact",
        po::bool_switch(&exact_inference)->default_value(false),
        "Whether to use exact or approximate inference.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    start_agent(model_json,
                agents_json,
                params_dir,
                broker_json,
                map_json,
                converter_type,
                reporter_type,
                num_seconds,
                time_step_size,
                num_particles,
                num_jobs,
                num_players,
                exact_inference);
}
