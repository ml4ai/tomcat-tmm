#include <memory>

#include <boost/program_options.hpp>
#include <gsl/gsl_rng.h>

#include "asist/study1/ASISTSinglePlayerMessageConverter.h"
#include "asist/study2/ASISTMultiPlayerMessageConverter.h"
#include "asist/study3/ASISTStudy3MessageConverter.h"
#include "experiments/Experimentation.h"
#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/OnlineLogger.h"
#include "reporter/EstimateReporter.h"

/**
 * This program is responsible for starting an agent's real-time inference
 * engine. It connects to an informed message bus and calculates inferences
 * and predictions as new messages come arrive. It only finishes running when
 * the program is explicitly killed.
 */

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

void start_agent(const string& model_dir,
                 const string& agent_json,
                 const string& params_dir,
                 const string& broker_json,
                 const string& map_json,
                 const string& reporter_type,
                 const string& reporter_settings_json,
                 const string& eval_dir,
                 const string& log_dir,
                 int study_num,
                 int num_seconds,
                 int time_step_size,
                 int num_jobs,
                 int num_players) {

    shared_ptr<gsl_rng> random_generator(gsl_rng_alloc(gsl_rng_mt19937));

    EstimateReporterPtr reporter;
    if (!reporter_type.empty()) {
        fstream file;
        file.open(reporter_settings_json);
        if (file.is_open()) {
            nlohmann::json reporter_settings = nlohmann::json::parse(file);
            reporter =
                EstimateReporter::factory(reporter_type, reporter_settings);
        }
        else {
            throw TomcatModelException(
                "File with reporter settings was not found.");
        }
    }

    MsgConverterPtr converter;
    if (study_num == 1) {
        converter = make_shared<ASISTSinglePlayerMessageConverter>(
            num_seconds, time_step_size, map_json);
    }
    else if (study_num == 2) {
        converter = make_shared<ASISTMultiPlayerMessageConverter>(
            num_seconds, time_step_size, map_json, num_players);
    }
    else {
        converter = make_shared<ASISTStudy3MessageConverter>(
            num_seconds, time_step_size, map_json, num_players);
    }

    Experimentation experimentation(random_generator,
                                    Timer::get_current_timestamp());
    int num_time_steps = num_seconds / time_step_size;
    experimentation.set_online_estimation_process(agent_json,
                                                  model_dir,
                                                  num_jobs,
                                                  num_time_steps - 1,
                                                  broker_json,
                                                  converter,
                                                  reporter,
                                                  log_dir);
    experimentation.start_real_time_estimation(params_dir, eval_dir);
}

int main(int argc, char* argv[]) {
    string model_dir;
    string agent_json;
    string params_dir;
    string broker_json;
    string map_json;
    string reporter_type;
    string reporter_settings_json;
    string eval_dir;
    string log_dir;
    unsigned int num_seconds;
    unsigned int time_step_size;
    unsigned int num_jobs;
    unsigned int num_players;
    unsigned int study_num;

    po::options_description desc("Allowed options");
    desc.add_options()(
        "help,h",
        "This program starts a new ToMCAT agent to make inferences and "
        "predictions in real time, as new relevant messages are published to "
        "a message bus. ToMCAT can publish back to the message bus if a "
        "reporter is "
        "provided.")("model_dir",
                     po::value<string>(&model_dir)->required(),
                     "Directory where the agent's model definition is saved.")(
        "agent_json",
        po::value<string>(&agent_json)->required(),
        "Filepath of the json file containing definitions about the agent")(
        "params_dir",
        po::value<string>(&params_dir)->default_value(""),
        "Directory where the pre-trained model's parameters are saved.")(
        "broker_json",
        po::value<string>(&broker_json),
        "Json containing the address and port of the message broker to"
        " connect to.\n")(
        "map_json",
        po::value<string>(&map_json)->default_value(""),
        "Path to the json file containing the map configuration.")(
        "seconds",
        po::value<unsigned int>(&num_seconds)->required(),
        "Number of seconds in the mission. The agent won't close after "
        "messages beyond this number of seconds have arrived, but no more "
        "inferences will be made.")(
        "step_size",
        po::value<unsigned int>(&time_step_size)->default_value(1)->required(),
        "Size of a time step in seconds.")(
        "jobs",
        po::value<unsigned int>(&num_jobs)->default_value(4),
        "Number of jobs used for multi-thread inference.")(
        "players",
        po::value<unsigned int>(&num_players)->default_value(3),
        "Number of players in the multiplayer mission.")(
        "study_num",
        po::value<unsigned int>(&study_num)->default_value(3)->required(),
        "Study number for message conversion and reporter definition: 1, 2 or "
        "3")("reporter",
             po::value<string>(&reporter_type)->default_value(""),
             "asist_study2_reporter\nasist_study3_reporter")(
        "reporter_settings_json",
        po::value<string>(&reporter_settings_json)->default_value(""),
        "Filepath to a json file containing reporter settings.")(
        "eval_dir",
        po::value<string>(&eval_dir)->default_value(""),
        "Directory where the evaluation file and report (if "
        "requested) will be saved.")(
        "log_dir",
        po::value<string>(&log_dir)->default_value("")->required(),
        "Directory where the log file must be created.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    start_agent(model_dir,
                agent_json,
                params_dir,
                broker_json,
                map_json,
                reporter_type,
                reporter_settings_json,
                eval_dir,
                log_dir,
                (int)study_num,
                (int)num_seconds,
                (int)time_step_size,
                (int)num_jobs,
                (int)num_players);
}
