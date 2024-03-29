#include "DBNSamplingTrainer.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        DBNSamplingTrainer::DBNSamplingTrainer(
            const shared_ptr<gsl_rng>& random_generator,
            const shared_ptr<Sampler>& sampler,
            int num_samples)
            : DBNTrainer(sampler->get_model()),
              random_generator(random_generator), sampler(sampler),
              num_samples(num_samples) {}

        DBNSamplingTrainer::~DBNSamplingTrainer() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        DBNSamplingTrainer::DBNSamplingTrainer(
            const DBNSamplingTrainer& trainer)
            : DBNTrainer(dynamic_pointer_cast<DynamicBayesNet>(this->model)) {
            this->copy_trainer(trainer);
        }

        DBNSamplingTrainer&
        DBNSamplingTrainer::operator=(const DBNSamplingTrainer& trainer) {
            this->copy_trainer(trainer);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void
        DBNSamplingTrainer::copy_trainer(const DBNSamplingTrainer& trainer) {
            this->random_generator = trainer.random_generator;
            this->sampler = trainer.sampler;
            this->num_samples = trainer.num_samples;
            this->param_label_to_samples = trainer.param_label_to_samples;
            this->model = trainer.model;
        }

        void DBNSamplingTrainer::prepare() { this->sampler->prepare(); }

        void DBNSamplingTrainer::fit(const EvidenceSet& training_data) {
            this->sampler->set_num_in_plate_samples(
                training_data.get_num_data_points());
            this->sampler->add_data(training_data);

            if (training_data.is_event_based()) {
                vector<int> time_steps_per_sample(
                    training_data.get_num_data_points());
                for (int d = 0; d < time_steps_per_sample.size(); d++) {
                    time_steps_per_sample[d] =
                        training_data.get_num_events_for(d);
                }
                this->sampler->set_time_steps_per_sample(time_steps_per_sample);
            }

            this->sampler->sample(this->random_generator, this->num_samples);

            shared_ptr<DynamicBayesNet> model = this->sampler->get_model();

            this->param_label_to_samples.push_back(
                unordered_map<string, Tensor3>());
            int split_idx = this->param_label_to_samples.size() - 1;
            for (const auto& param_node : model->get_parameter_nodes()) {
                if (!dynamic_pointer_cast<RandomVariableNode>(param_node)
                         ->is_frozen()) {
                    const string& param_label =
                        param_node->get_metadata()->get_label();
                    this->param_label_to_samples[split_idx][param_label] =
                        this->sampler->get_samples(param_label);
                }
            }

            this->update_model_from_partials(split_idx, false);
        }

        void DBNSamplingTrainer::get_info(nlohmann::json& json) const {
            json["type"] = "sampling";
            json["num_samples"] = this->num_samples;
            this->sampler->get_info(json["algorithm"]);
        }

    } // namespace model
} // namespace tomcat
