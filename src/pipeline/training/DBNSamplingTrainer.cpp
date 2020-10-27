#include "DBNSamplingTrainer.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        DBNSamplingTrainer::DBNSamplingTrainer(
            shared_ptr<gsl_rng> random_generator,
            shared_ptr<Sampler> sampler,
            int num_samples)
            : random_generator(random_generator), sampler(sampler),
              num_samples(num_samples) {}

        DBNSamplingTrainer::~DBNSamplingTrainer() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        DBNSamplingTrainer::DBNSamplingTrainer(
            const DBNSamplingTrainer& trainer) {
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
        }

        void DBNSamplingTrainer::prepare() {}

        void DBNSamplingTrainer::fit(const EvidenceSet& training_data) {
            this->param_label_to_samples.clear();
            this->sampler->set_num_in_plate_samples(
                training_data.get_num_data_points());
            this->sampler->add_data(training_data);
            this->sampler->sample(this->random_generator, this->num_samples);

            shared_ptr<DynamicBayesNet> model = this->sampler->get_model();

            for (const auto& param_label : model->get_parameter_node_labels()) {
                this->param_label_to_samples[param_label] =
                    this->sampler->get_samples(param_label);
            }

            this->update_model_from_partials(false);
        }

        void DBNSamplingTrainer::get_info(nlohmann::json& json) const {
            json["type"] = "sampling";
            json["num_samples"] = this->num_samples;
            this->sampler->get_info(json["algorithm"]);
        }

        shared_ptr<DynamicBayesNet> DBNSamplingTrainer::get_model() const {
            return this->sampler->get_model();
        }

    } // namespace model
} // namespace tomcat
