#include "Estimator.h"

#include <sstream>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Estimator::Estimator(const shared_ptr<Model>& model)
            : model(model) {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Estimator::copy(const Estimator& estimator) {
            this->model = estimator.model;
            this->training_data = estimator.training_data;
            this->show_progress = estimator.show_progress;
        }

        void Estimator::prepare() {
        }

        void Estimator::cleanup() {
        }

        void Estimator::set_show_progress(bool show_progress) {
            this->show_progress = show_progress;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void Estimator::set_training_data(const EvidenceSet& training_data) {
            this->training_data = training_data;
        }

        const shared_ptr<Model>& Estimator::get_model() const {
            return model;
        }

    } // namespace model
} // namespace tomcat
