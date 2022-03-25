#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pipeline/Model.h"

namespace tomcat {
    namespace model {

        /**
         * Represents an intervention model for study 3.
         */
        class ASISTStudy3InterventionModel : public Model {
          public:
            inline static const std::string NAME =
                "asist_study3_intervention_model";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------
            explicit ASISTStudy3InterventionModel(
                const nlohmann::json& json_settings);

            ~ASISTStudy3InterventionModel() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTStudy3InterventionModel(
                const ASISTStudy3InterventionModel& model) = delete;

            ASISTStudy3InterventionModel&
            operator=(const ASISTStudy3InterventionModel& model) = delete;

            ASISTStudy3InterventionModel(ASISTStudy3InterventionModel&&) =
                default;

            ASISTStudy3InterventionModel&
            operator=(ASISTStudy3InterventionModel&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            const std::shared_ptr<RandomVariableNode>&
            get_encouragement_node() const;

          private:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------
            ASISTStudy3InterventionModel() = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void save_to(const std::string& output_dir) const override;

            void load_from(const std::string& input_dir,
                           bool freeze_model) override;

            std::unique_ptr<Model> clone() const override;

            //            /**
            //             * Copy contents from another study 3 intervention
            //             model;
            //             *
            //             * @param model: study 3 intervention model
            //             */
            //            void copy(const ASISTStudy3InterventionModel& model);

            /**
             * Parse a json object containing the model definition.
             *
             * @param json_settings: json object containing the model definition
             */
            void parse_settings(const nlohmann::json& json_settings);

            /**
             * Create all components of the model.
             */
            void create_components();

            /**
             * Create motivation component.
             */
            void create_motivation_component();

            //------------------------------------------------------------------
            // Data member
            //------------------------------------------------------------------
            std::shared_ptr<RandomVariableNode> encouragement_node;
        };

    } // namespace model
} // namespace tomcat
