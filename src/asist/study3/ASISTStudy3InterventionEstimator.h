#pragma once

#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/Estimator.h"
#include "pipeline/estimation/SumProductEstimator.h"
#include "utils/Definitions.h"

namespace tomcat::model {

    /**
     * This class represents an estimator to perform interventions in Study 3 of
     * the ASIST program. It consists of a collection of models to keep track of
     * different beliefs for different players in the mission.
     */
    class ASISTStudy3InterventionEstimator : public Estimator {
      public:
        //------------------------------------------------------------------
        // Constructors & Destructor
        //------------------------------------------------------------------

        ASISTStudy3InterventionEstimator(const std::string& map_filepath,
                              const std::string& threat_room_model_filepath);

        ~ASISTStudy3InterventionEstimator() = default;

        //------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //------------------------------------------------------------------
        ASISTStudy3InterventionEstimator(const ASISTStudy3InterventionEstimator& estimator);

        ASISTStudy3InterventionEstimator&
        operator=(const ASISTStudy3InterventionEstimator& estimator);

        ASISTStudy3InterventionEstimator(ASISTStudy3InterventionEstimator&&) = default;

        ASISTStudy3InterventionEstimator& operator=(ASISTStudy3InterventionEstimator&&) = default;

        //------------------------------------------------------------------
        // Static functions
        //------------------------------------------------------------------

        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------
        void estimate(const EvidenceSet& new_data) override;

        void get_info(nlohmann::json& json) const override;

        std::string get_name() const override;

        bool is_binary_on_prediction() const override;

        void prepare() override;

        //------------------------------------------------------------------
        // Getters & Setters
        //------------------------------------------------------------------

        //------------------------------------------------------------------
        // Virtual member functions
        //------------------------------------------------------------------

      protected:
        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------

        /**
         * Copies data member from another estimator.
         *
         * @param estimator: other estimator
         */
        void copy(const ASISTStudy3InterventionEstimator& estimator);

        //------------------------------------------------------------------
        // Data members
        //------------------------------------------------------------------

      private:
        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------

        /**
         * Parses the map to identify and store room ids.
         *
         * @param map_filepath: path to the descriptive file of the map used in
         * the mission
         */
        void parse_map(const std::string& map_filepath);

        /**
         * Initialize all estimators used by the intervention estimator.
         *
         * @param threat_room_model_filepath: path to the json file containing
         * the model definition for belief about threat rooms
         */
        void
        create_belief_estimators(const std::string& threat_room_model_filepath);

        /**
         * Initializes estimators that deal with belief about threat rooms.
         *
         * @param threat_room_model_filepath: path to the json file containing
         * the model definition for belief about threat rooms
         */
        void create_threat_room_belief_estimators(
            const std::string& threat_room_model_filepath);

        /**
         * Estimate beliefs about threat rooms given new evidence.
         *
         * @param new_dat: new evidence data
         */
        void update_threat_room_beliefs(const EvidenceSet& new_data);

        //------------------------------------------------------------------
        // Data members
        //------------------------------------------------------------------

        // Indices of each role
        const static int NUM_ROLES = 3;

        const static int MEDIC = 0;
        const static int TRANSPORTER = 1;
        const static int ENGINEER = 2;

        std::unordered_map<std::string, int> room_id_to_idx;
        std::vector<std::string> room_ids;

        // Stores belief estimates about threat rooms per player and room.
        std::vector<std::vector<SumProductEstimator>>
            threat_room_belief_estimators;
    };

} // namespace tomcat::model
