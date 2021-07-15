#pragma once

#include "pipeline/estimation/SamplerEstimator.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a metric to estimate the distribution of the possible
         * marker legend version assignments given samples of individual player
         * marker legend versions generated until the end of the mission.
         */
        class MarkerLegendVersionAssignmentEstimator : public SamplerEstimator {
          public:
            inline const static std::string NAME =
                "MarkerLegendVersionAssignment";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the estimator.
             *
             * @param model: DBN
             *
             */
            MarkerLegendVersionAssignmentEstimator(
                const std::shared_ptr<DynamicBayesNet>& model);

            ~MarkerLegendVersionAssignmentEstimator();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            MarkerLegendVersionAssignmentEstimator(
                const MarkerLegendVersionAssignmentEstimator& estimator);

            MarkerLegendVersionAssignmentEstimator&
            operator=(const MarkerLegendVersionAssignmentEstimator& estimator);

            MarkerLegendVersionAssignmentEstimator(
                MarkerLegendVersionAssignmentEstimator&&) = default;

            MarkerLegendVersionAssignmentEstimator&
            operator=(MarkerLegendVersionAssignmentEstimator&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void prepare() override;

            std::string get_name() const override;

            /**
             * Estimate the marker version assignment distribution based on the
             * particles generated until the end of the mission related to
             * individual player marker legend version.
             *
             * @param particles: particles for the last time step processed
             * @param projected_particles: particles generated by transitioning
             * some time steps into the future
             * @param marginals: marginal distributions for nodes that were
             * rao-blackwellized
             * @param data_point_idx: index of the data point for which
             * particles were generated
             * @param time_step: time step when the first particles in de set
             * were generated
             */
            void estimate(const EvidenceSet& particles,
                          const EvidenceSet& projected_particles,
                          const EvidenceSet& marginals,
                          int data_point_idx,
                          int time_step) override;
        };

    } // namespace model
} // namespace tomcat