#pragma once

#include <memory>
#include <vector>

#include "sampling/Sampler.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a custom metric computed from generated samples by a
         * sampler.
         */
        class CustomSamplingMetric {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an abstract instance of a custom sampling metric.
             *
             * @param inference_horizon: window in the future where estimates
             * have to be computed at. If negative, indicates a adaptable
             * horizon, that comprises the current time step being evaluated
             * and the end of the mission.
             *
             */
            CustomSamplingMetric(int inference_horizon = -1);

            virtual ~CustomSamplingMetric();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            CustomSamplingMetric(const CustomSamplingMetric&) = delete;

            CustomSamplingMetric&
            operator=(const CustomSamplingMetric&) = delete;

            CustomSamplingMetric(CustomSamplingMetric&&) = default;

            CustomSamplingMetric& operator=(CustomSamplingMetric&&) = default;

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Calculates a value for the metric based on samples generated
             * by a sampler.
             *
             * @param sampler: sampler that holds the generated samples
             * @param time_step: time step at which the metric is being
             * computed
             *
             * @return List of values (one per player or other kinds of
             * divisions if necessary).
             */
            virtual std::vector<double>
            calculate(const std::shared_ptr<Sampler>& sampler, int time_step) const = 0;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            int get_inference_horizon() const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Returns a matrix with 1 where there was a transition from 1 to
             * 0 in the binary_matrix.
             *
             * @param binary_matrix: matrix of zeros and ones.
             *
             * @return Successful transitions matrix.
             */
            Eigen::MatrixXi
            get_transitions(const Eigen::MatrixXi& binary_matrix) const;

            /**
             * Copy attributes from another instance.
             *
             * @param node_probability: another instance
             */
            void copy(const CustomSamplingMetric& node_probability);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            int inference_horizon;

        };

    } // namespace model
} // namespace tomcat

//
// Created by Paulo Soares on 3/17/21.
//

#ifndef TOMCAT_TMM_CUSTOMSAMPLINGMETRIC_H
#define TOMCAT_TMM_CUSTOMSAMPLINGMETRIC_H

class CustomSamplingMetric {};

#endif // TOMCAT_TMM_CUSTOMSAMPLINGMETRIC_H
