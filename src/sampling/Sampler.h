#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        /**
         * Abstract representation of a class to generate samples from a DBN
         * model.
         */
        class Sampler {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------
            /**
             * Creates an instance of an abstract sampler.
             */
            Sampler();

            /**
             * Creates an abstract instance of an abstract sampler for an
             * unrolled DBN.
             *
             * @param model: unrolled DBN
             * @param num_jobs: number of threads for parallel sampling
             */
            Sampler(const std::shared_ptr<DynamicBayesNet>& model,
                    int num_jobs);

            virtual ~Sampler();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses. To deep copy, the clone method must be used.
            Sampler(const Sampler&) = delete;

            Sampler& operator=(const Sampler&) = delete;

            Sampler(Sampler&&) = default;

            Sampler& operator=(Sampler&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Generates samples for the latent nodes.
             *
             * @param random_generator: random number generator
             * @param num_samples: number of samples to generate
             */
            void sample(const std::shared_ptr<gsl_rng>& random_generator,
                        int num_samples);

            /**
             * Adds data to the sampler.
             *
             * @param data: observed values for the observable nodes over time.
             */
            void add_data(const EvidenceSet& data);

            /**
             * Saves generated samples to files in a specific folder.
             * @param output_folder: folder where the files should be saved.
             * @param excluding: list of node labels to exclude.
             */
            void save_samples_to_folder(
                const std::string& output_folder,
                const std::unordered_set<std::string> excluding = {}) const;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            /**
             * Returns samples generated for a specific latent node.
             *
             * @param node_label: latent node label
             *
             * @return Samples over time. A matrix of dimension (num_samples,
             * time_steps).
             */
            virtual Tensor3 get_samples(const std::string& node_label) const;

            /**
             * Method to do any sort of initialization prior a call to the
             * sample function.
             */
            virtual void prepare();

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Function called by the function sample to generate samples for
             * unfrozen nodes. The function sample freezes observable nodes and
             * releases them in the end. Calling this function in between.
             *
             * @param random_generator: random number generator
             */
            virtual void
            sample_latent(const std::shared_ptr<gsl_rng>& random_generator) = 0;

            /**
             * Writes information about the splitter in a json object.
             *
             * @param json: json object
             */
            virtual void get_info(nlohmann::json& json) const = 0;

            /**
             * Creates a deep copy of the sampler.
             *
             * @param unroll_model: whether the model's clone must be
             * unrolled into the same number of time steps as the original model
             *
             * @return Pointer to the new sampler created.
             */
            virtual std::unique_ptr<Sampler> clone(bool unroll_model) const = 0;

            /**
             * Return labels of all the nodes sampled.
             *
             * @return Labels of sampled nodes.
             */
            virtual std::unordered_set<std::string>
            get_sampled_node_labels() const = 0;

            // -----------------------------------------------------------------
            // Getters & Setters
            // -----------------------------------------------------------------
            void set_num_in_plate_samples(int num_in_plate_samples);

            const std::shared_ptr<DynamicBayesNet>& get_model() const;

            void set_model(const std::shared_ptr<DynamicBayesNet>& model);

            int get_num_jobs() const;

            int get_num_samples() const;

            void set_show_progress(bool show_progress);

            void
            set_time_steps_per_sample(std::vector<int>& time_steps_per_sample);

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copy data members from another sampler instance.
             *
             * @param sampler: Sampler
             */
            void copy_sampler(const Sampler& sampler);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            std::shared_ptr<DynamicBayesNet> model;

            // Labels of the nodes that were sampled (this has to be filled in
            // the derived classes).
            std::unordered_set<std::string> sampled_node_labels;

            // The number of samples generated for in-plate nodes. If data is
            // provided for some node, the number of data points has to be the
            // same as the number of in-plate samples.
            int num_in_plate_samples = -1;

            EvidenceSet data;

            // Number of threads created for parallel sampling.
            int num_jobs = 1;

            int num_samples = 0;

            bool show_progress = true;

            // Used to allow nodes to have different number of time steps per
            // sample (event based learning). Nodes with time step higher than
            // the maximum time step for a specific assignment row, will have
            // value equals to NO_OBS in such a row.
            std::vector<int> time_steps_per_sample;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Freeze nodes that have data assigned to them.
             */
            void freeze_observable_nodes();

            /**
             * Unfreeze nodes that have data assigned to them.
             */
            void unfreeze_observable_nodes();
        };
    } // namespace model
} // namespace tomcat
