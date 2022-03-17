#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fstream>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

#include "pgm/RandomVariableNode.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Represents an abstract model that an agent can use. A realization of
         * an abstract model can be a PGM, a neural network etc.
         */
        class Model {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a DBN.
             */
            Model() = default;

            virtual ~Model() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Model(const Model&) = delete;

            Model& operator=(const Model&) = delete;

            Model(Model&&) = default;

            Model& operator=(Model&&) = default;

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Saves model's parameter values in individual files inside a given
             * directory. The directory is created if it does not exist.
             *
             * @param output_dir: folder where the files must be saved
             */
            virtual void save_to(const std::string& output_dir) const = 0;

            /**
             * Loads model's parameter assignments from files previously saved
             * in a specific directory and freeze the parameter nodes.
             *
             * @param input_dir: directory where the files with the parameters'
             * values are saved
             * @param freeze_model: if true parameters that had their values
             * loaded, will be frozen
             */
            virtual void load_from(const std::string& input_dir,
                                   bool freeze_model) = 0;

            /**
             * Creates a deep copy of the model.
             *
             * @return a deep copy of this DBN
             */
            virtual std::unique_ptr<Model> clone() const = 0;

        };

    } // namespace model
} // namespace tomcat
