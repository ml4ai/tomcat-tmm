#include "Node.h"

#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * A node with constant numerical values assigned to it.
         */
        class NumericNode : public Node {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates a constant node with a numerical value assigned to it.
             *
             * @param value: node's numerical assignment
             * @param label: node's label
             */
            NumericNode(double value, const std::string& label = "unlabeled");

            /**
             * Creates a constant node with a multidimensional value assigned to
             * it.
             *
             * @param values: node's constant assignment
             * * @param label: node's label
             */
            NumericNode(const Eigen::VectorXd& values,
                         const std::string& label = "unlabeled");

            /**
             * Creates a constant node with a multidimensional value assigned to
             * it.
             *
             * @param values: node's constant assignment
             * * @param label: node's label
             */
            NumericNode(const Eigen::VectorXd&& values,
                         const std::string& label = "unlabeled");

            ~NumericNode();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            NumericNode(const NumericNode& node);

            NumericNode& operator=(const NumericNode& node);

            NumericNode(NumericNode&&) = default;

            NumericNode& operator=(NumericNode&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            std::unique_ptr<Node> clone() const override;

            std::string get_description() const override;

            std::string get_timed_name() const override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Creates a default metadata for a constant node.
             *
             * @param label: node's label
             * @param sample_size: dimensionality of the value stored in the
             * node.
             */
            void create_default_metadata(const std::string& label,
                                         int sample_size);
        };

    } // namespace model
} // namespace tomcat
