//#pragma once
//
//#include <vector>
//
//#include "eigen3/Eigen/Dense"
//
//#include "pgm/inference/VariableNode.h"
//#include "utils/Definitions.h"
//#include "distribution/Distribution.h"
//
//
//namespace tomcat {
//    namespace model {
//
//        /**
//         * Represents aggregated probabilities of possible past segment
//         * combinations for Explicit Duration HMM. This node stores
//         * posteriors and performs operations to push estimates forward in
//         * time and efficiently compute exact inference in models with nodes
//         * controlled by a timer.
//         */
//        class SegmentNode : public MessageNode {
//          public:
//            //------------------------------------------------------------------
//            // Constructors & Destructor
//            //------------------------------------------------------------------
//
//            /**
//             * Creates a ...
//             *
//             */
//            SegmentNode();
//
//            ~SegmentNode();
//
//            //------------------------------------------------------------------
//            // Copy & Move constructors/assignments
//            //------------------------------------------------------------------
//
//            SegmentNode(const SegmentNode&) = default;
//
//            SegmentNode& operator=(const SegmentNode&) = default;
//
//            SegmentNode(SegmentNode&&) = default;
//
//            SegmentNode& operator=(SegmentNode&&) = default;
//
//            //------------------------------------------------------------------
//            // Pure virtual functions
//            //------------------------------------------------------------------
//
//            //------------------------------------------------------------------
//            // Virtual functions
//            //------------------------------------------------------------------
//
//            //------------------------------------------------------------------
//            // Member functions
//            //------------------------------------------------------------------
//
//            //------------------------------------------------------------------
//            // Getters & Setters
//            //------------------------------------------------------------------
//
//            //------------------------------------------------------------------
//            // Data members
//            //------------------------------------------------------------------
//
//            // Variable nodes that are parents of the segment duration
//            // distribution, excluding the node that is controlled by the
//            // timer that defines the segments durations (state node).
//            VarNodePtrVec duration_dependencies;
//
//            // Variable nodes that are parents of the state transition
//            // distribution, excluding the state node itself.
//            VarNodePtrVec transition_dependencies;
//
//            // Variable node that is controlled by the timer that defines the
//            // segments.
//            VarNodePtrVec state_node;
//
//            // Matrix containing the probabilities of starting a new segment in
//            // the currently processed time step. Whenever a new segment is
//            // created, the immediate previous segment is closed as there's a
//            // transition to a new state. One matrix per each combination of
//            // duration dependencies assignments.
//            std::vector<Eigen::MatrixXd> closed_segments;
//
//            // Matrix containing the probabilities of extending the past
//            // segment one more time step into the future. One matrix per
//            // each combination of duration dependencies assignments.
//            std::vector<Eigen::MatrixXd> open_segments;
//
//            // One duration distribution per each combination of state
//            // assignment and duration dependencies assignments.
//            std::vector<Distribution> duration_distributions;
//        };
//
//    } // namespace model
//} // namespace tomcat
