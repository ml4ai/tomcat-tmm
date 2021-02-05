#include "NodeMetadata.h"
#include "RandomVariableNode.h"
#include <fmt/format.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        NodeMetadata::NodeMetadata() {}

        NodeMetadata::NodeMetadata(const string& label,
                                   bool replicable,
                                   bool parameter,
                                   bool single_time_link,
                                   bool in_plate,
                                   bool timer,
                                   int initial_time_step,
                                   int sample_size,
                                   int cardinality)
            : label(label), replicable(replicable), parameter(parameter),
              single_time_link(single_time_link), in_plate(in_plate),
              timer(timer), initial_time_step(initial_time_step),
              sample_size(sample_size), cardinality(cardinality) {

            if (replicable && parameter) {
                throw TomcatModelException("Parameter nodes cannot be "
                                           "replicable. They show up once and"
                                           " are shareable among the "
                                           "distributions that govern the "
                                           "timed copies of data nodes.");
            }
        }

        //----------------------------------------------------------------------
        // Destructor
        //----------------------------------------------------------------------
        NodeMetadata::~NodeMetadata() {}

        //----------------------------------------------------------------------
        // Operator overload
        //----------------------------------------------------------------------
        ostream& operator<<(ostream& os, const NodeMetadata& metadata) {

            os << metadata.get_description();

            return os;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        string NodeMetadata::get_timed_name(const string& label,
                                            int time_step) {
            return fmt::format("({},{})", label, time_step);
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        string NodeMetadata::get_description() const {
            stringstream ss;

            ss << "Metadata: {\n";
            ss << " Label: " << this->label << "\n";
            ss << " Cardinality: " << this->cardinality << "\n";
            ss << " Sample Size: " << this->sample_size << "\n";
            ss << " Initial Time Step: " << this->initial_time_step << "\n";
            ss << " Repeatable: " << this->replicable << "\n";
            ss << " Timer: " << this->timer << "\n";
            ss << " In-Plate: " << this->in_plate << "\n";
            ss << " Parameter: " << this->parameter << "\n";
            if (!this->parent_links.empty()) {
                ss << " Parent Links:\n";
                ss << " [\n";

                for (auto& link : this->parent_links) {
                    ss << "  (";
                    ss << link.parent_node_metadata->label;
                    ss << ", ";
                    ss << link.time_crossing;
                    ss << ")\n";
                }
                ss << " ]\n";
            }

            ss << "}";

            return ss.str();
        }

        void NodeMetadata::add_parent_link(
            const shared_ptr<NodeMetadata>& parent_node, bool time_crossing) {

            if (parent_node->cardinality == 0 && !parent_node->parameter &&
                !parent_node->is_timer()) {
                throw TomcatModelException("A non-parameter node cannot be "
                                           "child of a node sampled from a "
                                           "continuous distribution.");
            }

            if (!parent_node->replicable && !parent_node->single_time_link &&
                !time_crossing) {
                throw TomcatModelException("The parent node is "
                                           "non-replicable and multi-time, "
                                           "therefore its connections must "
                                           "cross time.");
            }

            if (parent_node->is_timer()) {
                throw TomcatModelException("A timer cannot be parent of "
                                           "another node.");
            }

            // At least one of the parents of the node that owns this metadata
            // is a parameter node?
            this->parameter_parents |= parent_node->parameter;

            // TODO -  This will be removed as soon as we forbid replicable
            //  parameter nodes.
            this->replicable_parameter_parent |=
                parent_node->parameter && parent_node->replicable;

            ParentLink link{parent_node, time_crossing};
            this->parent_links.push_back(move(link));
        }

        string NodeMetadata::get_timed_name(int time_step) const {
            return NodeMetadata::get_timed_name(this->label, time_step);
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        const string& NodeMetadata::get_label() const { return this->label; }

        int NodeMetadata::get_initial_time_step() const {
            return this->initial_time_step;
        }

        bool NodeMetadata::is_replicable() const { return this->replicable; }

        bool NodeMetadata::is_parameter() const { return this->parameter; }

        bool NodeMetadata::is_single_time_link() const {
            return this->single_time_link;
        }

        bool NodeMetadata::is_in_plate() const { return this->in_plate; }

        int NodeMetadata::get_sample_size() const { return this->sample_size; }

        int NodeMetadata::get_cardinality() const { return this->cardinality; }

        const vector<ParentLink>& NodeMetadata::get_parent_links() const {
            return this->parent_links;
        }

        bool NodeMetadata::has_parameter_parents() const {
            return this->parameter_parents;
        }

        bool NodeMetadata::has_replicable_parameter_parent() const {
            return replicable_parameter_parent;
        }

        bool NodeMetadata::is_timer() const { return timer; }

        bool NodeMetadata::is_connected() const { return connected; }

        void NodeMetadata::set_connected(bool connected) {
            this->connected = connected;
        }

        const shared_ptr<NodeMetadata>&
        NodeMetadata::get_timer_metadata() const {
            return timer_metadata;
        }

        void
        NodeMetadata::set_timer_metadata(shared_ptr<NodeMetadata>& metadata) {
            if (!metadata->is_timer()) {
                throw TomcatModelException("The metadata informed does not "
                                           "belong to a timer node.");
            }

            if (metadata->is_connected()) {
                throw TomcatModelException("The timer node is already "
                                           "associated to another node.");
            }

            if (!this->is_replicable()) {
                throw TomcatModelException("A timer must be associated to"
                                           " a replicable node.");
            }

            if (this->initial_time_step > 0) {
                throw TomcatModelException("The initial time step of a time "
                                           "controlled node must be 0.");
            }

            this->timer_metadata = metadata;

            // Previous timer controls the current node's assignment.
            ParentLink link{metadata, true};
            this->parent_links.push_back(link);

            // Mark timer metadata so it cannot be used to control another node.
            metadata->set_connected(true);
        }
    } // namespace model
} // namespace tomcat
