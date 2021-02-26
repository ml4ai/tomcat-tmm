#include "Agent.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Agent::Agent(const string& id) : id(id) {}

        Agent::~Agent() {}

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        const string& Agent::get_id() const { return id; }
    } // namespace model
} // namespace tomcat
