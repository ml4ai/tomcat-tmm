//=======================================================================
// generating the graph-based representation for Saturn map using Boost Graph Library
// Author: Liang Zhang
//

#include "gen_graph.h"

using namespace std;

int main() {
    gen_graph gg = gen_graph();  // initialize a gen_graph class to process the json map of Saturn
    gen_graph::Graph g = gg.generate_graph();
}

