//=======================================================================
// This code generates the graph-based representation for the saturn map using Boost Graph Library.
// The room and portal information are stored in the saturn_room.json and saturn_portal.json under the data folder.
// The vertices represent the rooms and the portals (doors, openings, etc).
// The edges show the connectivity between the two adjacent nodes in the graph.
// Author: Liang Zhang
// Email: liangzh@email.arizona.edu
//

#include "gen_graph.h"

using namespace std;

int main() {
    gen_graph gg = gen_graph();  // initialize a gen_graph object to process the json map of Saturn
    gen_graph::Graph g = gg.generate_graph();  // generate the graph-based map
    gg.get_vertices();  // get all vertices info
    gg.get_edges();  // get all edges info
    gg.get_connections();  // get all connection info
    gg.get_vertex_number();  // get the vertices number
    gg.get_edge_number();   // get the edges number
    gg.get_adjacent("tkt_5");  // get the adjacent vertices

    return EXIT_SUCCESS;
}

