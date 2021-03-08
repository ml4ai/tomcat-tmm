//
// This class supports to generate the graph-based representation for the saturn
// map. Author: Liang Zhang Email: liangzh@email.arizona.edu
//

#ifndef TOMCAT_TMM_GEN_GRAPH_H
#define TOMCAT_TMM_GEN_GRAPH_H

#include "json.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>

using json = nlohmann::json;
using namespace std;
using namespace boost;

class gen_graph {
  public:
    gen_graph() {}
    ~gen_graph() {}

    typedef adjacency_list<vecS,
                           vecS,
                           undirectedS,
                           no_property,
                           property<edge_weight_t, float>>
        Graph;

    typedef std::pair<int, int> Edge;
    vector<string> node_id_list;
    vector<Edge> edge_list;
    int room_number = 0;
    int portal_number = 0;
    string json_file_name = "../../data/Saturn_1.0_sm_v1.0.json";
    Graph graph;

    template <class Graph> struct exercise_vertex {
      private:
        Graph g;
        vector<string> node_id_list;

      public:
        exercise_vertex(Graph& g_, vector<string> node_id_list_) {
            g = g_;
            node_id_list = node_id_list_;
        }
        typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

        void operator()(const Vertex& v) const {
            using namespace boost;
            typename property_map<Graph, vertex_index_t>::type vertex_id =
                get(vertex_index, g);
            std::cout << "vertex: " << get(vertex_id, v) << std::endl;
            std::cout << "id: " << node_id_list[get(vertex_id, v)] << std::endl;

            // Write out the outgoing edges
            std::cout << "\tout-edges: ";
            typename graph_traits<Graph>::out_edge_iterator out_i, out_end;
            typename graph_traits<Graph>::edge_descriptor e;
            for (tie(out_i, out_end) = out_edges(v, g); out_i != out_end;
                 ++out_i) {
                e = *out_i;
                Vertex src = source(e, g), targ = target(e, g);
                std::cout << "(" << node_id_list[get(vertex_id, src)] << ","
                          << node_id_list[get(vertex_id, targ)] << ") ";
            }
            std::cout << std::endl;

            // Write out the incoming edges
            std::cout << "\tin-edges: ";
            typename graph_traits<Graph>::in_edge_iterator in_i, in_end;
            for (tie(in_i, in_end) = in_edges(v, g); in_i != in_end; ++in_i) {
                e = *in_i;
                Vertex src = source(e, g), targ = target(e, g);
                std::cout << "(" << node_id_list[get(vertex_id, src)] << ","
                          << node_id_list[get(vertex_id, targ)] << ") ";
            }
            std::cout << std::endl;

            // Write out all adjacent vertices
            std::cout << "\tadjacent vertices: ";
            typename graph_traits<Graph>::adjacency_iterator ai, ai_end;
            for (tie(ai, ai_end) = adjacent_vertices(v, g); ai != ai_end; ++ai)
                std::cout << node_id_list[get(vertex_id, *ai)] << " ";
            std::cout << std::endl;
        }
    };

    // saturn map parser
    void json_parser() {
        json j_file;
        try {
            std::ifstream in(this->json_file_name);
            if (!in) {
                std::cout << "Failed to open file" << endl;
            }
            j_file = json::parse(in);
        }
        catch (std::exception& e) {
            std::cout << "Exception:" << endl;
            std::cout << e.what() << endl;
        }

        // process room data
        for (int i = 0; i < j_file.at("locations").size(); i++) {
            if (!j_file.at("locations")[i].contains("child_locations")) {
                string room_id = j_file.at("locations")[i].at("id");
                this->node_id_list.push_back(room_id);
                this->room_number++;
            }
        }

        // process portal data
        for (int i = 0; i < j_file.at("connections").size(); i++) {
            string portal_id = j_file.at("connections")[i].at("id");
            this->node_id_list.push_back(portal_id);
            this->portal_number++;

            int current_node_no = this->node_id_list.size() - 1;

            int conn_num = j_file.at("connections")[i].at("connected_locations").size();

            for (int j = 0; j < conn_num; j++) {
                for (int k = 0; k < this->room_number; k++) {
                    if (this->node_id_list[k] == j_file.at("connections")[i].at("connected_locations")[j]) {
                        this->edge_list.push_back(Edge(current_node_no, k));
                        break;
                    }
                }
            }
        }
    }

    Graph generate_graph() {
        //        typedef std::pair<int, int> Edge;
        string file_name;
        json_parser();

        int node_number = this->node_id_list.size();
        int node_no[node_number];
        for (int i = 0; i < node_number; i++) {
            node_no[i] = i;
        }

        string node_name[node_number];
        for (int i = 0; i < node_number; i++) {
            node_name[i] = this->node_id_list[i];
        }

        int edge_number = this->edge_list.size();
        Edge edge_array[edge_number];
        for (int i = 0; i < edge_number; i++) {
            edge_array[i] = this->edge_list[i];
        }

        // time cost for each transition
        float transmission_delay[edge_number];

// declare a graph object, adding the edges and edge properties
#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
        // VC++ can't handle the iterator constructor
        Graph g(node_number);
        property_map<Graph, edge_weight_t>::type weightmap =
            get(edge_weight, g);
        for (std::size_t j = 0; j < edge_number; ++j) {
            graph_traits<Graph>::edge_descriptor e;
            bool inserted;
            tie(e, inserted) =
                add_edge(edge_array[j].first, edge_array[j].second, g);
            weightmap[e] = transmission_delay[j];
        }
#else
        Graph g(edge_array,
                edge_array + edge_number,
                transmission_delay,
                node_number);
#endif

        boost::property_map<Graph, vertex_index_t>::type vertex_id =
            get(vertex_index, g);
        boost::property_map<Graph, edge_weight_t>::type trans_delay =
            get(edge_weight, g);

        //  uncomment this section to help generate a graph if needed
        //        boost::write_graphviz(
        //            std::cout,
        //            g,
        //            make_label_writer(node_name),
        //            make_label_writer(trans_delay),
        //            make_graph_attributes_writer(graph_attr, vertex_attr,
        //            edge_attr));
        this->graph = g;
        return g;
    }

    void get_vertices() {
        Graph g = this->graph;
        boost::property_map<Graph, vertex_index_t>::type vertex_id =
            get(vertex_index, g);
        std::cout << "vertices(g) = ";
        typedef graph_traits<Graph>::vertex_iterator vertex_iter;
        std::pair<vertex_iter, vertex_iter> vp;
        for (vp = vertices(g); vp.first != vp.second; ++vp.first)
            std::cout << this->node_id_list[get(vertex_id, *vp.first)] << " ";
        std::cout << std::endl;
    }

    void get_edges() {
        Graph g = this->graph;
        boost::property_map<Graph, vertex_index_t>::type vertex_id =
            get(vertex_index, g);
        std::cout << "edges(g) = ";
        graph_traits<Graph>::edge_iterator ei, ei_end;
        for (tie(ei, ei_end) = edges(g); ei != ei_end; ++ei)
            std::cout << "("
                      << this->node_id_list[get(vertex_id, source(*ei, g))]
                      << ","
                      << this->node_id_list[get(vertex_id, target(*ei, g))]
                      << ") ";
        std::cout << std::endl;
    }

    void get_connections() {
        Graph g = this->graph;
        std::for_each(vertices(g).first,
                      vertices(g).second,
                      exercise_vertex<Graph>(g, this->node_id_list));
    }

    void get_vertex_number(){
        std::cout << this->node_id_list.size() << std::endl;
    }

    void get_edge_number(){
        std::cout << this->edge_list.size() << std::endl;
    }

    void get_adjacent(string id){
        int id_idx = -1;
        for (int i = 0; i < this->node_id_list.size();i++){
            if (this->node_id_list[i] == id){
                id_idx = i;
                break;
            }
        }
        if (id_idx == -1){
            std::cout << "no connections to this node!" << std::endl;
            return;
        }
        else {
            std::cout << "connections: " << std::endl;
            for (int i = 0; i < this->edge_list.size(); i++) {
                if (this->edge_list[i].first == id_idx) {
                    std::cout << this->node_id_list[this->edge_list[i].second] << std::endl;
                }
                if (this->edge_list[i].second == id_idx) {
                    std::cout << this->node_id_list[this->edge_list[i].first] << std::endl;
                }
            }
        }
    }


    vector<float> process_loc(string loc) {
        loc.erase(std::remove(loc.begin(), loc.end(), '('), loc.end());
        loc.erase(std::remove(loc.begin(), loc.end(), ')'), loc.end());
        loc.erase(std::remove(loc.begin(), loc.end(), ','), loc.end());
        istringstream istr1(loc); // istr1 will read from str
        float locs[2];
        istr1 >> locs[0] >> locs[1];
        vector<float> ret;
        ret.push_back(locs[0]);
        ret.push_back(locs[1]);
        return ret;
    }

    void* process_id(string id) {
        id.erase(std::remove(id.begin(), id.end(), '"'), id.end());
    }

    vector<string> process_connections(string connections) {
        connections.erase(
            std::remove(connections.begin(), connections.end(), '"'),
            connections.end());
        connections.erase(
            std::remove(connections.begin(), connections.end(), '['),
            connections.end());
        connections.erase(
            std::remove(connections.begin(), connections.end(), ']'),
            connections.end());
        connections.erase(
            std::remove(connections.begin(), connections.end(), '\''),
            connections.end());
        connections.erase(
            std::remove(connections.begin(), connections.end(), ','),
            connections.end());
        vector<string> ret = split(connections, " ");
        return ret;
    }

    vector<string> split(const string& str, const string& pattern) {
        vector<string> ret;
        if (pattern.empty())
            return ret;
        size_t start = 0, index = str.find_first_of(pattern, 0);
        while (index != str.npos) {
            if (start != index)
                ret.push_back(str.substr(start, index - start));
            start = index + 1;
            index = str.find_first_of(pattern, start);
        }
        if (!str.substr(start).empty())
            ret.push_back(str.substr(start));
        return ret;
    }
};

#endif // TOMCAT_TMM_GEN_GRAPH_H
