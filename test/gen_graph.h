//
// This class supports to generate the graph-based representation for the saturn map.
// Author: Liang Zhang
// Email: liangzh@email.arizona.edu
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

    Graph generate_graph() {
        //        typedef std::pair<int, int> Edge;
        string file_name;

        json j_file_room;
        json j_file_portal;

        try {
            file_name = "../../test/data/saturn_room.json";
            std::ifstream in_room(file_name);
            if (!in_room) {
                std::cout << "Failed to open file" << endl;
            }
            j_file_room = json::parse(in_room);

            file_name = "../../test/data/saturn_portal.json";
            std::ifstream in_portal(file_name);
            if (!in_portal) {
                std::cout << "Failed to open file" << endl;
            }
            j_file_portal = json::parse(in_portal);
        }
        catch (std::exception& e) {
            std::cout << "Exception:" << endl;
            std::cout << e.what() << endl;
        }

        // process room data
        for (int i = 0; i < j_file_room.at("id").size(); i++) {
            string room_id = j_file_room.at("id")[i];
            process_id(room_id);

            this->node_id_list.push_back(room_id);

            string loc = j_file_room.at("loc")[i];
            vector<float> locs = process_loc(loc);

            string j_conn = j_file_room.at("connections")[i];
            vector<string> connections = process_connections(j_conn);
        }

        // get the number of rooms
        int room_number = this->node_id_list.size();

        // process portal data
        for (int i = 0; i < j_file_portal.at("id").size(); i++) {
            string portal_id = j_file_portal.at("id")[i];
            process_id(portal_id);

            this->node_id_list.push_back(portal_id);
            int current_node_no = this->node_id_list.size() - 1;

            string loc = j_file_portal.at("loc")[i];
            vector<float> locs = process_loc(loc);

            string j_conn = j_file_portal.at("connections")[i];
            vector<string> connections = process_connections(j_conn);
            for (int i = 0; i < connections.size(); i++) {
                for (int j = 0; j < room_number; j++) {
                    if (this->node_id_list[j] == connections[i]) {
                        this->edge_list.push_back(Edge(current_node_no, j));
                        break;
                    }
                }
            }
        }

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
        float transmission_delay[] = {};

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
//            make_graph_attributes_writer(graph_attr, vertex_attr, edge_attr));

        return g;
    }

    void get_vertices(Graph g){
        boost::property_map<Graph, vertex_index_t>::type vertex_id =
            get(vertex_index, g);
        std::cout << "vertices(g) = ";
        typedef graph_traits<Graph>::vertex_iterator vertex_iter;
        std::pair<vertex_iter, vertex_iter> vp;
        for (vp = vertices(g); vp.first != vp.second; ++vp.first)
            std::cout << this->node_id_list[get(vertex_id, *vp.first)] << " ";
        std::cout << std::endl;
    }

    void get_edges(Graph g){
        boost::property_map<Graph, vertex_index_t>::type vertex_id =
            get(vertex_index, g);
        std::cout << "edges(g) = ";
        graph_traits<Graph>::edge_iterator ei, ei_end;
        for (tie(ei, ei_end) = edges(g); ei != ei_end; ++ei)
            std::cout << "(" << this->node_id_list[get(vertex_id, source(*ei, g))] << ","
                      << this->node_id_list[get(vertex_id, target(*ei, g))] << ") ";
        std::cout << std::endl;
    }

    void get_connections(Graph g){
        std::for_each(
            vertices(g).first, vertices(g).second, exercise_vertex<Graph>(g, this->node_id_list));
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
