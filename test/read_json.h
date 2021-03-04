//
// Created by Liang Zhang on 3/1/21.
//

#ifndef TOMCAT_TMM_READ_JSON_H
#define TOMCAT_TMM_READ_JSON_H

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>

using json = nlohmann::json;
using namespace std;

class read_json {
  public:
    read_json(string diff) : difficulty(diff) {}
    ~read_json() {}

    string difficulty;

    json read_json_file() {
        string file_name;

        json j_file_room;
        json j_file_portal;
        json j_file_victim;

        try {
            file_name = "../../test/data/room_" + this->difficulty + ".json";
            std::ifstream in_room(file_name);
            if (!in_room) {
                std::cout << "Failed to open file" << endl;
            }
            j_file_room = json::parse(in_room);

            file_name = "../../test/data/portal_" + this->difficulty + ".json";
            std::ifstream in_portal(file_name);
            if (!in_portal) {
                std::cout << "Failed to open file" << endl;
            }
            j_file_portal = json::parse(in_portal);

            file_name = "../../test/data/victim_" + this->difficulty + ".json";
            std::ifstream in_victim(file_name);
            if (!in_victim) {
                std::cout << "Failed to open file" << endl;
            }
            j_file_victim = json::parse(in_victim);

        }
        catch (std::exception& e) {
            std::cout << "Exception:" << endl;
            std::cout << e.what() << endl;
            return 0;
        }

        // process room data
        for (int i = 0; i < j_file_room.at("id").size(); i++) {
            string id = j_file_room.at("id")[i];
            process_id(id);

            string loc = j_file_room.at("loc")[i];
            vector<float> locs = process_loc(loc);

            string j_conn = j_file_room.at("connections")[i];
            vector<string> connections = process_connections(j_conn);
        }

        // process portal data
        for (int i = 0; i < j_file_portal.at("id").size(); i++) {
            string id = j_file_portal.at("id")[i];
            process_id(id);

            string loc = j_file_portal.at("loc")[i];
            vector<float> locs = process_loc(loc);

            string j_conn = j_file_portal.at("connections")[i];
            vector<string> connections = process_connections(j_conn);

        }

        // process victim data
        for (int i = 0; i < j_file_victim.at("id").size(); i++) {
            string id = j_file_victim.at("id")[i];
            process_id(id);

            string loc = j_file_victim.at("loc")[i];
            vector<float> locs = process_loc(loc);

            string j_type = j_file_victim.at("type")[i];
            process_id(j_type);
        }

        return NULL;
    }

    void process_json() {
        json j_room, j_portal, j_victim;
        j_room = read_json_file();
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

#endif // TOMCAT_TMM_READ_JSON_H
