//
//  main.cpp
//  delta_stepping
//
//  Created by Jiayi on 2023/4/12.
//

#include <iostream>
#include <vector>
#include <set>
#include <limits>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
using namespace std;
int max_buckets;
const int INF = numeric_limits<int>::max();

struct Edge {
    int dest;
    int weight;
};
vector<vector<Edge>> generateGraph(int num_vertices, int num_edges_per_vertex) {
    srand(time(0));
    vector<vector<Edge>> graph(num_vertices);

    for (int i = 0; i < num_vertices; ++i) {
        set<int> used_destinations;
        used_destinations.insert(i); // Avoid self-loops

        while (used_destinations.size() <= num_edges_per_vertex) {
            int dest = rand() % num_vertices;
            if (used_destinations.find(dest) == used_destinations.end()) {
                graph[i].push_back({dest, rand() % 15 + 1});
                used_destinations.insert(dest);
            }
        }
    }

    return graph;
}

int get_sum_of_edge_weights(const vector<vector<Edge>>& graph) {
    int sum = 0;
    for (const auto& node_edges : graph) {
        for (const auto& edge : node_edges) {
            sum += edge.weight;
        }
    }
    return sum;
}
vector<vector<Edge>> readGraphFromFile(const string& filename) {
    vector<vector<Edge>> graph;
    ifstream fin(filename);
    string line;
    while (getline(fin, line)) {
        vector<Edge> edges;
        istringstream iss(line);
        char ch;
        while (iss >> ch) {
            int dest, weight;
            iss >> dest >> ch >> weight;
            edges.push_back({ dest, weight });
        }
        graph.push_back(edges);
    }
    return graph;
}

int get_edge_weight(const vector<vector<Edge>>& graph, int u, int v) {
    for (const Edge& edge : graph[u]) {
        if (edge.dest == v) {
            return edge.weight;
        }
    }
    return INF;
}

void relax(int u, int v, int weight, vector<int> &distances, int delta, vector<vector<set<int>>> &buckets, const vector<int> &node_thread_assignment) {
    int old_distance = distances[v];
    int new_distance = distances[u] + weight;

    if (new_distance < old_distance) {
        distances[v] = new_distance;
        int old_bucket = old_distance / delta;
        int new_bucket = new_distance / delta;
        int v_tid = node_thread_assignment[v];

        if (old_distance != INF) {
            buckets[v_tid][old_bucket].erase(v);
        }
        buckets[v_tid][new_bucket].insert(v);
    }
}


void delta_stepping(int source, vector<vector<Edge>>& graph, vector<int>& distances, int delta,int num_of_threads) {
    int n = graph.size();
    distances.assign(n, INF);
    distances[source] = 0;

    vector<vector<set<int>>> buckets(num_of_threads, vector<set<int>>(max_buckets));
    vector<int> node_thread_assignment(n);

    // Assigning nodes to threads and putting the source node in the appropriate bucket
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        node_thread_assignment[i] = i % num_of_threads;
        if (i == source) {
            int initial_bucket = 0;
            buckets[node_thread_assignment[i]][initial_bucket].insert(i);
        }
    }

    // Processing light and heavy edges separately
    bool non_empty_bucket_found = true;
    while (non_empty_bucket_found) {
        non_empty_bucket_found = false;
//        #pragma omp parallel for
        for (int t = 0; t < num_of_threads; ++t) {
            for (int k = 0; k < max_buckets; ++k) {
                    if (buckets[t][k].empty()) {
                            continue;
                    }
                    non_empty_bucket_found = true;
            

            vector<vector<vector<pair<int, int>>>> light_requests(num_of_threads, vector<vector<pair<int, int>>>(num_of_threads));
            vector<vector<vector<pair<int, int>>>> heavy_requests(num_of_threads, vector<vector<pair<int, int>>>(num_of_threads));

            // Loop 1: Partitioning edges based on their weight and assigned threads
                for (int u_index = 0; u_index < buckets[t][k].size(); u_index++)
                {
                    int u = *next(buckets[t][k].begin(), u_index);
                    int u_tid = node_thread_assignment[u];
                    for (const Edge& edge : graph[u]) {
                        int v = edge.dest;
                        int v_tid = node_thread_assignment[v];
                        int weight = edge.weight;

                        if (weight <= delta) {
                            light_requests[u_tid][v_tid].push_back({u, v});
                        } else {
                            heavy_requests[u_tid][v_tid].push_back({u, v});
                        }
                    }
            }

            buckets[t][k].clear();

            // Loop 2: Relaxing light edges
            #pragma omp parallel num_threads(num_of_threads)
            {
                #pragma omp for collapse(2)
                for (int t = 0; t < num_of_threads; ++t) {
                    for (int r = 0; r < num_of_threads; ++r) {
                        for (const auto& edge : light_requests[t][r]) {
                            int u = edge.first;
                            int v = edge.second;
                            int weight = get_edge_weight(graph, u, v);
                            relax(u, v, weight, distances, delta, buckets,node_thread_assignment);
                        }
                        light_requests[t][r].clear();
                    }
                }
                    // Loop 3: Relaxing heavy edges
                   
                #pragma omp for collapse(2)
                for (int t = 0; t < num_of_threads; ++t) {
                            for (int r = 0; r < num_of_threads; ++r) {
                                for (const auto& edge : heavy_requests[t][r]) {
                                    int u = edge.first;
                                    int v = edge.second;
                                    int weight = get_edge_weight(graph, u, v);
                                    relax(u, v, weight, distances, delta, buckets,node_thread_assignment);
                                }
                                heavy_requests[t][r].clear();
                            }
                        }
            }
            

           
            }
        }
    }
}


int main(int argc, char** argv) {
    if(argc < 3) {
        cout << "usage: ./delta_stepping <number of threads> <delta> <vertex number>" << endl;
        return 1;
    }
    double delta = 5;
    int num_of_threads=2;
//    vector<vector<Edge>> graph = {
//        { {1, 10}, {2, 5} },
//        { {3, 1}, {2, 2} },
//        { {1, 3}, {3, 9}, {4, 2} },
//        { {4, 4} },
//        { {3, 6},{0,7} }
//    };
    num_of_threads=atoi(argv[1]);
    delta=atoi(argv[2]);
    int vertex_number=atoi(argv[3]);
    if(delta<1){cout << "delta needs to be greater than 1" << endl; return 1;}
    omp_set_num_threads(num_of_threads);
    vector<vector<Edge>> graph = generateGraph(vertex_number,5);
    vector<int> distances;
    cout<<num_of_threads<<" threads, "<<"delta="<<delta<<endl;
    int edge_sum=0;
    edge_sum=get_sum_of_edge_weights(graph);
    max_buckets=ceil(edge_sum / delta);
    
    double t_start = omp_get_wtime();
    delta_stepping(0, graph, distances, delta, num_of_threads);
    double time_taken = omp_get_wtime() - t_start;
    printf("Time taken for delta-stepping: %f\n", time_taken);
//    for (int i=0; i < distances.size(); ++i) {
//        cout << "Shortest distance from node 0 to node " << i << ": " << distances[i] << endl;
//    }

    return 0;
}







