// Assignment 1b - Sequential and Parallel DFS using OpenMP

#include <iostream>
#include <vector>
#include <omp.h> // OpenMP library

using namespace std;

const int MAX = 100000;
vector<int> graph[MAX]; // Adjacency list
bool visited[MAX];      // Shared visited array for both DFS

// Sequential DFS function
void sequentialDFS(int node) {
    visited[node] = true;
    cout << node << " ";

    for (int i = 0; i < graph[node].size(); i++) {
        int adj_node = graph[node][i];
        if (!visited[adj_node]) {
            sequentialDFS(adj_node);
        }
    }
}

// Parallel DFS function using OpenMP
void parallelDFS(int node) {
    #pragma omp critical
    {
        if (visited[node]) return;
        visited[node] = true;
        cout << node << " ";
    }

    #pragma omp parallel for
    for (int i = 0; i < graph[node].size(); i++) {
        int adj_node = graph[node][i];
        if (!visited[adj_node]) {
            parallelDFS(adj_node);
        }
    }
}

int main() {
    int n, m, start_node, choice;

    cout << "Enter No of Nodes, Edges, and Start Node: ";
    cin >> n >> m >> start_node;

    cout << "Enter Pairs of Edges:\n";
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u); // Undirected graph
    }

    cout << "\nChoose Traversal Method:\n";
    cout << "1. Sequential DFS\n";
    cout << "2. Parallel DFS\n";
    cout << "Enter your choice: ";
    cin >> choice;

    // Initialize visited array
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    if (choice == 1) {
        cout << "Sequential DFS Traversal: ";
        sequentialDFS(start_node);
    } else if (choice == 2) {
        cout << "Parallel DFS Traversal: ";
        parallelDFS(start_node);
    } else {
        cout << "Invalid choice!" << endl;
    }

    cout << endl;
    return 0;
}

