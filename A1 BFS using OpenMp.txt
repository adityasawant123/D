#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

const int MAX = 100000;           // Maximum number of nodes
vector<int> graph[MAX];           // Adjacency list representation of the graph
bool visited[MAX];                // Visited array to keep track of visited nodes

// Parallel BFS function
void parallelBFS(int start) {
    queue<int> q;                 // Queue for BFS
    q.push(start);               // Start node is pushed to queue
    visited[start] = true;       // Mark the start node as visited

    cout << "Parallel BFS Traversal: ";

    // Run BFS until the queue is empty
    while (!q.empty()) {
        int qSize = q.size();     // Number of nodes at the current level
        vector<int> currentLevel; // Store nodes of the current level

        // Dequeue all nodes at the current level and print them
        for (int i = 0; i < qSize; i++) {
            int node = q.front();
            q.pop();
            cout << node << " ";
            currentLevel.push_back(node); // Store for parallel neighbor processing
        }

        vector<int> nextLevel; // Store the nodes to be visited in the next level

        // Process all neighbors of the current level nodes in parallel
        #pragma omp parallel for
        for (int i = 0; i < currentLevel.size(); i++) {
            int node = currentLevel[i]; // Current node

            // Traverse all adjacent nodes
            for (int j = 0; j < graph[node].size(); j++) {
                int adj = graph[node][j]; // Adjacent node
                bool needToVisit = false;

                // Ensure only one thread updates the visited array at a time
                #pragma omp critical
                {
                    if (!visited[adj]) {
                        visited[adj] = true;   // Mark as visited
                        needToVisit = true;    // Indicate it's a new node
                    }
                }

                // Add the node to nextLevel if it's newly visited
                if (needToVisit) {
                    #pragma omp critical
                    {
                        nextLevel.push_back(adj); // Collect for next level traversal
                    }
                }
            }
        }

        // Enqueue all nodes of the next level
        for (int i = 0; i < nextLevel.size(); i++) {
            q.push(nextLevel[i]);
        }
    }

    cout << endl;
}

int main() {
    int n, m, start_node;

    // Input: number of nodes, number of edges, and the starting node
    cout << "Enter No of Nodes, Edges, and Start Node: ";
    cin >> n >> m >> start_node;

    // Input edges of the graph
    cout << "Enter Pairs of Edges:\n";
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v); // Add edge u->v
        graph[v].push_back(u); // Add edge v->u (undirected graph)
    }

    // Initialize visited array to false in parallel
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    // Call the parallel BFS function
    parallelBFS(start_node);

    return 0;
}


// g++ -fopenmp openmp_example.cpp -o openmp_example
// ./openmp_example
// openmp_example.exe
