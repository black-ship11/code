#include <iostream>
using namespace std;

// Maximum number of vertices and edges
#define MAX_V 100
#define MAX_E 100

// Structure for an edge
struct Edge {
    int u, v;       // Endpoints of the edge
    int weight;     // Weight of the edge
};

// Parent array for Union-Find
int parent[MAX_V];

// Find the root of a vertex
int find(int x) {
    while (parent[x] != x) {
        x = parent[x];
    }
    return x;
}

// Union of two sets
void unionSets(int u, int v) {
    parent[find(u)] = find(v);
}

// Sort edges by weight using bubble sort
void sortEdges(Edge edges[], int e) {
    for (int i = 0; i < e - 1; i++) {
        for (int j = 0; j < e - i - 1; j++) {
            if (edges[j].weight > edges[j + 1].weight) {
                Edge temp = edges[j];
                edges[j] = edges[j + 1];
                edges[j + 1] = temp;
            }
        }
    }
}

// Kruskal's Algorithm
void kruskal(Edge edges[], int V, int E) {
    // Initialize parent array
    for (int i = 0; i < V; i++) {
        parent[i] = i;
    }

    // Sort all edges by weight
    sortEdges(edges, E);

    // Array to store MST edges
    Edge mst[MAX_E];
    int mstCount = 0;
    int totalWeight = 0;

    // Process each edge
    for (int i = 0; i < E; i++) {
        int u = edges[i].u;
        int v = edges[i].v;
        int w = edges[i].weight;

        // Check if including this edge creates a cycle
        if (find(u) != find(v)) {
            mst[mstCount] = edges[i]; // Add edge to MST
            mstCount++;
            totalWeight += w;
            unionSets(u, v); // Combine the sets
        }
    }

    // Output the MST
    cout << "\nMinimum Spanning Tree edges:\n";
    for (int i = 0; i < mstCount; i++) {
        cout << mst[i].u << " - " << mst[i].v << ": " << mst[i].weight << endl;
    }
    cout << "Total weight: " << totalWeight << endl;
}

int main() {
    int V, E;
    Edge edges[MAX_E];

    // Input: number of vertices and edges
    cout << "Enter number of vertices: ";
    cin >> V;
    cout << "Enter number of edges: ";
    cin >> E;

    // Input: edges (u, v, weight)
    cout << "Enter edges (u v weight), 0-based index:\n";
    for (int i = 0; i < E; i++) {
        cin >> edges[i].u >> edges[i].v >> edges[i].weight;
    }

    // Run Kruskal's Algorithm
    kruskal(edges, V, E);

    return 0;
}


// Enter number of vertices: 4
// Enter number of edges: 5
// Enter edges (u v weight), 0-based index:
// 0 1 10
// 0 2 6
// 0 3 5
// 1 3 15
// 2 3 4


// Minimum Spanning Tree edges:
// 2 - 3: 4
// 0 - 3: 5
// 0 - 1: 10
// Total weight: 19
