#include <iostream>
#include <limits.h> // For INT_MAX
using namespace std;

#define V 100 // Max vertices (adjust as needed)

// Function to find the vertex with the minimum key value, from the set of vertices not yet included in MST
int minKey(int key[], bool mstSet[], int n) {
    int min = INT_MAX, min_index;

    for (int v = 0; v < n; v++) {
        if (!mstSet[v] && key[v] < min) {
            min = key[v];
            min_index = v;
        }
    }
    return min_index;
}

// Prim's Algorithm to print MST stored in parent[]
void primMST(int graph[V][V], int n) {
    int parent[V];   // To store constructed MST
    int key[V];      // Key values used to pick minimum weight edge
    bool mstSet[V];  // To represent set of vertices included in MST

    // Initialize all keys as infinite and mstSet[] as false
    for (int i = 0; i < n; i++) {
        key[i] = INT_MAX;
        mstSet[i] = false;
    }

    key[0] = 0;       // Start from vertex 0
    parent[0] = -1;   // First node is always root of MST

    // Build MST
    for (int count = 0; count < n - 1; count++) {
        int u = minKey(key, mstSet, n);  // Pick the min key vertex
        mstSet[u] = true;                // Add to MST set

        // Update key and parent of adjacent vertices
        for (int v = 0; v < n; v++) {
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
        }
    }

    // Print the MST
    cout << "\nMinimum Spanning Tree edges:\n";
    int totalWeight = 0;
    for (int i = 1; i < n; i++) {
        cout << parent[i] << " - " << i << " : " << graph[i][parent[i]] << endl;
        totalWeight += graph[i][parent[i]];
    }
    cout << "Total weight: " << totalWeight << endl;
}

int main() {
    int n; // Number of vertices
    int graph[V][V] = {0};

    cout << "Enter number of vertices: ";
    cin >> n;

    cout << "Enter adjacency matrix (0 for no edge):\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> graph[i][j];
        }
    }

    // Run Prim's algorithm
    primMST(graph, n);

    return 0;
}



// Enter number of vertices: 5
// Enter adjacency matrix (0 for no edge):
// 0 2 0 6 0
// 2 0 3 8 5
// 0 3 0 0 7
// 6 8 0 0 9
// 0 5 7 9 0


// Minimum Spanning Tree edges:
// 0 - 1 : 2
// 1 - 2 : 3
// 0 - 3 : 6
// 1 - 4 : 5
// Total weight: 16
