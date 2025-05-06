#include <iostream>
#include <queue>

using namespace std;

void dfs_recursive(int adj_mat[10][10], int n, int node, int visited[10]) {
    cout << node << " --> ";
    visited[node] = 1;
    for (int j = 0; j < n; j++) {
        if (adj_mat[node][j] == 1 && visited[j] == 0) {
            dfs_recursive(adj_mat, n, j, visited);
        }
    }
}

void bfs(int adj_mat[10][10], int n, int start_node, int visited[10]) {
    queue<int> q;
    q.push(start_node);
    visited[start_node] = 1;

    while (!q.empty()) {
        int i = q.front();
        q.pop();
        cout << i << " --> ";

        for (int j = 0; j < n; j++) {
            if (adj_mat[i][j] == 1 && visited[j] == 0) {
                q.push(j);
                visited[j] = 1;
            }
        }
    }
}

int main() {
    int n;
    int adj_mat[10][10] = {0};
    int visited[10] = {0};

    cout << "Enter the total number of nodes in the graph: ";
    cin >> n;

    // Input edges
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            char choice;
            cout << "Is there an edge between " << i << " and " << j << " (Y/N)? ";
            cin >> choice;
            if (choice == 'Y' || choice == 'y') {
                adj_mat[i][j] = adj_mat[j][i] = 1;
            }
        }
    }

    // DFS Traversal
    for (int i = 0; i < n; i++) {
        visited[i] = 0;
    }
    cout << "\nDFS traversal starting from node 0:\n";
    dfs_recursive(adj_mat, n, 0, visited);
    cout << "NULL" << endl;

    // BFS Traversal
    for (int i = 0; i < n; i++) {
        visited[i] = 0;
    }
    cout << "\nBFS traversal starting from node 0:\n";
    bfs(adj_mat, n, 0, visited);
    cout << "NULL" << endl;

    return 0;
}
