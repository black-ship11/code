#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <unordered_map>
#include <tuple>
using namespace std;

struct Node
{
    int x, y;
    int g, h;
    int f;

    Node(int x, int y, int g, int h) : x(x), y(y), g(g), h(h)
    {
        f = g + h;
    }

    bool operator>(const Node &other) const
    {
        return f > other.f;
    }
};

vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

int manhattanHeuristic(int x1, int y1, int x2, int y2)
{
    return abs(x1 - x2) + abs(y1 - y2);
}

vector<pair<int, int>> aStar(int startX, int startY, int goalX, int goalY, vector<vector<int>> &grid)
{
    int n = grid.size(), m = grid[0].size();
    priority_queue<Node, vector<Node>, greater<Node>> openList;
    unordered_map<int, unordered_map<int, bool>> closedList;
    unordered_map<int, unordered_map<int, pair<int, int>>> cameFrom;

    openList.push(Node(startX, startY, 0, manhattanHeuristic(startX, startY, goalX, goalY)));

    while (!openList.empty())
    {
        Node current = openList.top();
        openList.pop();

        if (current.x == goalX && current.y == goalY)
        {
            vector<pair<int, int>> path;
            int x = goalX, y = goalY;
            while (x != startX || y != startY)
            {
                path.push_back({x, y});
                tie(x, y) = cameFrom[x][y];
            }
            path.push_back({startX, startY});
            reverse(path.begin(), path.end());
            return path;
        }

        closedList[current.x][current.y] = true;

        for (auto &direction : directions)
        {
            int newX = current.x + direction.first;
            int newY = current.y + direction.second;

            if (newX >= 0 && newY >= 0 && newX < n && newY < m && grid[newX][newY] == 0 && !closedList[newX][newY])
            {
                int g = current.g + 1;
                int h = manhattanHeuristic(newX, newY, goalX, goalY);
                openList.push(Node(newX, newY, g, h));
                cameFrom[newX][newY] = {current.x, current.y};
            }
        }
    }

    return {};
}

int main()
{
    int n, m;
    cout << "Enter the grid size (rows x columns): ";
    cin >> n >> m;

    vector<vector<int>> grid(n, vector<int>(m, 0));
    cout << "Enter the grid (0 = open, 1 = blocked):\n";
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            cin >> grid[i][j];
        }
    }

    int startX, startY, goalX, goalY;
    cout << "Enter start coordinates (x y): ";
    cin >> startX >> startY;
    cout << "Enter goal coordinates (x y): ";
    cin >> goalX >> goalY;

    vector<pair<int, int>> path = aStar(startX, startY, goalX, goalY, grid);

    if (!path.empty())
    {
        cout << "Path found:\n";
        for (auto &p : path)
        {
            cout << "(" << p.first << ", " << p.second << ")\n";
        }
    }
    else
    {
        cout << "No path found.\n";
    }

    return 0;
}




// Enter the grid size (rows x columns): 5 5
// Enter the grid (0 = open, 1 = blocked):
// 0 0 0 0 0
// 0 1 1 0 0
// 0 0 0 1 0
// 0 1 0 0 0
// 0 0 0 0 0
// Enter start coordinates (x y): 0 0
// Enter goal coordinates (x y): 4 4


// Path found:
// (0, 0)
// (1, 0)
// (2, 0)
// (2, 1)
// (2, 2)
// (3, 2)
// (4, 2)
// (4, 3)
// (4, 4)
