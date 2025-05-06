#include <iostream>
#include <vector>
using namespace std;

int solutionCount = 0;

void printBoard(vector<int> &board, int n) {
    cout << "\nSolution " << ++solutionCount << ":\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i] == j)
                cout << "Q ";
            else
                cout << ". ";
        }
        cout << endl;
    }
}

bool isSafe(vector<int> &board, int row, int col) {
    for (int i = 0; i < row; i++) {
        if (board[i] == col || abs(board[i] - col) == abs(i - row))
            return false;
    }
    return true;
}

void solveNQueens(vector<int> &board, int row, int n) {
    if (row == n) {
        printBoard(board, n);
        return;
    }

    for (int col = 0; col < n; col++) {
        if (isSafe(board, row, col)) {
            board[row] = col;
            solveNQueens(board, row + 1, n);
        }
    }
}

int main() {
    int n;
    cout << "Enter the value of N for N-Queens: ";
    cin >> n;

    vector<int> board(n);
    solveNQueens(board, 0, n);

    if (solutionCount == 0)
        cout << "No solution exists for N = " << n << endl;
    else
        cout << "\nTotal Solutions Found: " << solutionCount << endl;

    return 0;
}
