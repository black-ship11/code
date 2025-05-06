#include <iostream>
#include<vector>
#include <algorithm>
using namespace std;

// Structure to represent a job
struct Job {
    int id;
    int deadline;
    int profit;
};

bool comparison(Job a, Job b) {
    return a.profit > b.profit;
}

void jobScheduling(Job arr[], int n) {
    sort(arr, arr + n, comparison);

    int maxDeadline = 0;
    for (int i = 0; i < n; i++) {
        maxDeadline = max(maxDeadline, arr[i].deadline);
    }

    vector<int> schedule(maxDeadline + 1, -1);
   
    int totalProfit = 0, jobsScheduled = 0;

    for (int i = 0; i < n; i++) {
        for (int j = arr[i].deadline; j > 0; j--) {
            if (schedule[j] == -1) {
                schedule[j] = arr[i].id;
                totalProfit += arr[i].profit;
                jobsScheduled++;
                break;
            }
        }
    }

    cout << "Scheduled Jobs: ";
    for (int i = 1; i <= maxDeadline; i++) {
        if (schedule[i] != -1) {
            cout << "J" << schedule[i] << " ";
        }
    }
   
    cout << "\nTotal Profit: " << totalProfit << endl;
}

int main() {
    Job arr[] = { {1, 2, 100}, {2, 1, 50}, {3, 2, 10}, {4, 1, 20}, {5, 3, 80} };
    int n = sizeof(arr) / sizeof(arr[0]);

    jobScheduling(arr, n);

    return 0;
}
