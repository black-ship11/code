#include <iostream>
using namespace std;

void selectionSort(int arr[], int n) {
    for(int i = 0; i <= n-2; i++) {
        int mini = i;
        for(int j = i+1; j < n; j++) { 
            if(arr[j] < arr[mini]) {
                mini = j;
            }
        }
        int temp = arr[mini];
        arr[mini] = arr[i];
        arr[i] = temp;
    }
}

int main() {
    int n;
    cout << "Enter array size :";
    cin >> n;
    int *arr = new int[n];
    cout << "Enter array elements : " << endl;
    for(int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    selectionSort(arr, n);

    cout << "Array after sorting :" << endl;
    for(int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }

    delete[] arr;
    return 0;
}
