
## 1️⃣ ML: Gradient Descent
```python
# Paste your Gradient Descent code here
# Gradient Descent to find the local minima of y = (x + 3)^2
# Simple and beginner-friendly version

import numpy as np
import matplotlib.pyplot as plt

# Define the function y = (x + 3)^2
def f(x):
    return (x + 3)**2

# Define its derivative (gradient)
def df(x):
    return 2 * (x + 3)

# --- Algorithm Parameters ---
x = 2              # starting point
learning_rate = 0.1
iterations = 30

print("---- Gradient Descent Execution ----")
print(f"Starting point: x = {x}")
print(f"Learning rate: {learning_rate}")
print(f"Number of iterations: {iterations}\n")

# Store steps for visualization
x_values = [x]

# --- Gradient Descent Loop ---
for i in range(iterations):
    grad = df(x)
    x = x - learning_rate * grad
    x_values.append(x)
    print(f"Iteration {i+1:2}: x = {x:.6f}, f(x) = {f(x):.6f}")

print("\n---- Result ----")
print(f"Local minima found at x = {x:.6f}")
print(f"Value of the function (minimum y) = {f(x):.6f}")

# --- Visualization ---
# Create smooth curve of the function
x_plot = np.linspace(-8, 3, 100)
y_plot = f(x_plot)

# Plot the function and gradient descent steps
plt.plot(x_plot, y_plot, label="y = (x + 3)²")
plt.scatter(x_values, [f(i) for i in x_values], color="red", label="Descent steps")
plt.title("Gradient Descent to find Local Minima")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

````

---

## 2️⃣ ML: Email Classification (KNN + SVM)

```python
# Paste your Email Classification code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("emails.csv")
print("Dataset loaded successfully ✅")
print(df.head())
print("\nDataset shape:", df.shape)
df.isna().sum()

plt.figure(figsize=(5,4))
df['Prediction'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("Distribution of Email Types")
plt.xlabel("Email Class (0 = Not Spam, 1 = Spam)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(5,4))
plt.hist(df["free"], bins=20, color='blue', alpha=0.7)
plt.title("Histogram of word: 'free'")
plt.xlabel("Frequency Count")
plt.ylabel("Number of Emails")
plt.show()
# Drop unnecessary column
df.drop("Email No.", axis=1, inplace=True)

# Separate features (X) and target (y)
X = df.iloc[:, :-1]      # all word-frequency columns
y = df.iloc[:, -1]       # Prediction column (1 = Spam, 0 = Not Spam)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numeric features (important for KNN & SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data cleaned, split, and scaled successfully ✅")


# Simple KNN Concept Plot (Dummy Visualization)
plt.figure(figsize=(5,5))
# Fake points just for explanation
plt.scatter([1,2,1.5], [2,1,1.2], color='green', label='Not Spam')
plt.scatter([4,5,4.5], [5,4,4.2], color='red', label='Spam')
# New email point
plt.scatter(3,3, color='black', label='New Email', s=100)
plt.title("KNN Concept Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()



knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print("\n--- K-Nearest Neighbors (KNN) ---")
print("Accuracy:", round(knn_acc*100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("Classification Report:\n", classification_report(y_test, knn_pred))



# Simple SVM Concept Plot (Dummy Visualization)
plt.figure(figsize=(5,5))
# Fake dataset points
plt.scatter([1,2,1.5], [2,1,1.2], color='green', label='Not Spam')
plt.scatter([4,5,4.5], [5,4,4.2], color='red', label='Spam')
# A simple separating line
plt.plot([0,6], [0,6], color='blue', linestyle='--', label='SVM Decision Boundary')
plt.title("SVM Decision Boundary Concept")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()



svm = SVC(kernel='rbf', gamma='auto', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print("\n--- Support Vector Machine (SVM) ---")
print("Accuracy:", round(svm_acc*100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("Classification Report:\n", classification_report(y_test, svm_pred))

```

---

## 3️⃣ ML: KNN on Diabetes Dataset

```python
# Paste your Diabetes KNN code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
df = pd.read_csv("diabetes.csv")
print(df.head())
print("Dataset Shape:", df.shape)

plt.bar(df['Outcome'].value_counts().index,
        df['Outcome'].value_counts().values,
        color=['green','red'])

plt.title("Count of Output Classes")
plt.xlabel("No Diabetes (0), Diabetes (1)")
plt.ylabel("Number of Patients")
plt.show()

cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for c in cols:
    df[c] = df[c].replace(0, df[c].mean())
X = df.drop('Outcome', axis=1)   # All columns except Outcome
y = df['Outcome']                # Target column
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("\nAccuracy:", round(accuracy,3))
print("Error Rate:", round(error_rate,3))
print("Precision:", round(precision,3))
print("Recall:", round(recall,3))
k_values = [3, 5, 7]
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, pred))

plt.plot(k_values, scores, marker='o')
plt.title("KNN Accuracy vs K")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
```

---

## 4️⃣ ML: K-Means Clustering

```python
# Paste your K-Means Clustering code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df = pd.read_csv("sales_data_sample.csv", encoding='latin1')

print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Select only numeric columns needed for clustering
features = ["SALES", "QUANTITYORDERED", "PRICEEACH", "MSRP"]
df_cluster = df[features]
plt.figure(figsize=(6,4))
plt.scatter(df_cluster["SALES"], df_cluster["MSRP"], color='blue', alpha=0.4)
plt.title("Sales vs MSRP (Before Clustering)")
plt.xlabel("SALES")
plt.ylabel("MSRP")
plt.show()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)
wcss = []   # Within-Cluster Sum of Squares

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7,4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title("Elbow Method to Determine Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nSample Data with Assigned Clusters:")
print(df[["SALES", "QUANTITYORDERED", "PRICEEACH", "MSRP", "Cluster"]].head())
plt.figure(figsize=(7,5))
sns.scatterplot(
    x=df["SALES"],
    y=df["MSRP"],
    hue=df["Cluster"],
    palette='Set2',
    alpha=0.8
)
plt.title("K-Means Clustering Visualization (K = 3)")
plt.xlabel("SALES")
plt.ylabel("MSRP")
plt.legend(title="Cluster")
plt.show()
```

---

## 5️⃣ ML: Uber Ride Fare Prediction

```python
# Paste your Uber Ride Prediction code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("uber.csv")
df.head()
df.isna().sum()
# Drop useless columns
df.drop(["Unnamed: 0", "key"], axis=1, inplace=True)

# Drop missing rows
df.dropna(inplace=True)

# Convert date column to datetime
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# Remove rows with negative fares
df = df[df["fare_amount"] > 0]
df.info()

plt.boxplot(df["fare_amount"])
plt.title("Outliers in Fare Amount")
plt.show()

# Remove extreme values (top & bottom 1%)
q_low = df["fare_amount"].quantile(0.01)
q_hi = df["fare_amount"].quantile(0.99)
df = df[(df["fare_amount"] > q_low) & (df["fare_amount"] < q_hi)]

# Check cleaned data summary
df.describe()

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Separate features (X) and target (y)
X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

# Convert datetime to numeric values
X["pickup_datetime"] = pd.to_numeric(pd.to_datetime(X["pickup_datetime"]))

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
r2_lr = r2_score(y_test, pred_lr)

print("Linear Regression → RMSE:", round(rmse_lr, 3))
print("Linear Regression → R²:", round(r2_lr, 3))

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
pred_rf = rf.predict(x_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
r2_rf = r2_score(y_test, pred_rf)

print("Random Forest → RMSE:", round(rmse_rf, 3))
print("Random Forest → R²:", round(r2_rf, 3))

# ✅ Added accuracy line
accuracy_rf = rf.score(x_test, y_test) * 100
print("Random Forest → Accuracy:", round(accuracy_rf, 2), "%")

plt.scatter(y_test, pred_rf, color='blue', alpha=0.4)
plt.title("Predicted vs Actual Fare (Random Forest)")
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.show()

```

---

## 6️⃣ DAA: N-Queens (Backtracking)

```cpp
// Paste your N-Queens Backtracking code here
#include <iostream>
#include <vector>   
#include <cmath>    
using namespace std;

using namespace std;

int solutionCount = 0;

void printBoard(vector<int> &board, int n) {
    cout << "Solution: " << ++solutionCount << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i] == j)
                cout << "Q ";
            else
                cout << ". ";
        }
        cout << endl;
    }
    cout << endl;
}

bool isSafe(vector<int> &board, int row, int col) {
    for (int i = 0; i < row; i++) {
        if (board[i] == col || abs(board[i] - col) == abs(i - row))
            return false;
    }
    return true;
}

void solveQueens(vector<int> &board, int row, int n) {
    if (row == n) {
        printBoard(board, n);
        return;
    }

    for (int col = 0; col < n; col++) {
        if (isSafe(board, row, col)) {
            board[row] = col;
            solveQueens(board, row + 1, n);   // ✔ FIXED
        }
    }
}

int main() {
    int n;
    cout << "Enter number of n for n queens: ";
    cin >> n;

    vector<int> board(n);
    solveQueens(board, 0, n);

    if (solutionCount == 0)
        cout << "No solution found." << endl;
    else
        cout << "Total solutions: " << solutionCount << endl;

    return 0;
}

/*
TIME COMPLEXITY (Backtracking):
--------------------------------
Worst Case: O(N!)
Reason:
- In row 1 → N choices
- In row 2 → N-1 choices (due to conflicts)
- In row 3 → N-2 choices
- ...
- Total ≈ N * (N-1) * (N-2) ... ≈ N!

SPACE COMPLEXITY:
-----------------
O(N)
Reason:
- Board stores 1 queen position per row → N size
- Recursion depth = N
So total auxiliary space = O(N)
*/
```

---

## 7️⃣ DAA: 0/1 Knapsack (DP)

```cpp
// Paste your 0/1 Knapsack DP code here
#include <iostream>
#include <vector>
using namespace std;

/*
0/1 KNAPSACK (Dynamic Programming)

TIME COMPLEXITY:
----------------
O(n * W)
Because we fill an n x W DP table.

SPACE COMPLEXITY:
-----------------
O(n * W)
Because DP table size is (n+1) x (W+1)

Where:
n = number of items
W = capacity of knapsack
*/

int knapsack(int W, vector<int> &wt, vector<int> &val, int n) {

    // Create DP table of size (n+1) x (W+1)
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    // Fill DP table (bottom-up)
    for (int i = 1; i <= n; i++) {          // for each item
        for (int w = 1; w <= W; w++) {      // for each weight
            if (wt[i - 1] <= w) {
                // Choose max of: take item OR don't take
                dp[i][w] = max(val[i - 1] + dp[i - 1][w - wt[i - 1]],
                               dp[i - 1][w]);
            } else {
                // Cannot include item
                dp[i][w] = dp[i - 1][w];
            }
        }
    }

    return dp[n][W];   // Final answer
}

int main() {
    int n, W;

    cout << "Enter number of items: ";
    cin >> n;

    vector<int> wt(n), val(n);

    cout << "Enter weight of each item:\n";
    for (int i = 0; i < n; i++) {
        cin >> wt[i];
    }

    cout << "Enter value of each item:\n";
    for (int i = 0; i < n; i++) {
        cin >> val[i];
    }

    cout << "Enter capacity of knapsack: ";
    cin >> W;

    int ans = knapsack(W, wt, val, n);

    cout << "Maximum value in 0/1 Knapsack = " << ans << endl;

    return 0;
}

```

---

## 8️⃣ DAA: Fractional Knapsack (Greedy)

```cpp
// Paste your Fractional Knapsack code here
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Structure to store item info
struct Item
{
    int weight;
    int value;
};

// Comparator based on value/weight ratio
bool cmp(Item a, Item b)
{
    double r1 = (double)a.value / a.weight;
    double r2 = (double)b.value / b.weight;
    return r1 > r2; // sort descending
}

double fractionalKnapsack(int capacity, vector<Item> &items)
{
    sort(items.begin(), items.end(), cmp); // sort by ratio

    double totalValue = 0.0;

    for (int i = 0; i < items.size(); i++)
    {

        if (capacity == 0)
            break;

        if (items[i].weight <= capacity)
        {
            totalValue += items[i].value;
            capacity -= items[i].weight;
        }
        else
        {
            double fraction = (double)capacity / items[i].weight;
            totalValue += items[i].value * fraction;
            capacity = 0; // knapsack full
        }
    }

    return totalValue;
}

int main()
{
    int n, capacity;

    cout << "Enter number of items: ";
    cin >> n;

    vector<Item> items(n);

    cout << "Enter weight of each item:\n";
    for (int i = 0; i < n; i++)
    {
        cin >> items[i].weight;
    }

    cout << "Enter value of each item:\n";
    for (int i = 0; i < n; i++)
    {
        cin >> items[i].value;
    }

    cout << "Enter knapsack capacity: ";
    cin >> capacity;

    double maxValue = fractionalKnapsack(capacity, items);

    cout << "Maximum value in fractional knapsack = " << maxValue << endl;

    return 0;
}


/*
TIME COMPLEXITY:
----------------
Sorting items by ratio: O(n log n)
Taking items: O(n)

Total: O(n log n)

SPACE COMPLEXITY:
-----------------
O(1) extra space (only a few variables)
*/


```

---

## 9️⃣ DAA: Fibonacci (Recursive + Iterative)

```cpp
// Paste your Fibonacci code here
#include <iostream>
using namespace std;

// ------------------------------
// Recursive Fibonacci Function
// ------------------------------
// Time Complexity:  O(2^n)
//   Because each call splits into 2 calls.
// Space Complexity: O(n)
//   Due to recursion call stack depth.
// ------------------------------
int fib_recursive(int n) {
    if (n <= 1)
        return n;
    return fib_recursive(n - 1) + fib_recursive(n - 2);
}

// ------------------------------
// Non-Recursive (Iterative) Fibonacci
// ------------------------------
// Time Complexity:  O(n)
//   Single loop from 2 to n.
// Space Complexity: O(1)
//   Uses constant variables only.
// ------------------------------
int fib_iterative(int n) {
    if (n <= 1)
        return n;

    int a = 0, b = 1, next = 0;
    for (int i = 2; i <= n; i++) {
        next = a + b;
        a = b;
        b = next;
    }
    return b;
}

int main() {
    int n;
    cout << "Enter n: ";
    cin >> n;

    cout << "\nRecursive Fibonacci of " << n << " = " 
         << fib_recursive(n) << endl;

    cout << "Iterative Fibonacci of " << n << " = " 
         << fib_iterative(n) << endl;

    return 0;
}

```

---

## 🔟 BLT: Student Data Contract

```solidity
// Paste your Student Data Solidity contract here
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {
    struct Student {
        uint id;
        string name;
        uint age;
    }

    Student[] public students;

    function addStudent(uint _id, string memory _name, uint _age) public {
        students.push(Student(_id, _name, _age));
    }

    function getStudentCount() public view returns (uint) {
        return students.length;
    }

    function getStudent(uint _index) public view returns (Student memory) {
        require(_index < students.length, "Invalid index");
        return students[_index];
    }

    function getAllStudents() external view returns (Student[] memory) {
        return students;
    }

    // Receive function - called when contract receives plain Ether
    receive() external payable {}

    // Fallback function - called when no function matches AND data is sent
    fallback() external payable {}
}

```

---

## 1️⃣1️⃣ BLT: Bank Details Contract

```solidity
// Paste your Bank Details Solidity contract here
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleBank {

    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint _amount) public {
        require(balances[msg.sender] >= _amount, "Not enough balance");
        balances[msg.sender] -= _amount;
        payable(msg.sender).transfer(_amount);
    }

    function getBalance() public view returns (uint) {
        return balances[msg.sender];
    }
}

```

---

```

---

If you want, I can also generate a **single PDF**, **GitHub README**, or **practical file format** with this template.
```
