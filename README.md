# Task 6: K-Nearest Neighbors (KNN) Classification

A complete implementation of KNN classification on the Iris dataset using Scikit-learn, Pandas, and Matplotlib.

---

## 📁 Project Structure

```text
├── knn_classification.py   # Main script
├── Iris.csv
│             # Dataset
├── knn_iris_results.png   # Generated visualizations
└── README.md              # Documentation
```

---

## 📌 Objective

Understand and implement KNN for classification by:

- Normalizing features using `StandardScaler`
- Training `KNeighborsClassifier` with different K values (1–25)
- Selecting the best K via 5-fold cross-validation
- Evaluating using accuracy, confusion matrix, and classification report
- Visualizing decision boundaries and model behavior

---

## 📊 Dataset

**Iris Dataset** — 150 samples, 4 features, 3 classes

| Feature | Description |
|----------|-------------|
| SepalLengthCm | Length of the sepal in cm |
| SepalWidthCm | Width of the sepal in cm |
| PetalLengthCm | Length of the petal in cm |
| PetalWidthCm | Width of the petal in cm |
| Species | Target class (Setosa / Versicolor / Virginica) |

Each class contains **50 samples**, making the dataset balanced and suitable for classification experiments.

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install scikit-learn pandas matplotlib seaborn
```

### Recommended Versions

| Library | Version |
|----------|----------|
| scikit-learn | ≥ 1.0 |
| pandas | ≥ 1.3 |
| matplotlib | ≥ 3.4 |
| seaborn | ≥ 0.11 |
| numpy | ≥ 1.21 |

---

## 🚀 How to Run

### 1. Place Dataset

Create the following structure:

```text
project_folder/
│
├── knn_classification.py
├── iris_data/
│   └── Iris.csv
└── README.md
```

### 2. Install Dependencies

```bash
pip install scikit-learn pandas matplotlib seaborn
```

### 3. Execute the Program

```bash
python knn_classification.py
```

### 4. Generated Output

The script creates:

```text
knn_iris_results.png
```

containing all analysis plots and visualizations.

---

## 🔬 Implementation Workflow

### Step 1: Load and Explore Data

- Load dataset using Pandas
- Remove unnecessary `Id` column
- Display dataset shape
- Check class distribution
- Generate descriptive statistics

### Step 2: Feature and Target Preparation

- Separate input features (`X`)
- Separate target labels (`y`)
- Encode labels using `LabelEncoder`

### Step 3: Train-Test Split

```python
train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

Ensures balanced class distribution in both training and testing sets.

### Step 4: Feature Scaling

Use:

```python
StandardScaler()
```

to normalize features and improve KNN distance calculations.

### Step 5: K Value Experimentation

Evaluate K values from:

```python
1 → 25
```

For each K:

- Train model
- Compute training accuracy
- Compute testing accuracy
- Perform 5-fold cross-validation

### Step 6: Select Optimal K

Choose the K value with the highest mean cross-validation score.

**Best K = 5**

### Step 7: Final Model Evaluation

Metrics used:

- Accuracy Score
- Classification Report
- Confusion Matrix

### Step 8: Decision Boundary Visualization

Train KNN using:

- Petal Length vs Petal Width
- Sepal Length vs Sepal Width

Generate mesh-grid predictions to visualize class regions.

---

## 📈 Results

| Metric | Value |
|----------|----------|
| Best K | **5** |
| Train Accuracy | **97.50%** |
| Test Accuracy | **93.33%** |

### Classification Report (K = 5)

| Class | Precision | Recall | F1-Score |
|---------|---------|---------|---------|
| Iris-setosa | 1.00 | 1.00 | 1.00 |
| Iris-versicolor | 0.83 | 1.00 | 0.91 |
| Iris-virginica | 1.00 | 0.80 | 0.89 |
| Overall | 0.94 | 0.93 | 0.93 |

---

## 📊 Visualizations Generated

The output image contains **9 plots**:

| No. | Visualization | Purpose |
|------|---------------|----------|
| 1 | K vs Accuracy | Compare train and test performance |
| 2 | Cross-Validation Scores | Mean CV score with standard deviation |
| 3 | Confusion Matrix | Classification performance summary |
| 4 | Petal Length Distribution | Species-wise histogram |
| 5 | Correlation Heatmap | Feature relationships |
| 6 | Petal Length vs Width Scatter Plot | Class separation visualization |
| 7 | Petal Decision Boundary | KNN classification regions |
| 8 | Sepal Decision Boundary | KNN classification regions |
| 9 | Accuracy Comparison Bar Chart | Performance for selected K values |
