"""
Task 6: K-Nearest Neighbors (KNN) Classification

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  KNN CLASSIFICATION — IRIS DATASET")
print("=" * 60)

df = pd.read_csv("Iris.csv")
df.drop(columns=["Id"], inplace=True)          # drop ID column

print("\n📊 Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df["Species"].value_counts())
print("\nBasic statistics:")
print(df.describe())

# ─────────────────────────────────────────────
# 2. FEATURE & LABEL PREPARATION
# ─────────────────────────────────────────────
X = df.drop(columns=["Species"])
y = df["Species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)               # setosa=0, versicolor=1, virginica=2

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ─────────────────────────────────────────────
# 4. FEATURE NORMALIZATION (StandardScaler)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n✅ Train size:", X_train_scaled.shape, "| Test size:", X_test_scaled.shape)

# ─────────────────────────────────────────────
# 5. EXPERIMENT WITH DIFFERENT K VALUES
# ─────────────────────────────────────────────
k_values   = range(1, 26)
train_accs = []
test_accs  = []
cv_means   = []
cv_stds    = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_accs.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_accs.append(accuracy_score(y_test,  knn.predict(X_test_scaled)))
    cv = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_means.append(cv.mean())
    cv_stds.append(cv.std())

best_k = k_values[np.argmax(cv_means)]
print(f"\n🏆 Best K (by 5-fold CV): {best_k}")

# ─────────────────────────────────────────────
# 6. FINAL MODEL WITH BEST K
# ─────────────────────────────────────────────
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred = best_knn.predict(X_test_scaled)

print(f"\n📈 Final Model  (K = {best_k})")
print(f"   Train Accuracy : {accuracy_score(y_train, best_knn.predict(X_train_scaled)):.4f}")
print(f"   Test  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0f1117")
COLOR_BG   = "#0f1117"
COLOR_CARD = "#1a1d27"
COLOR_TEXT = "#e0e0e0"
PALETTE    = ["#4cc9f0", "#f72585", "#7209b7"]

# ── helper ──
def style_ax(ax, title):
    ax.set_facecolor(COLOR_CARD)
    ax.set_title(title, color=COLOR_TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors=COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.yaxis.label.set_color(COLOR_TEXT)

# ── Plot 1 : K vs Accuracy ──
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(k_values, train_accs, marker="o", color=PALETTE[0],
         linewidth=2, markersize=5, label="Train")
ax1.plot(k_values, test_accs,  marker="s", color=PALETTE[1],
         linewidth=2, markersize=5, label="Test")
ax1.axvline(best_k, color="yellow", linestyle="--", linewidth=1.5,
            label=f"Best K={best_k}")
ax1.set_xlabel("K"); ax1.set_ylabel("Accuracy")
ax1.legend(facecolor=COLOR_CARD, labelcolor=COLOR_TEXT, fontsize=9)
style_ax(ax1, "K Value vs Accuracy")

# ── Plot 2 : CV Score with Error Bands ──
ax2 = fig.add_subplot(3, 3, 2)
cv_means_arr = np.array(cv_means)
cv_stds_arr  = np.array(cv_stds)
ax2.plot(k_values, cv_means_arr, color=PALETTE[2], linewidth=2, marker="D",
         markersize=5)
ax2.fill_between(k_values,
                 cv_means_arr - cv_stds_arr,
                 cv_means_arr + cv_stds_arr,
                 alpha=0.25, color=PALETTE[2])
ax2.axvline(best_k, color="yellow", linestyle="--", linewidth=1.5)
ax2.set_xlabel("K"); ax2.set_ylabel("CV Mean Accuracy")
style_ax(ax2, "5-Fold Cross-Validation Score")

# ── Plot 3 : Confusion Matrix ──
ax3 = fig.add_subplot(3, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="magma",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=ax3, cbar=False,
            annot_kws={"color": "white", "fontsize": 12})
ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
style_ax(ax3, f"Confusion Matrix (K={best_k})")
ax3.tick_params(axis="x", rotation=30)

# ── Plot 4 : Feature Distributions ──
features = X.columns.tolist()
ax4 = fig.add_subplot(3, 3, 4)
for i, (species, color) in enumerate(zip(le.classes_, PALETTE)):
    subset = df[df["Species"] == species]
    ax4.hist(subset["PetalLengthCm"], bins=15, alpha=0.65,
             color=color, label=species, edgecolor="none")
ax4.set_xlabel("Petal Length (cm)"); ax4.set_ylabel("Count")
ax4.legend(facecolor=COLOR_CARD, labelcolor=COLOR_TEXT, fontsize=9)
style_ax(ax4, "Petal Length Distribution by Species")

# ── Plot 5 : Correlation Heatmap ──
ax5 = fig.add_subplot(3, 3, 5)
corr = df.drop(columns=["Species"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax5, cbar=False,
            annot_kws={"color": "white", "fontsize": 10})
style_ax(ax5, "Feature Correlation Heatmap")

# ── Plot 6 : Pairplot-style scatter (PetalLen vs PetalWid) ──
ax6 = fig.add_subplot(3, 3, 6)
for i, (species, color) in enumerate(zip(le.classes_, PALETTE)):
    mask = y_encoded == i
    ax6.scatter(X.loc[mask, "PetalLengthCm"], X.loc[mask, "PetalWidthCm"],
                c=color, label=species, alpha=0.8, edgecolors="none", s=50)
ax6.set_xlabel("Petal Length (cm)"); ax6.set_ylabel("Petal Width (cm)")
ax6.legend(facecolor=COLOR_CARD, labelcolor=COLOR_TEXT, fontsize=9)
style_ax(ax6, "Petal Length vs Petal Width")

# ── Plot 7 & 8 : Decision Boundaries (2 feature pairs) ──
def plot_decision_boundary(ax, feat1, feat2, title):
    """Train KNN on 2 features and visualize decision boundary."""
    idx1 = list(X.columns).index(feat1)
    idx2 = list(X.columns).index(feat2)

    X2 = X_train_scaled[:, [idx1, idx2]]
    knn2 = KNeighborsClassifier(n_neighbors=best_k)
    knn2.fit(X2, y_train)

    h = 0.03
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # soft region colors
    region_colors = ["#1a3a4a", "#3a1a2a", "#2a1a3a"]
    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap(region_colors)
    ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_bg)

    # scatter points
    X2_test = X_test_scaled[:, [idx1, idx2]]
    for i, (species, color) in enumerate(zip(le.classes_, PALETTE)):
        mask = y_train == i
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=color, s=40, alpha=0.7, edgecolors="none")
    # mark test points with black border
    for i, (species, color) in enumerate(zip(le.classes_, PALETTE)):
        mask = y_test == i
        ax.scatter(X2_test[mask, 0], X2_test[mask, 1],
                   c=color, s=70, edgecolors="white", linewidths=0.8)

    ax.set_xlabel(f"{feat1} (scaled)")
    ax.set_ylabel(f"{feat2} (scaled)")
    patches = [mpatches.Patch(color=c, label=s)
               for c, s in zip(PALETTE, le.classes_)]
    ax.legend(handles=patches, facecolor=COLOR_CARD,
              labelcolor=COLOR_TEXT, fontsize=8)
    style_ax(ax, title)

ax7 = fig.add_subplot(3, 3, 7)
plot_decision_boundary(ax7, "PetalLengthCm", "PetalWidthCm",
                        f"Decision Boundary\nPetal Length vs Width (K={best_k})")

ax8 = fig.add_subplot(3, 3, 8)
plot_decision_boundary(ax8, "SepalLengthCm", "SepalWidthCm",
                        f"Decision Boundary\nSepal Length vs Width (K={best_k})")

# ── Plot 9 : Accuracy bar chart for K=1,3,5,7,best_k ──
ax9 = fig.add_subplot(3, 3, 9)
highlight_ks = sorted(set([1, 3, 5, 7, 9, best_k]))
h_accs = [test_accs[k - 1] for k in highlight_ks]
bar_colors = [PALETTE[1] if k == best_k else PALETTE[0] for k in highlight_ks]
bars = ax9.bar([str(k) for k in highlight_ks], h_accs,
               color=bar_colors, edgecolor="none")
for bar, acc in zip(bars, h_accs):
    ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{acc:.3f}", ha="center", color=COLOR_TEXT, fontsize=10)
ax9.set_xlabel("K"); ax9.set_ylabel("Test Accuracy")
ax9.set_ylim(0.85, 1.02)
style_ax(ax9, "Test Accuracy for Selected K Values")

# ── Final Layout ──
plt.suptitle("KNN Classification — Iris Dataset",
             fontsize=18, fontweight="bold", color=COLOR_TEXT, y=1.01)
plt.tight_layout(pad=2.5)
plt.savefig("knn_iris_results.png", dpi=150, bbox_inches="tight",
            facecolor=COLOR_BG)
print("\n✅ Plot saved → knn_iris_results.png")
plt.close()

# ─────────────────────────────────────────────
# 8. SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY: K vs Test Accuracy")
print("=" * 60)
print(f"{'K':>5} | {'Train Acc':>10} | {'Test Acc':>9} | {'CV Mean':>8} | {'CV Std':>7}")
print("-" * 50)
for k in k_values:
    marker = " ← best" if k == best_k else ""
    print(f"{k:>5} | {train_accs[k-1]:>10.4f} | {test_accs[k-1]:>9.4f} | "
          f"{cv_means[k-1]:>8.4f} | {cv_stds[k-1]:>7.4f}{marker}")
print("=" * 60)
print(f"\n🎯 Best K = {best_k}  |  Test Accuracy = {test_accs[best_k-1]:.4f}")