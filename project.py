# =====================================================================
#  RAINFALL PREDICTION USING LOGISTIC REGRESSION  (Unit 1 to Unit 4)
#  SYBCA (AI) Sem-III - 303 Machine Learning - Mini Project
#
#  Use: Rainfall.csv file ane aa .py file ek j folder ma rakho.
#  Run: python Rainfall_Unitwise_Code.py   (ya Jupyter ma unit pramane cells banavo)
#  Graph window close karo etle next graph aavse.
#  Install (ek vaar): pip install pandas numpy matplotlib seaborn scikit-learn
# =====================================================================

# ---------------------------------------------------------------------
# COMMON IMPORTS (badhi libraries ahi j import)
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# =====================================================================
# UNIT 1 : FUNDAMENTALS OF AI AND ML  (Understanding Data + EDA)
# =====================================================================
print("=" * 60); print("UNIT 1 : UNDERSTANDING DATA AND EDA"); print("=" * 60)

# ---- 1.1 Dataset load karvo ----
df = pd.read_csv("Rainfall.csv")
df.columns = df.columns.str.strip()      # column name ma extra space hata (winddirection)

print("\nFirst 5 rows:\n", df.head())
print("\nShape (rows, columns):", df.shape)
print("\nColumns:", df.columns.tolist())
df.info()                                # dataset information

# ---- 1.2 Quantitative ane Qualitative data (1.4) ----
print("\nDatatypes:\n", df.dtypes)
quantitative = df.select_dtypes(include=["int64", "float64"])
qualitative = df.select_dtypes(include=["object", "category"])
print("\nQuantitative columns:", quantitative.columns.tolist())
print("Qualitative columns :", qualitative.columns.tolist())

# ---- 1.3 Missing values ane duplicates ----
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# ---- 1.4 Target variable ----
print("\nRainfall counts:\n", df["rainfall"].value_counts())
df["target"] = df["rainfall"].map({"yes": 1, "no": 0})   # yes=1, no=0

# num_cols = numeric feature columns (day ane target sivay)
num_cols = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
            "humidity", "cloud", "sunshine", "winddirection", "windspeed"]

# ---- 1.5 Spread of data: Skewness ane Kurtosis (1.5) ----
print("\nSkewness:\n", df[num_cols].skew().round(3))
print("\nKurtosis:\n", df[num_cols].kurt().round(3))

# ---- 1.6 UNIVARIATE : Histogram + Normal curve (KDE) ----
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(col)
plt.suptitle("Histogram of all features")
plt.tight_layout()
plt.show()

# ---- 1.7 Box Plot (spread ane outliers) - BEFORE handling ----
plt.figure(figsize=(13, 5))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=40)
plt.title("Box Plot - Before Outlier Handling")
plt.tight_layout()
plt.show()

# ---- 1.8 Pie Chart ane Bar Chart (Count Plot) of target ----
df["rainfall"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Rainfall (Yes / No)")
plt.ylabel("")
plt.show()

sns.countplot(x="rainfall", data=df)
plt.title("Rainfall Count")
plt.xlabel("Rainfall")
plt.ylabel("Number of days")
plt.show()

# ---- 1.9 BIVARIATE : Feature vs Rainfall ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, ["cloud", "sunshine", "humidity"]):
    sns.boxplot(x="rainfall", y=col, data=df, ax=ax)
    ax.set_title(col + " vs rainfall")
plt.tight_layout()
plt.show()

# Scatter plot (two numerical variables)
sns.scatterplot(x="humidity", y="cloud", hue="rainfall", data=df)
plt.title("Humidity vs Cloud")
plt.show()

# ---- 1.10 MULTIVARIATE : Pair Plot ----
sns.pairplot(df[["humidity", "cloud", "sunshine", "dewpoint", "rainfall"]], hue="rainfall")
plt.show()

# ---- 1.11 Handling Missing Data (1.3.2) -> Median Imputation ----
for col in ["winddirection", "windspeed"]:
    df[col] = df[col].fillna(df[col].median())
print("\nMissing values after handling:\n", df[num_cols].isnull().sum().sum())

# ---- 1.12 Detecting and Handling Outliers (1.3.3) -> IQR + Capping ----
print("\nOutlier count (IQR method) and Before / After:")
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    before = (df[col].min(), df[col].max())
    df[col] = df[col].clip(low, high)                     # capping
    after = (df[col].min(), df[col].max())
    print(f"{col:14s} outliers={n_out:3d}  before={before}  after={after}")

# Box plot AFTER handling
plt.figure(figsize=(13, 5))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=40)
plt.title("Box Plot - After Outlier Handling (Capping)")
plt.tight_layout()
plt.show()


# =====================================================================
# UNIT 2 : AUTOMATE EDA  (Pandas, NumPy, Covariance, Correlation)
# =====================================================================
print("\n" + "=" * 60); print("UNIT 2 : AUTOMATE EDA"); print("=" * 60)

# ---- 2.1 NumPy ----
arr = df["humidity"].to_numpy()            # dataframe column -> numpy array
print("\nHumidity -> Mean:", np.mean(arr).round(2), " Median:", np.median(arr),
      " Std:", np.std(arr).round(2), " Variance:", np.var(arr).round(2))

# ---- 2.2 Pandas functions ----
print("\nhead(3):\n", df.head(3))
print("\ntail(3):\n", df.tail(3))
print("\nRows 2 to 4 (slicing):\n", df[2:5])
print("\niloc[0]:\n", df.iloc[0])
print("\nDays where humidity > 90 (condition):\n", df[df.humidity > 90][["day", "humidity", "rainfall"]].head())
print("\nDescribe:\n", df[num_cols].describe().round(2))
print("\nUnique values:\n", df.nunique())

# Central tendency of cloud
print("\nCloud -> Mean:", df.cloud.mean().round(2), " Median:", df.cloud.median(),
      " Mode:", df.cloud.mode()[0], " Std:", df.cloud.std().round(2))

# Group by rainfall (mean of features for yes / no)
print("\nAverage value for rain yes / no:\n", df.groupby("rainfall")[["cloud", "sunshine", "humidity", "dewpoint"]].mean().round(2))

# ---- 2.3 Covariance and Correlation (2.2.3, 2.2.4) ----
print("\nCovariance (selected):\n", df[["humidity", "cloud", "sunshine", "target"]].cov().round(2))
corr = df[num_cols + ["target"]].corr()
print("\nCorrelation with rainfall:\n", corr["target"].drop("target").sort_values(ascending=False).round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# ---- 2.4 Automated EDA (koi pan EK use karvu) ----
# Pehla install karo:  pip install ydata-profiling   (ya)   pip install sweetviz
try:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(df.drop(columns=["target"]), title="Rainfall EDA Report")
    profile.to_file("EDA_Report.html")
    print("\nEDA_Report.html banyu (ydata-profiling)")
except ImportError:
    print("\nydata-profiling install nathi. Run: pip install ydata-profiling")
    # Option 2 - Sweetviz:
    # import sweetviz as sv
    # sv.analyze(df.drop(columns=["target"])).show_html("EDA_Report.html")
    # Option 3 - D-Tale:
    # import dtale
    # dtale.show(df).open_browser()


# =====================================================================
# UNIT 3 : UNDERSTANDING MACHINE LEARNING
#          (Supervised Learning, Features/Labels, Train-Test Split)
# =====================================================================
print("\n" + "=" * 60); print("UNIT 3 : SUPERVISED LEARNING SETUP"); print("=" * 60)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Type: Supervised Learning -> Classification (output = yes / no)")

# ---- 3.1 Features (X) ane Label (y) ----
features = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
            "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
X = df[features]            # Independent variables (2D)
y = df["target"]            # Dependent variable / Label (1D)
print("\nFeatures (X):\n", X.head())
print("\nLabel (y):\n", y.head())

# ---- 3.2 Training ane Testing data (80% : 20%) ----
# test_size=0.20  -> 20% testing, random_state=42 -> darek vaar same split
# stratify=y      -> yes/no nu ratio train ane test banne ma same rahe
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))

# ---- 3.3 Scaling (badha features ni scale same karva) ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)     # train par fit + transform
X_test_s = scaler.transform(X_test)           # test par fakt transform

# Note: MSE ane MAE (3.4) regression mate che. Aa project classification che,
# etle evaluation Unit 4 ma Accuracy, Precision, Recall, F1 thi thase.


# =====================================================================
# UNIT 4 : SUPERVISED LEARNING ALGORITHMS
#          (Logistic Regression, Evaluation, Cross-Validation, Regularization)
# =====================================================================
print("\n" + "=" * 60); print("UNIT 4 : LOGISTIC REGRESSION"); print("=" * 60)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, log_loss)
from sklearn.model_selection import cross_val_score

# ---- 4.1 Sigmoid function (4.2.1) ----
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))               # s(z) = 1 / (1 + e^-z)
plt.plot(z, sigmoid)
plt.axhline(0.5, color="red", linestyle="--", label="Threshold = 0.5")
plt.xlabel("z (weighted input)")
plt.ylabel("Probability")
plt.title("Sigmoid Function")
plt.legend()
plt.grid(True)
plt.show()

# ---- 4.2 Cost function: Log Loss (4.2.2) - formula example ----
# y=1 ane predicted probability 0.90 -> cost = -log(0.90)
print("\nLog loss (y=1, p=0.90):", round(-np.log(0.90), 4))     # good model -> small
print("Log loss (y=1, p=0.20):", round(-np.log(0.20), 4))       # bad model  -> large

# ---- 4.3 Model build (Logistic Regression) ----
# Andar sigmoid + log loss + gradient descent type optimization automatic thay che
model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)                # training
y_pred = model.predict(X_test_s)             # testing data par prediction

print("\nActual   :", y_test.values[:15])
print("Predicted:", y_pred[:15])

# Probability (sigmoid no output) - pehla 5 test days
prob = model.predict_proba(X_test_s[:5])
print("\nProbability [No, Yes] of first 5 test days:\n", prob.round(3))

# Log loss of model
print("Model Log Loss (test):", round(log_loss(y_test, model.predict_proba(X_test_s)), 4))

# ---- 4.4 Evaluation Metrics (4.3.4) ----
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

print("\nAccuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---- 4.5 Overfitting / Underfitting check (4.3.1, 4.3.2) ----
train_acc = accuracy_score(y_train, model.predict(X_train_s))
test_acc = accuracy_score(y_test, y_pred)
print("\nTraining accuracy:", round(train_acc, 4))
print("Testing accuracy :", round(test_acc, 4))
if train_acc - test_acc > 0.10:
    print("Result: Overfitting (train accuracy bahu vadhare che)")
elif train_acc < 0.70:
    print("Result: Underfitting (banne accuracy ochi che)")
else:
    print("Result: Model balanced che (train ane test accuracy najik che)")

# ---- 4.6 K-Fold Cross-Validation (4.3.3.2) ----
X_all = StandardScaler().fit_transform(X)
cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X_all, y, cv=5, scoring="accuracy")
print("\nAccuracy of each fold:", cv_scores.round(3))
print("Average CV Accuracy  :", round(cv_scores.mean(), 4))

plt.bar(["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"], cv_scores)
plt.axhline(cv_scores.mean(), color="red", linestyle="--", label="Mean")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("5-Fold Cross-Validation")
plt.legend()
plt.show()

# ---- 4.7 Regularization L1 and L2 (4.3.3.1) ----
# L1 (Lasso): penalty = lambda * sum(|coefficient|)  -> coefficient 0 kari shake
l1_model = LogisticRegression(penalty="l1", solver="saga", max_iter=5000)
l1_model.fit(X_train_s, y_train)
l1_pred = l1_model.predict(X_test_s)

# L2 (Ridge): penalty = lambda * sum(coefficient^2) -> coefficient nana kare
l2_model = LogisticRegression(penalty="l2", solver="saga", max_iter=5000)
l2_model.fit(X_train_s, y_train)
l2_pred = l2_model.predict(X_test_s)

print("\nL1 Accuracy:", round(accuracy_score(y_test, l1_pred), 4))
print("L2 Accuracy:", round(accuracy_score(y_test, l2_pred), 4))

coef_df = pd.DataFrame({"Feature": features,
                        "Default": model.coef_[0].round(3),
                        "L1 (Lasso)": l1_model.coef_[0].round(3),
                        "L2 (Ridge)": l2_model.coef_[0].round(3)})
print("\nCoefficients:\n", coef_df)
print("L1 e jena coefficient 0 karya (feature selection):",
      [f for f, c in zip(features, l1_model.coef_[0]) if c == 0])

# Coefficient chart
coef_df.set_index("Feature")[["L1 (Lasso)", "L2 (Ridge)"]].plot(kind="barh", figsize=(9, 6))
plt.title("Feature Coefficients: L1 vs L2")
plt.tight_layout()
plt.show()

# ---- 4.8 Compare results ----
def get_metrics(yt, yp):
    return [accuracy_score(yt, yp), precision_score(yt, yp), recall_score(yt, yp), f1_score(yt, yp)]

result = pd.DataFrame({"Default": get_metrics(y_test, y_pred),
                       "L1 (Lasso)": get_metrics(y_test, l1_pred),
                       "L2 (Ridge)": get_metrics(y_test, l2_pred)},
                      index=["Accuracy", "Precision", "Recall", "F1 Score"]).round(4)
print("\nComparison:\n", result)

result.plot(kind="bar", figsize=(9, 5))
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.title("Model Comparison")
plt.ylabel("Score")
plt.show()

# ---- 4.9 Navo prediction (new day) ----
new_day = pd.DataFrame([[1015, 28, 25, 22, 21, 90, 90, 1.0, 50, 20]], columns=features)
result_new = model.predict(scaler.transform(new_day))[0]
prob_new = model.predict_proba(scaler.transform(new_day))[0]
print("\nNew day prediction:", "Rainfall = YES" if result_new == 1 else "Rainfall = NO")
print("Probability of No  :", round(prob_new[0], 3))
print("Probability of Yes :", round(prob_new[1], 3))

print("\n*** PROJECT COMPLETE ***")


# =====================================================================
#  RAINFALL PREDICTION - COMPLETE PROJECT (Unit 1 to 4)
#  Aa file ek j vaar chalavo etle:
#   - badha charts Reports folder ma save thase
#   - Logistic Regression (Classification) + GridSearch tuning
#   - Linear Regression (sunshine hours prediction: R2, MSE, RMSE)
#   - model Models folder ma save thase (joblib)
#   - test_predictions.csv banse
#   - PDF report automatic banse ane khulse
#
#  Folder: Rainfall.csv ane aa file ek j folder ma rakho.
#  Install (ek vaar): pip install pandas numpy matplotlib seaborn scikit-learn joblib reportlab
# =====================================================================
import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # graph window nahi khule, file ma save thase
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, r2_score, mean_squared_error,
                             mean_absolute_error)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# FOLDERS
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "Reports"
MODELS_DIR = ROOT / "Models"
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

NUM_COLS = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
            "humidity", "cloud", "sunshine", "winddirection", "windspeed"]


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / name, dpi=110)
    plt.close()


# =====================================================================
# STEP 1 : LOAD DATA AND UNDERSTAND IT                        (Unit 1)
# =====================================================================
def load_data():
    section("STEP 1: LOAD DATASET AND UNDERSTAND DATA")
    path = ROOT / "Rainfall.csv"
    if not path.exists():
        path = ROOT / "Dataset" / "dataset.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["target"] = df["rainfall"].map({"yes": 1, "no": 0})
    print("Shape:", df.shape)
    print(df.head())
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("Duplicate rows:", df.duplicated().sum())
    print("\nRainfall counts:\n", df["rainfall"].value_counts())
    return df


# =====================================================================
# STEP 2 : EDA (CHARTS)                                  (Unit 1 and 2)
# =====================================================================
def run_eda(df):
    section("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("Skewness:\n", df[NUM_COLS].skew().round(3))
    print("Kurtosis:\n", df[NUM_COLS].kurt().round(3))

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, col in zip(axes.ravel(), NUM_COLS):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
    save_fig("01_histograms.png")

    plt.figure(figsize=(13, 5))
    sns.boxplot(data=df[NUM_COLS])
    plt.xticks(rotation=40)
    plt.title("Box Plot - Before Outlier Handling")
    save_fig("02_boxplot_before.png")

    df["rainfall"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("Rainfall (Yes / No)")
    plt.ylabel("")
    save_fig("03_pie_rainfall.png")

    sns.countplot(x="rainfall", data=df)
    plt.title("Rainfall Count")
    save_fig("04_bar_rainfall.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["cloud", "sunshine", "humidity"]):
        sns.boxplot(x="rainfall", y=col, data=df, ax=ax)
        ax.set_title(col + " vs rainfall")
    save_fig("05_bivariate.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[NUM_COLS + ["target"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    save_fig("06_correlation.png")
    print("EDA charts saved in:", REPORTS_DIR)

    # Automated EDA (optional)
    try:
        from ydata_profiling import ProfileReport
        ProfileReport(df.drop(columns=["target"]), title="Rainfall EDA").to_file(
            REPORTS_DIR / "EDA_Report.html")
        print("EDA_Report.html generated")
    except ImportError:
        print("(Optional) pip install ydata-profiling  -> EDA_Report.html banse")


# =====================================================================
# STEP 3 : MISSING VALUES AND OUTLIERS                          (Unit 2)
# =====================================================================
def clean_data(df):
    section("STEP 3: HANDLE MISSING VALUES AND OUTLIERS")
    for col in ["winddirection", "windspeed"]:
        df[col] = df[col].fillna(df[col].median())
    rows = []
    for col in NUM_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        df[col] = df[col].clip(low, high)
        rows.append((col, n_out))
        print(f"{col:14s} outliers capped: {n_out}")

    plt.figure(figsize=(13, 5))
    sns.boxplot(data=df[NUM_COLS])
    plt.xticks(rotation=40)
    plt.title("Box Plot - After Outlier Handling")
    save_fig("07_boxplot_after.png")
    return df, rows


# =====================================================================
# STEP 4 : CLASSIFICATION - LOGISTIC REGRESSION                (Unit 3, 4)
# =====================================================================
def classification(df):
    section("STEP 4: LOGISTIC REGRESSION (Rainfall Yes / No)")
    features = [c for c in NUM_COLS]
    X, y = df[features], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)

    base = {"Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred)}
    print("Base model:", {k: round(v, 4) for k, v in base.items()})
    print("Train accuracy:", round(accuracy_score(y_train, model.predict(X_train_s)), 4))

    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    save_fig("08_confusion_matrix.png")

    # ---- Cross validation ----
    X_all = StandardScaler().fit_transform(X)
    cv = cross_val_score(LogisticRegression(max_iter=1000), X_all, y, cv=5, scoring="accuracy")
    print("5-Fold CV:", cv.round(3), "Mean:", round(cv.mean(), 4))

    # ---- Regularization L1 and L2 ----
    reg = {}
    for name, pen in [("L1 (Lasso)", "l1"), ("L2 (Ridge)", "l2")]:
        m = LogisticRegression(penalty=pen, solver="saga", max_iter=5000).fit(X_train_s, y_train)
        p = m.predict(X_test_s)
        reg[name] = {"Accuracy": accuracy_score(y_test, p), "Precision": precision_score(y_test, p),
                     "Recall": recall_score(y_test, p), "F1": f1_score(y_test, p)}
        print(name, {k: round(v, 4) for k, v in reg[name].items()})

    # ---- Tuning (GridSearchCV) : best C (regularization strength) ----
    grid = GridSearchCV(LogisticRegression(solver="saga", max_iter=5000),
                        {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]},
                        cv=5, scoring="accuracy")
    grid.fit(X_train_s, y_train)
    best = grid.best_estimator_
    tuned_pred = best.predict(X_test_s)
    tuned = {"Accuracy": accuracy_score(y_test, tuned_pred), "Precision": precision_score(y_test, tuned_pred),
             "Recall": recall_score(y_test, tuned_pred), "F1": f1_score(y_test, tuned_pred)}
    print("Best parameters:", grid.best_params_)
    print("Tuned model:", {k: round(v, 4) for k, v in tuned.items()})

    comp = pd.DataFrame({"Default": base, "L1": reg["L1 (Lasso)"], "L2": reg["L2 (Ridge)"],
                         "Tuned": tuned}).round(4)
    comp.plot(kind="bar", figsize=(9, 5)); plt.ylim(0, 1.1); plt.xticks(rotation=0)
    plt.title("Model Comparison"); plt.ylabel("Score")
    save_fig("09_model_comparison.png")

    coef = pd.Series(best.coef_[0], index=features).sort_values()
    coef.plot(kind="barh", figsize=(8, 5)); plt.title("Feature Coefficients (Tuned Model)")
    save_fig("10_coefficients.png")

    # ---- Save model + predictions ----
    joblib.dump({"model": best, "scaler": scaler, "features": features},
                MODELS_DIR / "rainfall_model.joblib")
    out = X_test.copy()
    out["Actual"] = y_test.map({1: "yes", 0: "no"})
    out["Predicted"] = pd.Series(tuned_pred, index=y_test.index).map({1: "yes", 0: "no"})
    out["Prob_Rain"] = best.predict_proba(X_test_s)[:, 1].round(3)
    out.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    metrics = {"base": base, "tuned": tuned, "reg": reg, "cm": cm.tolist(),
               "cv_scores": cv.round(3).tolist(), "cv_mean": float(cv.mean()),
               "best_params": {k: str(v) for k, v in grid.best_params_.items()},
               "train_size": len(X_train), "test_size": len(X_test),
               "pred_yes": int((tuned_pred == 1).sum()), "pred_no": int((tuned_pred == 0).sum())}
    return metrics


# =====================================================================
# STEP 5 : REGRESSION - LINEAR REGRESSION (Sunshine hours)       (Unit 4.1)
# =====================================================================
def regression(df):
    section("STEP 5: LINEAR REGRESSION - SUNSHINE HOURS PREDICTION")
    target = "sunshine"
    feats = [c for c in NUM_COLS if c != target]
    X, y = df[feats], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    results = {}
    for name, mdl in [("Linear", LinearRegression()), ("Ridge (L2)", Ridge(alpha=1.0)),
                      ("Lasso (L1)", Lasso(alpha=0.05))]:
        mdl.fit(X_train, y_train)
        p = mdl.predict(X_test)
        mse = mean_squared_error(y_test, p)
        results[name] = {"R2": r2_score(y_test, p), "MSE": mse,
                         "RMSE": float(np.sqrt(mse)), "MAE": mean_absolute_error(y_test, p)}
        print(f"{name:11s} R2={results[name]['R2']:.4f}  MSE={mse:.4f}  "
              f"RMSE={results[name]['RMSE']:.4f}  MAE={results[name]['MAE']:.4f}")

    lin = LinearRegression().fit(X_train, y_train)
    p = lin.predict(X_test)
    plt.scatter(y_test, p)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Sunshine"); plt.ylabel("Predicted Sunshine")
    plt.title("Linear Regression: Actual vs Predicted")
    save_fig("11_regression_actual_vs_pred.png")
    results["target"] = target
    return results


# =====================================================================
# STEP 6 : SUMMARY PRINT
# =====================================================================
def print_summary(m, reg_m):
    section("FINAL RAINFALL PREDICTION SUMMARY REPORT")
    print(f" Model Name           : Tuned Logistic Regression {m['best_params']}")
    print(f" Model Test Accuracy  : {m['tuned']['Accuracy']*100:.2f}%")
    print(f" Model F1-Score       : {m['tuned']['F1']:.4f}")
    print("-" * 64)
    print(f" Total Tested Days    : {m['test_size']}")
    print(f" Predicted RAIN (1)   : {m['pred_yes']} days ({m['pred_yes']/m['test_size']*100:.1f}%)")
    print(f" Predicted NO RAIN (0): {m['pred_no']} days ({m['pred_no']/m['test_size']*100:.1f}%)")
    print("-" * 64)
    lr = reg_m["Linear"]
    print(f" Regression target    : {reg_m['target']}")
    print(f" Regression R2        : {lr['R2']:.4f}")
    print(f" Regression RMSE      : {lr['RMSE']:.4f}")
    print(f" Predictions saved in : {REPORTS_DIR / 'test_predictions.csv'}")


# =====================================================================
# STEP 7 : PDF REPORT (automatic)
# =====================================================================
def create_project_report(df, outliers, m, reg_m):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                        TableStyle, Image, PageBreak)
        from PIL import Image as PILImage
    except ImportError:
        print("PDF mate: pip install reportlab pillow")
        return None

    ss = getSampleStyleSheet()
    S = []

    def tbl(data, widths=None):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        return t

    def pic(name, w=15):
        p = REPORTS_DIR / name
        if not p.exists():
            return Spacer(1, 1)
        iw, ih = PILImage.open(p).size
        return Image(str(p), width=w * cm, height=w * cm * ih / iw)

    S += [Spacer(1, 4 * cm), Paragraph("RAINFALL PREDICTION", ss["Title"]),
          Paragraph("using Logistic Regression and Linear Regression", ss["Heading2"]),
          Spacer(1, 1 * cm),
          Paragraph("SYBCA (AI) Sem-III | 303 Fundamentals of Machine Learning | Mini Project", ss["Normal"]),
          Paragraph("Student: [Enter name] | Roll No: [Enter roll no]", ss["Normal"]), PageBreak()]

    S += [Paragraph("1. Problem Statement and Objectives", ss["Heading1"]),
          Paragraph("Predict whether it will rain (Yes/No) from daily weather data and also predict "
                    "sunshine hours using regression. Workflow: EDA, preprocessing, model building, "
                    "evaluation, cross-validation and regularization.", ss["Normal"]),
          Paragraph("2. Dataset Description", ss["Heading1"]),
          Paragraph(f"Records: {df.shape[0]}, Columns: {df.shape[1] - 1}. Target: rainfall (yes/no). "
                    f"Train/Test split: {m['train_size']}/{m['test_size']}.", ss["Normal"]),
          Paragraph("3. Data Preprocessing", ss["Heading1"]),
          Paragraph("Missing values were filled with the median. Outliers were capped using the IQR method.", ss["Normal"]),
          tbl([["Feature", "Outliers capped"]] + [[a, b] for a, b in outliers], [6 * cm, 4 * cm]),
          PageBreak(), Paragraph("4. Exploratory Data Analysis", ss["Heading1"])]
    for f in ["01_histograms.png", "02_boxplot_before.png", "07_boxplot_after.png",
              "05_bivariate.png", "06_correlation.png"]:
        S += [pic(f, 14), Spacer(1, 8)]
    S += [PageBreak(), Paragraph("5. Model Evaluation (Classification)", ss["Heading1"])]
    rows = [["Model", "Accuracy", "Precision", "Recall", "F1"]]
    rows.append(["Default"] + [round(m["base"][k], 4) for k in ["Accuracy", "Precision", "Recall", "F1"]])
    for k, v in m["reg"].items():
        rows.append([k] + [round(v[x], 4) for x in ["Accuracy", "Precision", "Recall", "F1"]])
    rows.append(["Tuned"] + [round(m["tuned"][k], 4) for k in ["Accuracy", "Precision", "Recall", "F1"]])
    S += [tbl(rows, [4 * cm] + [3 * cm] * 4), Spacer(1, 8),
          Paragraph(f"5-Fold CV scores: {m['cv_scores']} | Mean: {m['cv_mean']:.4f}", ss["Normal"]),
          Paragraph(f"Best parameters (GridSearch): {m['best_params']}", ss["Normal"]),
          pic("08_confusion_matrix.png", 8), pic("09_model_comparison.png", 12), pic("10_coefficients.png", 12),
          PageBreak(), Paragraph("6. Regression Results (Sunshine Hours)", ss["Heading1"])]
    rr = [["Model", "R2", "MSE", "RMSE", "MAE"]]
    for k in ["Linear", "Ridge (L2)", "Lasso (L1)"]:
        v = reg_m[k]
        rr.append([k] + [round(v[x], 4) for x in ["R2", "MSE", "RMSE", "MAE"]])
    S += [tbl(rr, [4 * cm] + [3 * cm] * 4), pic("11_regression_actual_vs_pred.png", 11),
          Paragraph("7. Conclusion and Future Scope", ss["Heading1"]),
          Paragraph(f"The tuned Logistic Regression model reached {m['tuned']['Accuracy']*100:.1f}% accuracy. "
                    "Cloud, sunshine and humidity are the most important factors. Future scope: larger dataset, "
                    "more algorithms (Decision Tree, Random Forest), deployment as a web app.", ss["Normal"])]

    pdf_path = REPORTS_DIR / "Rainfall_Project_Report.pdf"
    SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm).build(S)
    return pdf_path


# =====================================================================
# MAIN
# =====================================================================
def main():
    df = load_data()
    run_eda(df)
    df, outliers = clean_data(df)
    metrics = classification(df)
    reg_metrics = regression(df)
    print_summary(metrics, reg_metrics)

    pdf_path = create_project_report(df, outliers, metrics, reg_metrics)

    section("PROJECT COMPLETED SUCCESSFULLY")
    print(f"All charts and reports are in : {REPORTS_DIR}")
    print(f"Saved model is in             : {MODELS_DIR / 'rainfall_model.joblib'}")

    # PDF ane Reports folder potano khule (Windows)
    if os.environ.get("NO_OPEN") != "1" and sys.platform.startswith("win"):
        if pdf_path and Path(pdf_path).exists():
            os.startfile(pdf_path)
        os.startfile(REPORTS_DIR)


if __name__ == "__main__":
    main()