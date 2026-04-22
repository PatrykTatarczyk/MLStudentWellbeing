import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("Raw_Data.csv")
print(f"Dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn\n")

wyklucz = [
    "1. Age", "2. Gender", "3. University", "4. Department",
    "5. Academic Year", "6. Current CGPA",
    "7. Did you receive a waiver or scholarship at your university?",
    "Anxiety Value", "Anxiety Label",
    "Depression Value", "Depression Label",
    "Stress Value", "Stress Label"
]

cechy = [col for col in df.columns if col not in wyklucz]
print(f"Liczba cech: {len(cechy)}")

X = df[cechy].values

y = np.where(df["Stress Label"] == "High Perceived Stress", 1, 0)

print(f"Rozkład klas: 0 (Low/Moderate) = {np.sum(y==0)}, 1 (High) = {np.sum(y==1)}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Zbiór treningowy: {X_train.shape[0]} próbek")
print(f"Zbiór testowy:    {X_test.shape[0]} próbek\n")

def wykres(tytul, param_nazwa, wartosci, acc_train, acc_test, plik):
    plt.figure(figsize=(8, 5))
    x = range(len(wartosci))
    plt.plot(x, acc_train, "o-", label="Trening", color="steelblue")
    plt.plot(x, acc_test,  "s--", label="Test",    color="tomato")
    plt.xticks(x, [str(w) for w in wartosci])
    plt.xlabel(param_nazwa)
    plt.ylabel("Dokładność (Accuracy)")
    plt.title(tytul)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plik, dpi=100)
    plt.close()
    print(f"  Wykres zapisany: {plik}")

print("=" * 60)
print("METODA 1: KNN — K-Nearest Neighbors")
print("=" * 60)

print("\nParametr: n_neighbors (liczba sąsiadów)")
wartosci_k = [1, 3, 5, 10]
acc_tr, acc_te = [], []

for k in wartosci_k:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train_s)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test_s)))
    print(f"  k={k:2d}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("KNN — wpływ liczby sąsiadów", "n_neighbors",
       wartosci_k, acc_tr, acc_te, "knn_n_neighbors.png")

print("\nParametr: metric (metryka odległości)")
metryki = ["euclidean", "manhattan", "chebyshev", "minkowski"]
acc_tr, acc_te = [], []

for met in metryki:
    model = KNeighborsClassifier(n_neighbors=5, metric=met)
    model.fit(X_train_s, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train_s)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test_s)))
    print(f"  metric={met}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("KNN — wpływ metryki odległości", "metric",
       metryki, acc_tr, acc_te, "knn_metric.png")

print("\nParametr: weights (funkcja wagowania sąsiadów)")

def waga_dist2(odleglosci):
    return 1 / (odleglosci ** 2 + 1e-10)

def waga_dist05(odleglosci):
    return 1 / (odleglosci ** 0.5 + 1e-10)

funkcje_wag = [
    ("uniform",   "uniform"),
    ("distance",  "distance"),
    ("1/d²",      waga_dist2),
    ("1/√d",      waga_dist05),
]
acc_tr, acc_te = [], []
etykiety_w = []

for etykieta, w in funkcje_wag:
    model = KNeighborsClassifier(n_neighbors=5, weights=w)
    model.fit(X_train_s, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train_s)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test_s)))
    etykiety_w.append(etykieta)
    print(f"  weights={etykieta:8}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("KNN — wpływ funkcji wagowania sąsiadów", "weights",
       etykiety_w, acc_tr, acc_te, "knn_weights.png")

print("\n" + "=" * 60)
print("METODA 2: Random Forest")
print("=" * 60)

print("\nParametr: n_estimators (liczba drzew w lesie)")
n_drzew = [10, 50, 100, 200]
acc_tr, acc_te = [], []

for n in n_drzew:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test)))
    print(f"  n_estimators={n:3d}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("Random Forest — wpływ liczby drzew", "n_estimators",
       n_drzew, acc_tr, acc_te, "rf_n_estimators.png")

print("\nParametr: max_depth (maksymalna głębokość drzewa)")
glebokosci = [3, 5, 10, None]
acc_tr, acc_te = [], []

for g in glebokosci:
    model = RandomForestClassifier(n_estimators=100, max_depth=g, random_state=42)
    model.fit(X_train, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test)))
    print(f"  max_depth={str(g):4}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("Random Forest — wpływ głębokości drzewa", "max_depth",
       ["3", "5", "10", "None"], acc_tr, acc_te, "rf_max_depth.png")

print("\nParametr: min_samples_leaf (min. próbek w liściu)")
min_liscie = [1, 5, 10, 20]
acc_tr, acc_te = [], []

for ml in min_liscie:
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=ml, random_state=42)
    model.fit(X_train, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test)))
    print(f"  min_samples_leaf={ml:2d}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("Random Forest — wpływ min. próbek w liściu", "min_samples_leaf",
       min_liscie, acc_tr, acc_te, "rf_min_samples_leaf.png")

print("\n" + "=" * 60)
print("METODA 3: SVM — Support Vector Machine")
print("=" * 60)

print("\nParametr: kernel (rodzaj jądra)")
jadra = ["linear", "rbf", "poly", "sigmoid"]
acc_tr, acc_te = [], []

for j in jadra:
    model = SVC(kernel=j, random_state=42)
    model.fit(X_train_s, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train_s)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test_s)))
    print(f"  kernel={j:7}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("SVM — wpływ rodzaju jądra", "kernel",
       jadra, acc_tr, acc_te, "svm_kernel.png")

print("\nParametr: C (regularyzacja)")
wartosci_C = [0.1, 1, 10, 100]
acc_tr, acc_te = [], []

for c in wartosci_C:
    model = SVC(kernel="rbf", C=c, random_state=42)
    model.fit(X_train_s, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train_s)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test_s)))
    print(f"  C={c:5}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("SVM — wpływ parametru C (kernel=rbf)", "C",
       wartosci_C, acc_tr, acc_te, "svm_C.png")

print("\nParametr: degree (stopień wielomianu, kernel=poly)")
stopnie = [2, 3, 4, 5]
acc_tr, acc_te = [], []

for deg in stopnie:
    model = SVC(kernel="poly", degree=deg, random_state=42)
    model.fit(X_train_s, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train_s)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test_s)))
    print(f"  degree={deg}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("SVM — wpływ stopnia wielomianu (kernel=poly)", "degree",
       stopnie, acc_tr, acc_te, "svm_degree.png")

print("\nParametr: criterion (kryterium podziału węzłów)")
kryteria_test = [("gini", None), ("entropy", None), ("log_loss", None), ("gini", 5)]
acc_tr, acc_te = [], []
etykiety_kr = []

for kr, md in kryteria_test:
    model = DecisionTreeClassifier(criterion=kr, max_depth=md, random_state=42)
    model.fit(X_train, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test)))
    etykiety_kr.append(f"{kr}\ndepth={md}")
    print(f"  criterion={kr:9}, max_depth={str(md):4}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("Decision Tree — wpływ kryterium podziału", "criterion",
       etykiety_kr, acc_tr, acc_te, "dt_criterion.png")

print("\nParametr: max_depth (maksymalna głębokość drzewa)")
dt_glebokosci = [3, 5, 10, None]
acc_tr, acc_te = [], []

for g in dt_glebokosci:
    model = DecisionTreeClassifier(max_depth=g, random_state=42)
    model.fit(X_train, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test)))
    print(f"  max_depth={str(g):4}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("Decision Tree — wpływ maksymalnej głębokości", "max_depth",
       ["3", "5", "10", "None"], acc_tr, acc_te, "dt_max_depth.png")

print("\nParametr: min_samples_split (min. próbek do podziału)")
min_split = [2, 5, 10, 20]
acc_tr, acc_te = [], []

for ms in min_split:
    model = DecisionTreeClassifier(min_samples_split=ms, random_state=42)
    model.fit(X_train, y_train)
    acc_tr.append(accuracy_score(y_train, model.predict(X_train)))
    acc_te.append(accuracy_score(y_test,  model.predict(X_test)))
    print(f"  min_samples_split={ms:2d}: trening={acc_tr[-1]:.3f}, test={acc_te[-1]:.3f}")

wykres("Decision Tree — wpływ min_samples_split", "min_samples_split",
       min_split, acc_tr, acc_te, "dt_min_samples_split.png")

wyniki = {
    "KNN (k=5, euclidean)":
        accuracy_score(y_test, KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train).predict(X_test_s)),
    "Random Forest (100 drzew)":
        accuracy_score(y_test, RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train).predict(X_test)),
    "SVM (kernel=linear)":
        accuracy_score(y_test, SVC(kernel="linear", random_state=42).fit(X_train_s, y_train).predict(X_test_s)),
    "SVM (kernel=rbf, C=10)":
        accuracy_score(y_test, SVC(kernel="rbf", C=10, random_state=42).fit(X_train_s, y_train).predict(X_test_s)),
    "Decision Tree (max_depth=10)":
        accuracy_score(y_test, DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train, y_train).predict(X_test)),
}

print("\nAccuracy na zbiorze testowym")
for nazwa, acc in sorted(wyniki.items(), key=lambda x: -x[1]):
    print(f"  {acc:.4f}  {nazwa}")