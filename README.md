# 👕 Classifieur de vêtements — Fashion-MNIST

Projet de **Deep Learning / Computer Vision** permettant de classifier automatiquement des images de vêtements à l'aide d'un **réseau de neurones convolutif (CNN)** développé avec **TensorFlow et Keras**.

Le modèle est entraîné sur le dataset **Fashion-MNIST** et atteint environ **91 % de précision** sur le jeu de test.

---

## 🎯 Objectif du projet

L'objectif est de construire un modèle capable de reconnaître automatiquement la catégorie d'un vêtement à partir d'une image en niveaux de gris de taille **28 × 28 pixels**.

Le projet couvre l'ensemble du pipeline de Machine Learning :

* chargement des données ;
* exploration du dataset ;
* prétraitement et normalisation ;
* création d'un CNN ;
* entraînement du modèle ;
* évaluation des performances ;
* matrice de confusion ;
* rapport de classification ;
* sauvegarde du modèle entraîné.

---

## 📊 Dataset Fashion-MNIST

Le dataset **Fashion-MNIST** contient **70 000 images** de vêtements en niveaux de gris.

## 🧹 Prétraitement des données

Les valeurs des pixels, initialement comprises entre **0 et 255**, sont normalisées entre **0 et 1** :

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

Une dimension correspondant au canal de l'image est ensuite ajoutée afin de rendre les données compatibles avec les couches convolutionnelles :

```python
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```

---

## 🧠 Architecture du CNN

Le modèle utilise une architecture CNN relativement légère :

```text
Input (28 × 28 × 1)
        │
        ▼
Conv2D — 32 filtres, 3×3, ReLU
        │
        ▼
MaxPooling2D — 2×2
        │
        ▼
Conv2D — 64 filtres, 3×3, ReLU
        │
        ▼
MaxPooling2D — 2×2
        │
        ▼
Flatten
        │
        ▼
Dense — 64 neurones, ReLU
        │
        ▼
Dense — 10 neurones, Softmax
```

Le modèle possède environ **121 930 paramètres entraînables**.

---

## ⚙️ Configuration de l'entraînement

Le modèle est compilé avec :

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

L'entraînement est réalisé pendant **10 époques** :

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)
```

---

## 📈 Résultats

Après 10 époques, le modèle obtient approximativement :

| Métrique                | Résultat |
| ----------------------- | -------: |
| Accuracy entraînement   |    ~95 % |
| Accuracy test           |    ~91 % |
| Nombre d'images testées |   10 000 |

La précision finale observée sur le jeu de test est d'environ **91,1 %**.

Certaines catégories sont particulièrement bien reconnues, notamment :

* Trouser ;
* Sandal ;
* Sneaker ;
* Bag ;
* Ankle boot.

La classe **Shirt** est plus difficile à distinguer, notamment à cause de sa ressemblance avec les catégories **T-shirt**, **Pullover** et **Coat**.

---

## 📉 Évaluation du modèle

Le notebook contient plusieurs outils permettant d'analyser les performances du CNN.

### Courbe de Loss

Comparaison de la fonction de perte entre les données d'entraînement et de validation au fil des époques.

### Courbe d'Accuracy

Visualisation de l'évolution de la précision pendant l'entraînement.

### Matrice de confusion

Une matrice de confusion permet d'identifier les catégories que le modèle confond le plus souvent.

### Classification Report

Les métriques suivantes sont calculées pour chaque catégorie :

* Precision
* Recall
* F1-score
* Support

---

## 🛠️ Technologies utilisées

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 📂 Structure du projet

```text
Classifieur-de-v-tements-Fashion-MNIST/
│
├── .github/
│   └── workflows/
│
├── Classifieur de vêtements Fashion MNIST.ipynb
│
├── app.py
│
├── fashion_mnist_mini_vgg_model.h5
│
└── README.md
```

### Fichiers principaux

**`Classifieur de vêtements Fashion MNIST.ipynb`**
Notebook contenant le chargement des données, le prétraitement, la création du CNN, l'entraînement et l'évaluation du modèle.

**`app.py`**
Script d'application utilisant le modèle entraîné.

**`fashion_mnist_mini_vgg_model.h5`**
Modèle pré-entraîné sauvegardé au format HDF5.

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/datasifaw/Classifieur-de-v-tements-Fashion-MNIST.git
```

```bash
cd Classifieur-de-v-tements-Fashion-MNIST
```

### 2. Installer les dépendances

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

## ▶️ Utilisation

Lancer Jupyter Notebook :

```bash
jupyter notebook
```

Puis ouvrir :

```text
Classifieur de vêtements Fashion MNIST.ipynb
```

Exécuter ensuite les cellules dans l'ordre pour :

1. charger Fashion-MNIST ;
2. préparer les images ;
3. créer le CNN ;
4. entraîner le modèle ;
5. visualiser ses performances ;
6. générer la matrice de confusion ;
7. afficher le rapport de classification ;
8. sauvegarder le modèle.

---

## 💾 Sauvegarde du modèle

Le modèle peut être sauvegardé avec Keras :

```python
model.save("fashion_mnist_cnn_model.h5")
```

Il peut ensuite être rechargé sans réentraîner le réseau :

```python
from tensorflow.keras.models import load_model

model = load_model("fashion_mnist_cnn_model.h5")
```

---

## 🔮 Améliorations possibles

Plusieurs pistes peuvent être explorées pour améliorer le projet :

* ajout de Dropout pour limiter le surapprentissage ;
* Batch Normalization ;
* Data Augmentation ;
* Early Stopping ;
* optimisation des hyperparamètres ;
* architecture CNN plus profonde ;
* comparaison avec d'autres architectures ;
* interface Web permettant de charger une image et d'obtenir une prédiction ;
* déploiement du modèle avec Streamlit, Flask ou FastAPI ;
* containerisation avec Docker.

---

## ⭐ Projet

Si ce projet vous a été utile ou vous intéresse, n'hésitez pas à lui donner une ⭐ sur GitHub.
