# 🏠 Projet Airbnb - Machine Learning

## 📖 Description

Ce projet vise à **prédire les prix des locations Airbnb** à partir de diverses caractéristiques (localisation, nombre de chambres, équipements, etc.) en utilisant des **techniques de Machine Learning**.

L’objectif est de construire un modèle performant permettant d’estimer le prix optimal d’une annonce en fonction de ses attributs, afin d’aider les hôtes à mieux positionner leur bien sur le marché.

---

## 👩‍💻 Auteurs
- **Cyrille Malongo**
- **Gabriel Maccione**
- **Julien Maronne**
---

## ⚙️ Technologies utilisées

- **Python 3.x**
- **Pandas** – traitement et analyse des données  
- **NumPy** – calculs numériques  
- **Matplotlib / Seaborn** – visualisation des données  
- **Scikit-learn** – modélisation et évaluation des modèles de machine learning  
- **XGBoost / LightGBM** – modèles de boosting performants  
- **GeoPandas / Shapely / Contextily** – analyse et visualisation géographique  
- **Jupyter Notebook** – environnement d’expérimentation

---

## 📂 Structure du projet

```
📦 Projet_Airbnb_Machine_Learning
│
├── Projet_Machine_Learning.ipynb   # Notebook principal
├── data/                           # Jeux de données 
├── README.md                       # Ce fichier
└── requirements.txt                # Liste des dépendances
└── Projet_Machine_Learning.py      # Code Python
```

---

## 🚀 Installation et exécution

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/<ton-utilisateur>/Projet_Airbnb_Machine_Learning.git
   cd Projet_Airbnb_Machine_Learning
   ```

2. **Créer un environnement virtuel (optionnel mais recommandé) :**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (sous Linux/Mac)
   venv\Scripts\activate     # (sous Windows)
   ```

3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer le notebook :**
   ```bash
   jupyter notebook Projet_Machine_Learning.ipynb
   ```

---

## 📊 Résultats attendus

- Nettoyage et préparation du dataset Airbnb  
- Visualisation des corrélations et variables importantes  
- Entraînement de plusieurs modèles (ex. : **Linear Regression**, **Random Forest**, **XGBoost**, **LightGBM**)  
- Évaluation des performances (RMSE, R², etc.)  
- Interprétation des résultats et recommandations  

---

## 📈 Exemple de visualisation

Quelques exemples de graphiques produits dans le notebook :
- Répartition des prix par quartier
- Corrélation entre la taille du logement et le prix
- Importance des variables pour le modèle final

---

## 📜 Licence

Ce projet est distribué sous licence scolaire.  
Vous êtes libres de le réutiliser, le modifier et le partager sous réserve de mentionner les auteurs originaux.
