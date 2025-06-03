import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time


# Données
data = np.genfromtxt('data.csv', delimiter=',')
print("Shape des données :", data.shape)

# Ajout de l'ordonnée à l'origine (theta_0)
intercept = np.ones((data.shape[0], 1))
X = np.column_stack((intercept, data[:, 0]))
print("Shape de X :", X.shape)

# Extraction de y et redimensionnement en vecteur colonne
y = data[:, 1].reshape(-1, 1)  # Forme (97, 1)
print("Shape de y :", y.shape)



# Fonction de calcul du coût avec vectorisation
def computeCost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Fonction de descente du gradient
def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)  # Forme (2, 1)
        theta -= alpha * gradient
        cost = computeCost(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history



# Initialisation des paramètres
theta = np.zeros((2, 1))  # Forme (2, 1)

# Calcul du coût initial
initialCost = computeCost(X, y, theta)
print("Coût initial :", initialCost)

# Paramètres de la descente du gradient
iterations = 1500
alpha = 0.01

print("Appel de la descente du gradient...")

# Appel de la descente du gradient
theta_optimized, cost_history = gradientDescent(X, y, theta, alpha, iterations)

# Affichage des résultats
print("Paramètres optimisés :", theta_optimized)
print("Coût final :", cost_history[-1])



# Tracé de la ligne de régression après entraînement-----------------------------------------------------------------------------------------------------------
y_pred = X.dot(theta_optimized)  # Prédictions du modèle

# Tracé des données réelles
plt.scatter(X[:, 1], y, marker='x', label='Données réelles')
plt.xlabel('Population x 1000')
plt.ylabel('Chiffre d\'affaires x1000')

# Tracé de la ligne de régression
plt.plot(X[:, 1], y_pred, color='red', label='Ligne de régression')

# Affichage
plt.legend()
plt.title('Régression linéaire après entraînement')
plt.show()







#tracage de la courbe de cout---------------------------------------------------------------------------------------------------------
# Création d'une grille de valeurs pour theta_0 et theta_1
theta0_vals = np.linspace(-10, 10, 100)  # Plage de theta0
theta1_vals = np.linspace(-1, 2, 100)    # Plage de theta1

# Initialisation d'une matrice pour stocker les valeurs de la fonction de coût
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Calcul de la fonction de coût pour chaque combinaison de theta0 et theta1
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        theta = np.array([[theta0_vals[i]], [theta1_vals[j]]])  # Vecteur theta
        J_vals[i, j] = computeCost(X, y, theta)  # Calcul du coût

# Conversion en meshgrid pour le tracé 3D
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# Tracé en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
ax.set_xlabel('Theta 0 (θ₀)')
ax.set_ylabel('Theta 1 (θ₁)')
ax.set_zlabel('Coût (J)')
ax.set_title('Fonction de coût en 3D selon teta0 et teta1')
plt.show()



# Tracé du coût en fonction du nombre d'itérations on utilisera lhistorique des couts 
plt.plot(range(iterations), cost_history, color='blue')
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Coût (J)')
plt.title('Coût en fonction du nombre d\'itérations')
plt.grid(True)
plt.show()


#predire les valeurs de y (benefice) pour x=35000 et x=70000---------------------------------------------------------------------------------------------------------
# Prédictions
# Prédire pour une population de 35 000
predict1 = np.matmul([1, 3.5], theta_optimized)
print("Prédiction du benefice pour une population de 35 000 :", predict1[0])

# Prédire pour une population de 70 000
predict2 = np.matmul([1, 7], theta_optimized)
print("Prédiction du benefice pour une population de 70 000 :", predict2[0])






#---------------------------------------Régression linéaire avec plusieurs variables---------------------------------------------------------------------------------------------------------
#---------------------------------------Régression linéaire avec plusieurs variables---------------------------------------------------------------------------------------------------------
#---------------------------------------Régression linéaire avec plusieurs variables---------------------------------------------------------------------------------------------------------


print("==================================Régression linéaire avec plusieurs variables ==========================================================================")
print("==================================Régression linéaire avec plusieurs variables ==========================================================================")
# Chargement des données
dataMulti = np.genfromtxt('dataMulti.csv', delimiter=',')
X = dataMulti[:, 0:2]  # Caractéristiques (sans la colonne de 1)
y = dataMulti[:, 2].reshape(-1, 1)  # Variable cible

print("Shape de X :", X.shape)
print("Shape de y :", y.shape)
print("Première ligne de X :", X[0])

# Fonction de calcul du coût et du gradient sont les mêmes que précédemment mais avec la vectorisation


# Fonction de normalisation
def featureNormalization(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Ajout de la colonne de 1 pour theta_0
X_with_intercept = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

# Normalisation des caractéristiques
X_norm, mu, sigma = featureNormalization(X)
X_norm_with_intercept = np.concatenate([np.ones((X_norm.shape[0], 1)), X_norm], axis=1)

# Initialisation des paramètres
theta_without_norm = np.zeros((X_with_intercept.shape[1], 1))
theta_with_norm = np.zeros((X_norm_with_intercept.shape[1], 1))

# Paramètres de la descente du gradient
iterations = 1500
alpha = 0.01

# Descente du gradient SANS normalisation
start_time = time.time()
theta_without_norm_optimized, cost_history_without_norm = gradientDescent(
    X_with_intercept, y, theta_without_norm, alpha, iterations
)
time_without_norm = time.time() - start_time

# Descente du gradient AVEC normalisation
start_time = time.time()
theta_with_norm_optimized, cost_history_with_norm = gradientDescent(
    X_norm_with_intercept, y, theta_with_norm, alpha, iterations
)
time_with_norm = time.time() - start_time

# Affichage des résultats
print("=== Sans normalisation ===")
print("Paramètres optimisés :", theta_without_norm_optimized)
print("Coût final :", cost_history_without_norm[-1])
print("Temps d'exécution :", time_without_norm, "secondes")

print("\n=== Avec normalisation ===")
print("Paramètres optimisés :", theta_with_norm_optimized)
print("Coût final :", cost_history_with_norm[-1])
print("Temps d'exécution :", time_with_norm, "secondes")


#comparaison de mes algorithmes de descente du gradient avec scikitlearn ---------------------------------------------------------------------------------------------------------

start_time = time.time()
# Création du modèle de régression linéaire de scikit-learn
# fit_intercept=False car on a déjà ajouté une colonne de 1 pour l'interception
model_with_norm = LinearRegression(fit_intercept=False)

# Entraînement du modèle avec les données normalisées
model_with_norm.fit(X_norm_with_intercept, y)

# Calcul du temps total d'exécution
time_sklearn_with_norm = time.time() - start_time

# Récupération des paramètres optimisés (les coefficients du modèle)
# model_with_norm.coef_ est un tableau de (1, n), donc on le transpose pour correspondre à la forme (n, 1)
theta_sklearn_with_norm = model_with_norm.coef_.T

# Calcul du coût final en utilisant notre fonction computeCost déjà définie
cost_sklearn_with_norm = computeCost(X_norm_with_intercept, y, theta_sklearn_with_norm)

# Affichage des résultats pour scikit-learn
print("\n=== Scikit-learn (Avec normalisation) ===")
print("Paramètres optimisés :", theta_sklearn_with_norm.flatten())  # Affichage des coefficients optimisés
print("Coût final :", cost_sklearn_with_norm)  # Affichage du coût final après entraînement
print("Temps d'exécution :", time_sklearn_with_norm, "secondes")  # Affichage du temps d'exécution total