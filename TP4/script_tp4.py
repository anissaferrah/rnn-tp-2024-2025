import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time 

# données
data = np.genfromtxt('data.csv', delimiter=',', dtype=float)
data.shape

# rajoutons l'ordonnée à l'origine theta 0
intercept=np.ones((data.shape[0],1))
X=np.column_stack((intercept,data[:,0:2]))
y = data[:, 2]
# forcer y à avoir une seule colonne
y = y.reshape( y.shape[0], 1)

print('X', X.shape ,' y ', y.shape)

def mapping(X):
    
    cols = 28
    degree=7
    outX= np.ones((X.shape[0],cols))
    X1=X[:,1]
    X2=X[:,2]
    k=0
    for i in range(degree):
        for j in range(i+1):
            outX[:, k] = np.power(X1,i-j)*(np.power(X2,j))
            k=k+1
    return outX    

X2=mapping(X)
X2.shape



def Sigmoid(z):#--------------------------------------------------------------------------------------------------------------------------
    # pour une valeur donnée, cette fonction calculera sa sigmoid
    """
    Calcule la fonction sigmoïde de l'entrée z.

    Paramètre :
    - z : un scalaire, un vecteur ou une matrice (valeurs réelles)

    Retourne :
    - La transformation sigmoïde de z, avec des valeurs entre 0 et 1
    """
    return 1/(1+np.exp(-z))



def computeCostReg(X, y, theta, lambda_):#---------------------------------------------------------------------------------------------
    """
    Calcule le coût (J) pour la régression logistique régularisée.

    Paramètres :
    - X : matrice des caractéristiques (m exemples, n caractéristiques)
    - y : vecteur des étiquettes (m x 1)
    - theta : vecteur des paramètres (n x 1)
    - lambda_ : paramètre de régularisation
    - la regularistaion sert à éviter le sur-apprentissage (overfitting)

    Retour :
    - J : coût régularisé (valeur scalaire)
    """
    m = y.shape[0]  # nombre d'exemples

    # Hypothèses (probabilités prédites)
    h = 1 / (1 + np.exp(-X.dot(theta)))  # = Sigmoid(X @ theta)

    # Coût sans régularisation
    cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))

    # Régularisation (on exclut theta[0])
    reg = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))

    # Coût total
    J = cost + reg

    return J.item()  # .item() pour retourner un scalaire propre


def gradientDescent(X, y, theta, alpha, iterations, lambda_):#-----------------------------------------------------------------
    # garder aussi le cout à chaque itération 
    # pour afficher le coût en fonction de theta0 et theta1
    """
    Effectue la descente du gradient pour la régression logistique régularisée.

    Paramètres :
    - X : matrice des caractéristiques (m x n)
    - y : vecteur des étiquettes (m x 1)
    - theta : vecteur des paramètres (n x 1)
    - alpha : taux d’apprentissage
    - iterations : nombre d’itérations à effectuer
    - lambda_ : paramètre de régularisation

    Retour :
    - theta : paramètres optimisés
    - J_history : liste des coûts à chaque itération (pour visualisation)
    """
    m = y.shape[0]
    J_history = []

    for i in range(iterations):
        h = 1 / (1 + np.exp(-X.dot(theta)))  # prédictions

        # Erreur
        error = h - y

        # Gradient
        gradient = (1/m) * X.T.dot(error)
        # Régularisation (on ne régularise pas theta[0])
        reg = (lambda_/m) * theta
        reg[0] = 0  # pas de régularisation pour le biais

        # Mise à jour de theta
        theta = theta - alpha * (gradient + reg)

        # Sauvegarde du coût pour analyse
        cost = computeCostReg(X, y, theta, lambda_)
        J_history.append(cost)

    return theta, J_history



#initialisation de theta0 et theta1------------------------------------------
n=X.shape[1]
theta = np.zeros((n, 1))
theta


#calcule du coût initial-------------------------------------------------
lambda_ = 1
initialCost=computeCostReg(X, y, theta, lambda_)
print('Coût initial :')
print(initialCost)


#appel de la fonction gradientDescent-------------------------------------------------
# paramètres
iterations = 1500
alpha = 0.01

# paramètre de regression
lambdaa = 1

# Appel
start_time = time.time()
theta, J_history = gradientDescent(X, y, theta, alpha, iterations, lambdaa)
end_time = time.time()
execution_time_mm = (end_time - start_time) * 1000
# Affichage du coût final
print("coût finale :", J_history[-1])
print("Temps d'exécution (en ms) :", execution_time_mm)


# Traçage du coût en fonction des itérations---------------------------------------------------------------------
plt.plot(range(1, len(J_history) + 1), J_history, color='b', linestyle='-', marker='o')
plt.xlabel('Itérations')
plt.ylabel('Coût J(θ)')
plt.title('Convergence du coût pendant la descente du gradient')
plt.grid(True)
plt.show()



# Traçage de la frontière de décision et des données------------------------------------------------------------
def drawCircle(X, y, theta):
    """
    Dessine les données (avec des cercles) et la frontière de décision pour la régression logistique.
    
    Paramètres :
    - X : matrice des caractéristiques (m x n) où n=2 dans le cas 2D
    - y : vecteur des étiquettes (m x 1)
    - theta : vecteur des paramètres du modèle (n x 1)
    """
    
    # On sépare les exemples de classe 0 et classe 1
    pos = y == 1  # classe 1
    neg = y == 0  # classe 0
    
    # Affichage des exemples (en utilisant des cercles)
    plt.scatter(X[pos, 1], X[pos, 2], marker='o', label='Classe 1', color='b')  # Classe 1 en bleu
    plt.scatter(X[neg, 1], X[neg, 2], marker='x', label='Classe 0', color='r')  # Classe 0 en rouge
    
    # Tracer la frontière de décision (quand h(x) = 0.5)
    # Pour simplifier, on peut imaginer une frontière linéaire dans le cas de X 2D
    # La condition de la frontière est theta_0 + theta_1 * x1 + theta_2 * x2 = 0
    # Réorganiser pour obtenir x2 en fonction de x1 : x2 = (-theta_0 - theta_1 * x1) / theta_2
    
    # Création d'une gamme de valeurs x1 pour dessiner la ligne
    x1_vals = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    # Calcul de x2 en fonction de x1 (frontière de décision)
    x2_vals = (-theta[0] - theta[1] * x1_vals) / theta[2]
    
    # Affichage de la frontière de décision
    plt.plot(x1_vals, x2_vals, label='Frontière de décision', color='g', linewidth=2)
    
    # Paramètres d'affichage
    plt.xlabel('Test 1 (x1)')
    plt.ylabel('Test 2 (x2)')
    plt.legend()
    plt.title('Limite de décision et données')
    plt.grid(True)
    plt.show()
    
 #  Prédit les classes en fonction du seuil donné.----------------------------------------------------------- 
def predict(X, theta, seuil=0.5):
    """
    Prédit les classes en fonction du seuil donné.
    
    Paramètres :
    - X : matrice des caractéristiques (m x n), avec m exemples et n caractéristiques
    - theta : vecteur des paramètres (n x 1)
    - seuil : seuil à partir duquel la classification est effectuée (par défaut 0.5)
    
    Retourne :
    - y_pred : vecteur des prédictions de classe (m x 1), où 1 = classe positive, 0 = classe négative
    """
    # Calcul des prédictions h_theta(x) = sigmoid(X * theta)
    h = Sigmoid(np.dot(X, theta))  # X est de taille (m x n) et theta est de taille (n x 1)
    
    # Appliquer le seuil pour prédire la classe
    y_pred = (h >= seuil).astype(int)  # Si h >= seuil, classe 1, sinon classe 0
    
    return y_pred
# Appel de drawCircle pour afficher la frontière de décision
drawCircle(X, y.ravel(), theta)




# tracage du cout en fonction de theta0 et theta1------------------------------------------------------------

# Étape 1 : créer une grille de valeurs pour theta0 et theta1
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-10, 10, 100)

# Initialiser la matrice de coût
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# On garde les autres theta fixes (ou à 0 si il y a plus de 2 variables)
# ici on suppose que theta a au moins 3 éléments : theta0, theta1, theta2,...
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.copy(theta)
        t[0] = theta0_vals[i]
        t[1] = theta1_vals[j]
        # les autres thetas restent comme dans l'entraînement
        J_vals[i, j] = computeCostReg(X, y, t, lambdaa)

# Étape 2 : Tracer la surface 3D
theta0_vals_mesh, theta1_vals_mesh = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure(figsize=(12, 5))

# Graphe 3D
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(theta0_vals_mesh, theta1_vals_mesh, J_vals.T, cmap='viridis')
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel('Coût J(θ)')
ax.set_title('Surface du coût')

# Graphe de contour
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contour(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-1, 3, 20), cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel(r'$\theta_0$')
ax2.set_ylabel(r'$\theta_1$')
ax2.set_title('Contours du coût J(θ)')
ax2.plot(theta[0], theta[1], 'rx', markersize=10, label='Minimum')
ax2.legend()

plt.tight_layout()
plt.show()




#------------------------------------------------------------------------------------------
# Prédiction
y_pred = predict(X, theta)

# Calcul de la précision (pourcentage de bonnes prédictions)
precision = np.mean(y == y_pred) * 100

print(f"Précision du classifieur : {precision:.2f}%")






#comparatif avec lalgo sklearn====================================================================================================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# ================================
# 1. Préparation des données (avec mapping comme le tien)
# ================================

X_sklearn = mapping(X)  # On utilise le même mapping que toi
y_sklearn = y.ravel()   # scikit-learn attend un vecteur 1D pour y

# ================================
# 2. Modèle scikit-learn
# ================================

# C = 1/lambda_, donc ici lambda_ = 1 => C = 1.0
start_time_sklearn = time.time()
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1500)

# Entraînement
model.fit(X_sklearn, y_sklearn)

# Prédictions
y_pred_sklearn = model.predict(X_sklearn)
y_proba_sklearn = model.predict_proba(X_sklearn)[:, 1]

end_time_sklearn = time.time()
execution_time_sklearn_sl = (end_time_sklearn - start_time_sklearn) * 1000
# ================================
# 3. Évaluation
# ================================

# Coût initial (avec theta = 0)
theta_init = np.zeros(X_sklearn.shape[1])
initial_cost_sklearn = log_loss(y_sklearn, 1 / (1 + np.exp(-X_sklearn.dot(theta_init))))

# Coût final
final_cost_sklearn = log_loss(y_sklearn, y_proba_sklearn)

# Précision
accuracy_sklearn = accuracy_score(y_sklearn, y_pred_sklearn) * 100

# ================================
# 4. Affichage comparatif
# ================================

print("\n--- Comparaison ---")
print(f"Coût initial (mon modèle)      : {initialCost:.4f}")
print(f"Coût initial (scikit-learn)    : {initial_cost_sklearn:.4f}")
print(f"Coût final  (mon modèle)       : {J_history[-1]:.4f}")
print(f"Coût final  (scikit-learn)     : {final_cost_sklearn:.4f}")
print(f"Précision  (mon modèle)        : {precision:.2f}%")
print(f"Précision  (scikit-learn)      : {accuracy_sklearn:.2f}%")
print(f"Temps d'exécution (mon modèle) : {execution_time_mm:.2f} ms")
print(f"Temps d'exécution (scikit-learn): {execution_time_sklearn_sl:.2f} ms")
