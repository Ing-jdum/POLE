from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics as skm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

# Autor: John Urena
# Correo: ing.jdum@gmail.com

def plot_confusion_matrix(target, classifier, dataset):
    """
    Imprime la matriz de confusión utilizando en el cmd.

    Parameters:
    - target: Etiquetas de clase del conjunto de datos completo
    - classifier: Clasificador entrenado
    - dataset: Conjunto de características completo

    Returns:
    - None
    """
    cm = confusion_matrix(target, classifier.predict(dataset), labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.show()


def print_confusion_matrices(classifier, X_train, y_train, X_test, y_test, dataset, target):
    """
    Imprime matrices de confusión para conjuntos de entrenamiento, prueba y todo el conjunto de datos.

    Parameters:
    - classifier: Clasificador entrenado
    - X_train: Conjunto de características de entrenamiento
    - y_train: Etiquetas de clase de entrenamiento
    - X_test: Conjunto de características de prueba
    - y_test: Etiquetas de clase de prueba
    - dataset: Conjunto de características completo
    - target: Etiquetas de clase del conjunto de datos completo

    Returns:
    - None
    """
    print('### Matrices de confusión ###\n')

    # Matriz de confusión para el conjunto de entrenamiento
    print('Entrenamiento:')
    print(skm.confusion_matrix(y_train, classifier.predict(X_train)))

    # Matriz de confusión para el conjunto de prueba
    print('\nPruebas:')
    print(skm.confusion_matrix(y_test, classifier.predict(X_test)))

    # Matriz de confusión para todo el conjunto de datos
    print('\nTodo:')
    print(skm.confusion_matrix(target, classifier.predict(dataset)))


def print_classifier_scores(classifier, X_train, y_train, X_test, y_test, dataset, target):
    """
    Imprime el rendimiento del clasificador en el conjunto de entrenamiento, prueba y todo el conjunto de datos.

    Parameters:
    - classifier: Clasificador entrenado
    - X_train: Conjunto de características de entrenamiento
    - y_train: Etiquetas de clase de entrenamiento
    - X_test: Conjunto de características de prueba
    - y_test: Etiquetas de clase de prueba
    - dataset: Conjunto de características completo
    - target: Etiquetas de clase del conjunto de datos completo

    Returns:
    - None
    """
    print("\n")
    print("Rendimiento en el conjunto de entrenamiento: ", classifier.score(X_train, y_train))
    print("Rendimiento en el conjunto de prueba: ", classifier.score(X_test, y_test))
    print("Rendimiento en el conjunto total: ", classifier.score(dataset, target))


def get_best_random_forest_classifier(X_train, y_train, X_test, y_test):
    """
    Trains a RandomForest classifier with the best hyperparameters obtained using GridSearchCV.

    Parameters:
    - X_train: Training feature set
    - y_train: Training class labels
    - X_test: Test feature set
    - y_test: Test class labels

    Returns:
    - best_random_forest_classifier: RandomForest classifier trained with the best hyperparameters
    """

    # Define the parameter grid to explore
    param_grid = {
        'n_estimators': [100],  # number of trees in the forest
        'max_depth': [None, 6, 7],  # maximum depth of the tree
        'min_samples_split': [None, 5, 10],  # minimum number of samples required to split a node
        'min_samples_leaf': [None, 4, 6, 8],  # minimum number of samples required at each leaf node
        'class_weight': ['balanced']  # weights associated with classes
    }

    # Use F1 score as the evaluation metric
    scorer = make_scorer(f1_score, average='weighted')

    # Create a RandomForest classifier
    classifier_rf = RandomForestClassifier()

    # Perform grid search with cross-validation
    grid_search = RandomizedSearchCV(classifier_rf, param_grid, scoring=scorer, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_

    # Create a RandomForest classifier with the best parameters
    best_random_forest_classifier = RandomForestClassifier(**best_params)
    best_random_forest_classifier.fit(X_train, y_train)

    # Evaluate the performance on the separate test set
    training_accuracy = best_random_forest_classifier.score(X_train, y_train)
    test_accuracy = best_random_forest_classifier.score(X_test, y_test)

    print("Best Parameters:", best_params)
    print("Accuracy on Test Set:", test_accuracy)
    print("Accuracy on Training Set:", training_accuracy)

    return best_random_forest_classifier


def find_best_params(param_grid, centrality_measures, X_train, y_train, X_test, y_test):
    """
    Encuentra los mejores parámetros para un conjunto de medidas de centralidad dados los datos de entrenamiento y prueba.

    Args:
        param_grid (list): Una lista de listas binarias que representan diferentes combinaciones de medidas de centralidad a considerar.
        centrality_measures (list): Una lista de las medidas de centralidad disponibles.
        X_train (array-like): Los datos de entrenamiento (características).
        y_train (array-like): Las etiquetas de entrenamiento.
        X_test (array-like): Los datos de prueba (características).
        y_test (array-like): Las etiquetas de prueba.

    Returns:
        None
    """
    for params in param_grid:
        selected_measures = [centrality_measures[i] for i, include_measure in enumerate(params) if include_measure]
        selected_indices = [i for i, include_measure in enumerate(params) if include_measure]

        # Filtra las características según los índices seleccionados
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]

        clf = get_best_random_forest_classifier(X_train_selected, y_train, X_test_selected, y_test)
        if clf.score(X_test_selected, y_test) > 0.85:
            print(selected_measures)

