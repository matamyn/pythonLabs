import xgboost
from catboost import CatBoostClassifier
import numpy as np
import csv
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt


def load_blood_transfusion():
    data = []
    target = []
    with open('./datasets/transfusion.data.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            values = [float(val) for val in row]
            data.append(values[:-1])
            target.append(values[-1])

    bunch = Bunch()
    bunch.data = data
    bunch.target = target
    return bunch


def calc_hyperparams(classifier, params, x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.75, random_state=0)

    scoring = {'Accuracy': 'accuracy', 'Precision': 'precision', 'AUC': 'roc_auc', 'Recall': 'recall',
               'F1_score': 'f1_macro'}

    model = GridSearchCV(classifier, params, scoring=scoring, cv=2, refit='AUC')
    model.fit(train_x, train_y)

    print("Best parameters set found on development set:")
    print(model.best_params_)

    print('Accuracy: ', model.cv_results_['mean_test_Accuracy'][model.best_index_])
    print('Precision: ', model.cv_results_['mean_test_Precision'][model.best_index_])
    print('Recall: ', model.cv_results_['mean_test_Recall'][model.best_index_])
    print('AUC: ', model.cv_results_['mean_test_AUC'][model.best_index_])
    print('F1_score: ', model.cv_results_['mean_test_F1_score'][model.best_index_])

    return model.best_params_


dataset = load_blood_transfusion()
x = dataset.data
y = dataset.target
iterations = 500

# CatBoost example
'''classifier = CatBoostClassifier(iterations=iterations, thread_count=5)
hyper_params = calc_hyperparams(classifier, {
    'learning_rate': [0.5],
    'depth': [6]
}, x, y)

model = CatBoostClassifier(iterations=iterations, **hyper_params)'''

classifier = XGBClassifier()
'''hyper_params = calc_hyperparams(classifier, {
    'n_estimators': [1, 2, 3, 9, 12, 15, 30, 50, 100, 150, 300, 500, 1000],
    'learning_rate': [0.00000000001, 0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001],
    'max_depth': [3, 5, 6, 9, 12, 15, 30]
}, np.matrix(x), np.array(y))'''
hyper_params = calc_hyperparams(classifier, {
    'n_estimators': [9],
    'learning_rate': [0.0000001],
    'max_depth': [9]
}, np.matrix(x), np.array(y))

model = XGBClassifier(**hyper_params)
model.fit(np.matrix(x), np.array(y))

with open("model.json", "w") as f:
   f.write(model._Booster.get_dump(dump_format='json')[0])
plot_tree(model)
plt.show()