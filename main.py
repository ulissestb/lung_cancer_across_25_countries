import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report,roc_curve, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling  import SMOTE

dataframe = pd.read_csv('lung_cancer_prediction_dataset.csv')


def avalia_clf(clf, y, X, rótulos_y=['Não Sobreviveu', 'Sobreviveu'], base='treino'):
    # Calcular as classificações preditas
    pred = clf.predict(X)

    # Calcular a probabilidade de evento
    y_prob = clf.predict_proba(X)[:, -1]

    # Calculando acurácia e matriz de confusão
    cm = confusion_matrix(y, pred)
    ac = accuracy_score(y, pred)
    bac = balanced_accuracy_score(y, pred)

    print(f'\nBase de {base}:')
    print(f'A acurácia da árvore é: {ac:.1%}')
    print(f'A acurácia balanceada da árvore é: {bac:.1%}')

    # Calculando AUC
    auc_score = roc_auc_score(y, y_prob)
    print(f"AUC-ROC: {auc_score:.2%}")
    print(f"GINI: {(2 * auc_score - 1):.2%}")

    # Visualização gráfica
    sns.heatmap(cm,
                annot=True, fmt='d', cmap='viridis',
                xticklabels=rótulos_y,
                yticklabels=rótulos_y)

    # Relatório de classificação do Scikit
    print('\n', classification_report(y, pred))

    # Gerar a Curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_prob)

    # Plotar a Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Linha de referência (modelo aleatório)
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title(f"Curva ROC - base de {base}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def map_age(age: int) -> int:
    for x in range(0, 120, 10):
        if age < x:
            return x

def map_cigarettes_per_day(cigars: int) -> int:
    for x in range(0, 30, 5):
        if cigars <= x:
            return x

def map_years_of_smoking(years: int) -> int:
    if years is None or 0: return 0
    for x in range(0, 40, 5):
        if years <= x:
            return x



dataframe.drop(columns=['ID', 'Population_Size', 'Country', 'Mortality_Rate', 'Survival_Years', 'Cancer_Stage', 'Treatment_Type', 'Early_Detection',
                        'Developed_or_Developing', 'Healthcare_Access', 'Adenocarcinoma_Type', 'Occupational_Exposure', 'Annual_Lung_Cancer_Deaths', 'Lung_Cancer_Prevalence_Rate'], inplace=True)


# Tratamento na mão, mas removido para usar o get_dummies

# dataframe['Gender'] = dataframe['Gender'].map({'Female': 0, 'Male': 1})
# dataframe['Smoker'] = dataframe['Smoker'].map({'Yes': 1, 'No': 0})
# dataframe['Passive_Smoker'] = dataframe['Passive_Smoker'].map({'Yes': 1, 'No': 0})
dataframe['Lung_Cancer_Diagnosis'] = dataframe['Lung_Cancer_Diagnosis'].map({'Yes': 1, 'No': 0})
# dataframe['Family_History'] = dataframe['Family_History'].map({'Yes': 1, 'No': 0})
# #dataframe['Occupational_Exposure'] = dataframe['Occupational_Exposure'].map({'Yes': 1, 'No': 1})
# dataframe['Indoor_Pollution'] = dataframe['Indoor_Pollution'].map({'Yes': 1, 'No': 0})
#
# dataframe['Age'] = dataframe['Age'].map(lambda age: map_age(age))
# dataframe['Cigarettes_per_Day'] = dataframe['Cigarettes_per_Day'].map(lambda cigars: map_cigarettes_per_day(cigars))
# dataframe['Years_of_Smoking'] = dataframe['Years_of_Smoking'].map(lambda years: map_years_of_smoking(years))

dataframe['Years_of_Smoking'] = dataframe['Years_of_Smoking'].fillna(0)
dataframe['Cigarettes_per_Day'] = dataframe['Cigarettes_per_Day'].fillna(0)


dummies = pd.get_dummies(data=dataframe, drop_first=True)

y_axis = dummies['Lung_Cancer_Diagnosis']
dummies.drop(columns=['Lung_Cancer_Diagnosis'], inplace=True)
x_axis = dummies

print(f' Y missing:  {y_axis.isnull().sum()}')
print(f'X missing: {x_axis.isnull().sum()}')


x_train, x_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size=0.4, stratify=y_axis, random_state=42)

smote = SMOTE()
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

forest_clf = RandomForestClassifier()

grid_search = GridSearchCV(
    estimator=forest_clf,
    param_grid={
        'n_estimators': [10],
        'max_depth': [6],
        'min_samples_split': [8],
        'min_samples_leaf': [4],
        'max_features': [8],
    },
    scoring='roc_auc',
    cv=10,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train_resampled, y_train_resampled)

print(f'BEST PARAMS: {grid_search.best_params_}')
print(f'BEST SCORE: {grid_search.best_score_}')


best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

importances = best_model.feature_importances_
feature_names = x_axis.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Importância das Features')
plt.show()

best_model.fit(x_train, y_train.values.ravel())

avalia_clf(best_model, y_train, x_train, rótulos_y=['No Cancer', 'Has Cancer'], base='treino')
avalia_clf(best_model, y_test, x_test,rótulos_y=['No Cancer', 'Has Cancer'], base='teste')
