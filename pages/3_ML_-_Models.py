import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

st.set_page_config(
    page_title='Visualisation des données',
    layout='wide'
)

header_container = st.container()
stats_container = st.container()
conclusion_container = st.container()

feature_outliers = [
    'Kurtosis',
    'glrlm_LongRunHighGrayLevelEmphasis',
    'gzlm_SizeZoneNonUniformityNormalized',
    'MajorAxisLength',
    'Imc2',
    'TotalEnergy',
    'gzlm_LargeAreaLowGrayLevelEmphasis',
    'Idn',
    'gzlm_SmallAreaLowGrayLevelEmphasis',
    'gzlm_GrayLevelVariance'
]

feature_non_outliers = ['InverseVariance',
    'Range',
    'MaximumProbability',
    'Maximum3DDiameter',
    'Maximum2DDiameterColumn',
    'Entropy',
    'Maximum2DDiameterRow']

feature_to_keep = ["MGMT_value"]

with header_container:

	# for example a logo or a image that looks like a website header
    col1, col2, col3 = st.columns([2,4,2])

    st.title("Machine Learning Models")

    with col1:
        st.write("")

    with col2:
        st.image('images/brain_tumor.png')

    with col3:
        st.write("")


with stats_container:

    st.header('Prédictions avec des algorithmes de Machine learning')
    """
    Nous avons testé différents algorithmes de machine learning pour essayer de réaliser des prédictions. Notre ensemble de données d'entraînement se compose de 585 patients, ce qui est relativement limité pour l'apprentissage automatique. Un ensemble de données plus important fournirait des résultats plus robustes.
    Malheureusement nous ne sommes pas parvenus à trouver un modèle de Machine Learning ayant un niveau de précision satisfaisant. Dans cette partie, vous pourrez tester différents modèles de Machine Learning sur le dataset que nous avons généré après sélection et normalisation de nos variables.
    """


   #RFE RANDOM FOREST

    st.subheader('Premier modèle : random forest classifier')
    """
    Nous avons testé un premier modèle de type Random Forest Classifier, après sélection de nos variables comme expliqué dans la partie "Visualisation du jeu de données". Nous avons appliqué à ces variables une normalisation de type RobustScaler et StandardScaler et un algorithme de bagging.
    """

    sel_col, dis_col = st.columns(2)

    target = pd.read_csv('datasets/target.csv')
    features = pd.read_csv('datasets/features.csv')

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123)

    pipeline_steps = []

    preprocessing_name = "preprocessing"
    transformers = []
    transformers.append((f"ze_a_transformer", RobustScaler(), feature_outliers))
    transformers.append((f"ze_b_transformer", StandardScaler(), feature_non_outliers))

    preprocessor = ColumnTransformer(transformers)
    pipeline_steps.append((preprocessing_name, preprocessor))


    n_estimators = sel_col.select_slider(
        'Veuillez choisir la valeur de n_estimators',
        options=[50, 100, 200],
        key="n_estimators_rf")

    max_samples = sel_col.select_slider(
        'Veuillez choisir la valeur de max_samples',
        options=[0.5, 0.8, 1.0],
        key="max_samples_rf")

    max_features = sel_col.select_slider(
        'Veuillez choisir la valeur de max_features',
        options=[0.5, 0.8, 1.0],
        key="max_features_rf")

    max_depth = sel_col.select_slider(
        'Veuillez choisir la valeur de max_depth',
        options=[None, 5, 10],
        key="max_depth_rf")

    min_samples_split = sel_col.select_slider(
        'Veuillez choisir la valeur de min_samples_split',
        options=[2, 5, 10],
        key="min_samples_split_rf")

    min_samples_leaf = sel_col.select_slider(
        'Veuillez choisir la valeur de min_samples_leaf',
        options=[1, 2, 4],
        key="min_samples_leaf_rf")


    model_name = "model"
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model = BaggingClassifier(base_estimator=random_forest_model)

    pipeline_steps.append((model_name, model))
    pipeline = Pipeline(steps=pipeline_steps)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    dis_col.subheader("La précision de notre modèle est :")
    dis_col.write(accuracy_score(y_test, y_pred))

    dis_col.subheader("L'erreur quadratique moyenne de notre modèle est :")
    dis_col.write(mean_squared_error(y_test, y_pred))

    dis_col.subheader("L'erreur absolue moyenne de notre modèle est :")
    dis_col.write(mean_absolute_error(y_test, y_pred))

    dis_col.subheader("Le R score de notre modèle est :")
    dis_col.write(r2_score(y_test, y_pred))

    dis_col.subheader("Le rappel de notre modèle est : ")
    dis_col.write(recall_score(y_test, y_pred))


    #PCA RANDOM FOREST

    st.markdown('#')

    st.subheader('Deuxième modèle : Utilisation de PCA pour la réduction de dimension et utilisation d\'un modèle Random Forest')
    """
    Nous avons testé une autre méthode de dimension : l'analyse en composantes principales (PCA). Après avoir divisé nos données en un ensemble d'entraînement et un ensemble de test, nous avons testé un ensemble de modèles grâce à un StratifiedKFold.
    Parmis les modèles que nous avons testé, nous avons choisi de vous présenter un modèle de type Random Forest, et un modèle de régresssion logistique, mais nous avons également testé un Support Vector Machine (SVM), et un XGBoost.
    """

    sel_col, dis_col = st.columns(2)

    target_pca = pd.read_csv('datasets/target_pca.csv')
    target_pca = target_pca["MGMT_value"]
    features_pca = pd.read_csv('datasets/features_pca.csv')
    features_pca = features_pca[features_pca.columns[~features_pca.columns.isin(['ID'])]]


    pca = PCA()
    pca = PCA(n_components=10)
    X = pca.fit_transform(features_pca)

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X, target_pca, test_size=0.2, random_state=42)

    n_estimators_pca = sel_col.select_slider(
        'Veuillez choisir la valeur de n_estimators',
        options=[50, 100, 200],
        key="n_estimators_pca")

    max_samples_pca = sel_col.select_slider(
        'Veuillez choisir la valeur de max_samples',
        options=[0.5, 0.8, 1.0],
        key="max_samples_pca")

    max_features_pca = sel_col.select_slider(
        'Veuillez choisir la valeur de max_features',
        options=[0.5, 0.8, 1.0],
        key="max_features_pca")

    max_depth_pca = sel_col.select_slider(
        'Veuillez choisir la valeur de max_depth',
        options=[3, 5, 7,20],
        key="max_depth_pca")

    min_samples_split_pca = sel_col.select_slider(
        'Veuillez choisir la valeur de min_samples_split',
        options=[2, 5, 10],
        key="min_samples_split_pca")

    min_samples_leaf_pca = sel_col.select_slider(
        'Veuillez choisir la valeur de min_samples_leaf',
        options=[1, 2, 4],
        key="min_samples_leaf_pca")


    random_forest_model_pca = RandomForestClassifier(n_estimators=n_estimators_pca, max_samples=max_samples_pca, max_features=max_features_pca, max_depth=max_depth_pca, min_samples_split=min_samples_split_pca, min_samples_leaf=min_samples_leaf_pca)
    random_forest_model_pca.fit(X_train_pca, y_train_pca)

    y_pred_pca = random_forest_model_pca.predict(X_test_pca)

    dis_col.subheader("La précision de notre modèle est :")
    dis_col.write(accuracy_score(y_test_pca, y_pred_pca))

    dis_col.subheader("L'erreur quadratique moyenne de notre modèle est :")
    dis_col.write(mean_squared_error(y_test_pca, y_pred_pca))

    dis_col.subheader("L'erreur absolue moyenne de notre modèle est :")
    dis_col.write(mean_absolute_error(y_test_pca, y_pred_pca))

    dis_col.subheader("Le R score de notre modèle est :")
    dis_col.write(r2_score(y_test_pca, y_pred_pca))

    dis_col.subheader("Le rappel de notre modèle est : ")
    dis_col.write(recall_score(y_test_pca, y_pred_pca))


    #PCA LOGISTIC REGRESSION

    st.markdown('#')

    st.subheader('Troisième modèle : Régression logitique')
    """
    Parmis les modèles que nous avons testé, celui de la régression logistique fournissait des résultats intéressants pour le score de rappel.
    """

    sel_col, dis_col = st.columns(2)

    penalty = sel_col.select_slider(
        'Veuillez choisir la valeur de penalty',
        options=[None, 'l2'],
        key="penalty")

    c_param = sel_col.select_slider(
        'Veuillez choisir la valeur de C',
        options=[0.001,0.01,0.1, 1.0, 10.0],
        key="C")


    logistic_regression_model_pca = LogisticRegression(penalty=penalty, C=c_param)
    logistic_regression_model_pca.fit(X_train_pca, y_train_pca)

    y_pred_pca_lr = logistic_regression_model_pca.predict(X_test_pca)

    dis_col.subheader("La précision de notre modèle est :")
    dis_col.write(accuracy_score(y_test_pca, y_pred_pca_lr))

    dis_col.subheader("L'erreur quadratique moyenne de notre modèle est :")
    dis_col.write(mean_squared_error(y_test_pca, y_pred_pca_lr))

    dis_col.subheader("L'erreur absolue moyenne de notre modèle est :")
    dis_col.write(mean_absolute_error(y_test_pca, y_pred_pca_lr))

    dis_col.subheader("Le R score de notre modèle est :")
    dis_col.write(r2_score(y_test_pca, y_pred_pca_lr))

    dis_col.subheader("Le rappel de notre modèle est : ")
    dis_col.write(recall_score(y_test_pca, y_pred_pca_lr))


st.markdown('#')

with conclusion_container:
    st.header('Conclusion')

    """
    Tous les modèles ont une précision similaire, mais ils diffèrent en termes de rappel.
    Tous les modèles ont des scores de précision similaires, mais diffèrent par leur score de rappel. La Régression Logistique a un rappel élevé, mais cela se fait au détriment d'une précision plus faible. Les deux autres modèles ont un rappel de 1.0, ce qui signifie qu'ils identifient bien la classe positive, mais la précision est légèrement inférieure, ce qui signifie que le modèle ne manque aucun cas positif, mais cela se fait au détriment d'une précision plus faible.

    L'ensemble de données pourrait nécessiter un rééquilibrage ou d'autres techniques de gestion des données pour améliorer les performances du modèle, car il semble que le modèle soit biaisé vers la classe majoritaire.
    Le modèle Random Forest a tendance à sur-ajuster les données d'entraînement, ce
    qui se reflète dans une performance moins bonne sur l'ensemble de test.
    Pour de futures analyses, il serait judicieux d'explorer d'autres modèles, de considérer des techniques de rééquilibrage des données pour améliorer la prédiction de la classe négative, et d'évaluer la robustesse des modèles avec un ensemble de données plus large pour obtenir des performances plus fiables.
    """
