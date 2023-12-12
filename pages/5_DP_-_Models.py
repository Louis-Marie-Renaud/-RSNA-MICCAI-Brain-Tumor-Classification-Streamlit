import streamlit as st
import pandas as pd
import numpy as np
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical

st.set_page_config(
    page_title='Prédictions modèle Deep Learning',
    layout='wide'
)

header_container = st.container()
cnn_intro_container = st.container()
data_preprocesing_container = st.container()
conclusion_container = st.container()

feature_to_keep = ["MGMT_value"]

feature_outliers = ['glrlm_LongRunHighGrayLevelEmphasis','MajorAxisLength','gzlm_SizeZoneNonUniformityNormalized','Kurtosis','Idn',
'Idmn','gldm_LargeDependenceHighGrayLevelEmphasis','Mean','MinorAxisLenth','glrlm_RunPercentage']
feature_non_outliers = ['InverseVariance','Maximum3DDiameter']

target = pd.read_csv('./datasets/target.csv')
features = pd.read_csv('./datasets/features.csv')

with header_container:

	# for example a logo or a image that looks like a website header
    col1, col2, col3 = st.columns([2,4,2])

    st.title("Deep Learning Models")

    with col1:
        st.write("")

    with col2:
        st.image('./images/brain_tumor.png')

    with col3:
        st.write("")


with cnn_intro_container:
    st.header("Utilisation de réseaux de neurones convolutifs")

    """
    Dans le contexte de notre projet initial, nous avons adopté l'approche d'Uni-net en utilisant des IRM pour détecter la présence de la MGMT.
    Cependant, en raison de résultats non concluants lors de cette première tentative, nous avons décidé d'explorer d'autres solutions pour améliorer nos performances. Cette démarche nous a conduits à l'utilisation de réseaux de neurones convolutionnels (CNN).
    """
    training_col1, training_col2 = st.columns([5,3])

    with training_col1:
        st.subheader("Entraînement de notre modèle")
        """
        ● Taille du Lot (Batch Size) :
        L'entraînement se fait par lots de 32 images (16 - Centre - 16) avec 4 modélatées, par batch de 8 une limitation de Kaggle.

        ● Époques :
        Le modèle est entraîné pendant 32 époques.

        ● Optimiseur :
        Type : Descente de Gradient Stochastique (SGD);
        Explication : La Descente de Gradient Stochastique (SGD) ajuste de manière itérative les poids du modèle en utilisant les gradients calculés à partir des données d'entraînement pour minimiser la perte.

        ● Configuration :
        Taux d'Apprentissage : Un taux d'apprentissage de 0,001 permet d'atteindre un équilibre entre la vitesse de convergence et la stabilité.

        ● Optimisation l’apprentissage :
        Nous avons utiliser un Learning Rate Scheduler qui augmente progressivement la cadence d’apprentissage pour éviter les risques de divergence en début d’apprentissage.
        Ainsi qu'une successions d'échauffements et diminutions du taux d'apprentissage pour la phase de recherche et d'exploration dans la descente du gradient.

        Ansi que des callback Callbacks bestmodel, EarlyStopping
        """

    with training_col2:
        st.image("./images/training_dl_2.png")


    st.markdown('#')

    architecture_col1, architecture_col2 = st.columns([7,5])

    with architecture_col1:
        st.image("./images/architecture.png")


    with architecture_col2:
        st.subheader("Architecture du modèle : DeepScanModel")
        """
        ● Le modèle est un Réseau de Neurones Convolutionnels (CNN) 3D conçu spécifiquement pour traiter des séquences d'images médicales.

        ● Il prend quatre canaux correspondant à quatre séquences d'images, les concatène et les traite pour effectuer une classification binaire.
        """
        st.image("./images/7_cnn.gif", use_column_width=True, caption='3D CNN')


with cnn_intro_container:
    st.header("Fonctionnement du CNN en quelques étapes")

    """
    Afin de mieux comprendre le fonctionnement du CNN, examinons le processus en quelques étapes clés.
    """

    st.markdown('#')

    st.image("./images/1_cnn.gif", use_column_width=True, caption='Reseau CNN')


with cnn_intro_container:
    st.header("Convolution")

    """
    La convolution est le premier processus dans lequel des filtres sont appliqués à l'image d'entrée pour détecter des caractéristiques importantes.
    """

    st.markdown('#')

    st.image("./images/2_cnn.gif", use_column_width=False, caption='Convolution')




with cnn_intro_container:
    st.header("Intermediate Layer")

    """
    Cette couche intermédiaire montre la transition du pooling à la convolution. Après le pooling, une convolution est appliquée pour extraire des caractéristiques plus abstraites à partir des informations réduites.
    """

    st.markdown('#')

    st.image("./images/5_cnn.gif", use_column_width=True, caption='Intermediate Layer')


with cnn_intro_container:
    st.header("ReLU")

    """
    Après la convolution, la fonction ReLU est appliquée pour introduire une non-linéarité, aidant le réseau à apprendre des motifs plus complexes.
    """

    st.markdown('#')

    st.image("./images/3_cnn.gif", use_column_width=True, caption='ReLU')


with cnn_intro_container:
    st.header("Pooling (Pooling spatial)")

    """
    Le pooling est utilisé pour réduire la dimension spatiale de la représentation, tout en préservant les caractéristiques importantes.
    """

    st.markdown('#')

    st.image("./images/4_cnn.gif", use_column_width=True, caption='Pooling')

with cnn_intro_container:
    st.header("Dropout")

    """
    La couche de Dropout est ajoutée pour régulariser le modèle en désactivant aléatoirement certains neurones pendant l'entraînement, empêchant ainsi le surapprentissage.
    """



with cnn_intro_container:
    st.header("Fully Connected")

    """
    Enfin, les couches entièrement connectées sont utilisées pour combiner les caractéristiques extraites et effectuer la classification.
    """

    st.markdown('#')

    st.image("./images/6_cnn.gif", use_column_width=True, caption='Pooling')







st.markdown('##')

with conclusion_container:
    st.header('Conclusion')

    """
    Notre modèle CNN (Convolutional Neural Network) a été le fruit de recherches approfondies et d'un engagement soutenu pour améliorer la prédiction du sous-type génétique du  glioblastome basé sur la méthylation du promoteur MGMT.
    Bien que nous ayons rencontré des défis significatifs tout au long de ce projet, notre modèle a représenté une avancée importante dans la recherche de méthodes non invasives de diagnostic pour cette forme agressive de cancer du cerveau.
    Notre modèle, le DeepScanModel, est un CNN 3D conçu pour traiter des séquences d'images médicales. Il prend en compte les spécificités des données d'IRM en traitant quatre canaux correspondant à quatre séquences d'images.

    Malheureusement, malgré ces efforts, la prédiction sur l'ensemble de test n'a atteint qu'une précision de 74.8 %, ce qui indique une marge d'amélioration significative.
    """
