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

target = pd.read_csv('target.csv')
features = pd.read_csv('features.csv')

with header_container:

	# for example a logo or a image that looks like a website header
    col1, col2, col3 = st.columns([2,4,2])

    st.title("Streamlit Projet Brain Tumor")
    st.header("Bienvenue dans notre Streamlit de présentation de notre projet de datascience !")
    st.write("Le menu à gauche de l'écran vous permettra de naviguer sur les différentes parties de notre projet")

    with col1:
        st.write("")

    with col2:
        st.image('images/brain_tumor.png')

    with col3:
        st.write("")


with cnn_intro_container:
    st.header("Utilisation de réseaux de neurones convolutifs")

    """
    Dans le contexte de notre projet initial, nous avons adopté l'approche d'Uni-net en utilisant des IRM pour détecter la présence de la MGMT.
    Cependant, en raison de résultats non concluants lors de cette première tentative, nous avons décidé d'explorer d'autres solutions pour améliorer nos performances. Cette démarche nous a conduits à l'utilisation de réseaux de neurones convolutionnels (CNN).
    """

    st.markdown('##')


with data_preprocesing_container:
    st.header("Préparation des données")

    dl_crossed_validation1, dl_crossed_validation2 = st.columns([5,3])

    with dl_crossed_validation1:
        st.subheader("Validation croisée stratifiée en K-Fold")
        """
        La validation croisée est employée pour garantir une répartition équilibrée des classes dans chaque fold, améliorant ainsi l'évaluation et la généralisation du modèle.

        Configuration :
        
        ● Nombre de Folds : Généralement fixé à 5
        
        ● Fold de Validation : Dans ce cas, nous avons désigné le premier fold comme fold de validation.
        """

        st.subheader("Traitement des Données")
        """
        ● Largeur, Hauteur, Canaux :
        Les images sont initialement redimensionnées à une dimension uniforme de 128x128 pixels. De plus, les IRM converties en images sont au format niveau de gris, ce qui donne un seul canal pour chaque image.
        Il est important de noter que, en raison des limitations de Kaggle, nous ne pouvons pas augmenter davantage la taille des images à 224x224 pixels.

        ● Séquence :
        Pour chaque entrée, nous utilisons une séquence de 32 images séquentielles, qui sont traitées par lots.
        Cette séquence capture la dimension temporelle des scans IRM et permet au modèle d'analyser une série d'images pour effectuer des prédictions.
        
        ● Mise à l'Échelle :
        Lors de la préparation des données, les images sont réduites à 85 % de leur taille d'origine.
        Cette mise à l'échelle est appliquée à la fois aux données de test et de validation.
        Cette mise à l'échelle a pour but de supprimer tout espace vide autour de la bordure du cerveau, garantissant que le modèle se concentre principalement sur la région du cerveau elle-même.
        
        ● Mise au Point Central :
        Dans notre pipeline de traitement des données, nous mettons fortement l'accent sur les régions centrales, où se trouve la région d'intérêt (ROI).
        Plus précisément, nous donnons la priorité à l'image centrale et incluons 16 images avant et 16 images après l'image centrale dans chaque séquence.
        Cette approche garantit que les parties les plus informatives des scans IRM, correspondant à la zone centrale du cerveau et à la ROI, reçoivent le plus d'attention lors de l'entraînement du modèle.
        """

    with dl_crossed_validation2:
        st.image("images\crossed_validation.png")

    st.markdown('#')

    augmentation_col1, augmentation_col2 = st.columns([3,5])

    with augmentation_col1:
        st.image("augmentation_dl_nb.png")
        st.image("augmentation_dl_col.png")
    
    with augmentation_col2:
        st.subheader("Augmentation")
        """
        ● Augmentation des Données :
        Nous augmentons notre ensemble de données initial de manière significative, de 400 %, ce qui équivaut à quadrupler la quantité de données d'entraînement disponibles. Ceci est essentiel car l'ensemble de données initial ne compte qu'environ 500 échantillons, moins l’ensemble de validation contenant seulement 100 échantillons.

        ● Augmentation par Recadrage (Crop) :
        Les images sont recadrées de manière aléatoire tout en préservant entre 85 % et 95 % de leur taille d'origine.
        Cela améliore la capacité du modèle à reconnaître différentes régions cérébrales.
        
        ● Augmentation par Rotation :
        Des rotations aléatoires de 4 à 12 degrés aident le modèle à devenir invariant à l'orientation, lui permettant de gérer efficacement les variations de l'orientation des scans cérébraux.
        
        ● Augmentation par Translation :
        Des translations aléatoires sont appliquées à la fois horizontalement (2 à 6 pixels) et verticalement (0 à 2 pixels), simulant de légères variations de position couramment rencontrées en imagerie médicale.
        
        ● Augmentation par Flou (Blur) :
        Un effet de flou aléatoire est introduit avec une probabilité de 10 % à 15 %, imitant les imperfections de l'imagerie du monde réel et améliorant la généralisation du modèle.
        
        ● Augmentation de la Contraste et de la Luminosité :
        Le contraste de l'image est dynamiquement ajusté entre 0,8 et 1,2, tandis que la luminosité est réglée entre -2 et 2.
        Cette adaptation tient compte des différentes conditions d'éclairage dans les données d'entrée, rendant le modèle plus robuste face à différents scénarios d'éclairage.
        """

    st.markdown('#')

    training_col1, training_col2 = st.columns([5,3])
    
    with training_col1:
        st.subheader("Entraînement de notre modèle")
        """
        ● Taille du Lot (Batch Size) :
        L'entraînement se fait par lots de 8 images à la fois, une limitation imposée par Kaggle.
        
        ● Époques :
        Le modèle est entraîné pendant 32 époques.
        
        ● Optimiseur :
        Type : Descente de Gradient Stochastique (SGD);
        Explication : La Descente de Gradient Stochastique (SGD) ajuste de manière itérative les poids du modèle en utilisant les gradients calculés à partir des données d'entraînement pour minimiser la perte.
        
        ● Configuration :
        Taux d'Apprentissage : Un taux d'apprentissage de 0,001 permet d'atteindre un équilibre entre la vitesse de convergence et la stabilité.

        """

    with training_col2:
        st.image("images/training_dl_2.png")


    st.markdown('#')

    architecture_col1, architecture_col2 = st.columns([3,5])
    
    with architecture_col1:
        st.image("images/architecture.png")

    with architecture_col2:
        st.subheader("Architecture du modèle : DeepScanModel")
        """
        ● Le modèle est un Réseau de Neurones Convolutionnels (CNN) 3D conçu spécifiquement pour traiter des séquences d'images médicales.
        
        ● Il prend quatre canaux correspondant à quatre séquences d'images, les concatène et les traite pour effectuer une classification binaire.
        """
    

st.markdown('##')

with conclusion_container:
    st.header('Conclusion')

    """
    Notre modèle CNN (Convolutional Neural Network) a été le fruit de recherches approfondies et d'un engagement soutenu pour améliorer la prédiction du sous-type génétique du  glioblastome basé sur la méthylation du promoteur MGMT.
    Bien que nous ayons rencontré des défis significatifs tout au long de ce projet, notre modèle a représenté une avancée importante dans la recherche de méthodes non invasives de diagnostic pour cette forme agressive de cancer du cerveau.
    Notre modèle, le DeepScanModel, est un CNN 3D conçu pour traiter des séquences d'images médicales. Il prend en compte les spécificités des données d'IRM en traitant quatre canaux correspondant à quatre séquences d'images.
    
    Malheureusement, malgré ces efforts, la prédiction sur l'ensemble de test n'a atteint qu'une précision de 74.8 %, ce qui indique une marge d'amélioration significative.
    """