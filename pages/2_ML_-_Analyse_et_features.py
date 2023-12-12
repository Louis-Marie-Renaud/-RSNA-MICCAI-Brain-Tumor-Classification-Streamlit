import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pygwalker as pyg
import streamlit.components.v1 as components

## 1. Multipages

st.set_page_config(
    page_title='Visualisation des données',
    layout='wide'
)

header_container = st.container()
presentation_container = st.container()
correlation_container = st.container()
variable_selection_container = st.container()


data = pd.read_csv('datasets/dataset_df.csv')
corr_df = data.corr()


with header_container:

	# for example a logo or a image that looks like a website header
    col1, col2, col3 = st.columns([2,4,2])

    st.title("Machine Learning analyse et features")

    with col1:
        st.write("")

    with col2:
        st.image('images/brain_tumor.png')

    with col3:
        st.write("")

st.markdown('#')

with presentation_container:

    st.header('Présentation du jeu de données')

    """
    Notre ensemble de données d'entraînement se compose de 585 patients, ce qui est relativement limité pour l'apprentissage automatique. Un ensemble de données plus important fournirait des résultats plus robustes.
    """
    st.markdown('##')

    pytorch_presentation1, pytorch_presentation2 = st.columns([4, 4])

    with pytorch_presentation1:
        st.subheader('Utilisation de la bibliothèque U-Net for brain MRI')
        """
        Afin d'exploiter les capacités 3D pour la segmentation cérébrale, nous utiliserons le modèle U-Net pour l'IRM cérébrale et la classe RadiomicsShape3D.

        Le modèle U-Net pour l'IRM cérébrale en 3D étend l'architecture 2D pour traiter des images volumétriques du cerveau. Il suit une architecture en forme de U similaire avec des connexions de saut, mais les couches de convolution opèrent de manière tridimensionnelle. Le modèle se compose de chemins d'encodeur et de décodeur, chaque chemin contenant plusieurs niveaux de blocs. Le nombre de filtres dans les couches de convolution varie selon les niveaux, permettant au modèle de capturer différents niveaux de détails dans les données volumétriques.

        La segmentation cérébrale est une tâche cruciale en imagerie médicale car elle permet l'extraction d'informations précises sur différentes régions ou classes au sein des structures cérébrales. Une segmentation précise joue un rôle vital dans diverses applications médicales, notamment la détection de tumeurs, l'analyse anatomique et la planification du traitement.

        Nous avons utilisé, afin de générer un dataset à partir de nos fichiers au format DCM, une bibliothèque Python. La bibliothèque "mateuszbuda_brain-segmentation-pytorch_unet" a été choisie pour faciliter la segmentation cérébrale à partir d'images médicales. Il s'agit notamment de données géométriques sur la forme de la tumeur, par exemple la variable MaximumDiameter, ou des données de texture de la tumeur, par exemple avec Contrast.
        Nous pouvons observer les premières lignes de notre dataframe généré. Celui-ci contient un grand nombre de variables.

        En incorporant le modèle U-Net 3D pour l'IRM cérébrale et la classe RadiomicsShape3D, nous avons étendu les capacités de notre projet à l'analyse d'images volumétriques du cerveau. Cela permet une segmentation plus complète et une analyse de forme, facilitant larecherche médicale avancée et les applications cliniques.
        Initialement, nous avons travaillé avec des données 2D, où nous avons uniquement pris en compte la coupe centrale des images. Cependant, compte tenu de la taille de notre ensemble de données et des limites de la visualisation 2D pour fournir une compréhension complète de la structure cérébrale, il semblait approprié de passer à une approche 3D. En travaillant avec des volumes d'images 3D, nous avons pu enrichir nos données et obtenir une représentation plus complète des caractéristiques du cerveau. Cette transition vers le 3D offre de nouvelles perspectives pour une analyse plus détaillée et des résultats plus précis.

        Ainsi, en exploitant le modèle U-Net 3D et la classe RadiomicsShape3D, nous sommes en mesure de tirer pleinement parti des avantages offerts par l'analyse d'images volumétriques.
        """

    with pytorch_presentation2:
        st.image("images/unet.png")
        st.markdown('#')
        st.image("images/segmentation.png")

    st.markdown('##')


    @st.cache_data
    def get_station_rides(mgmt_value):
        subset = data[data['MGMT_value']==mgmt_value]

        return subset

    selected_station = st.selectbox(label="Visualiser le jeu de données pour une variable cible donnée :",options=set(data["MGMT_value"].tolist()))

    # use the function as you need it
    subset = get_station_rides(selected_station)
    st.dataframe(subset)

st.markdown('##')

with correlation_container:
    st.header("Analyse univariée et bivariée")

    st.subheader('Manipulation du jeu de données avec PygWalker')
    """
    Nous pouvons visualiser dans cette partie la répartition de nos deux classes d'individus pour chacune de nos variables explicatives.
    Grâce à la bibliothèque PygWalker, nous pouvons sélectionner les variables dont nous voulons étudier les effectifs selon la variable cible. Nous pouvons également sélectionner un ensemble de variables pour en étudier les corrélations.
    Nous pouvons également sélectionner un ensemble de variables pour en étudier les corrélations.
    """

    # Display PyGWalker
    df = pd.read_csv("datasets/2d_rsna_miccai_brain_tumor_brain_segmentation_pytorch_unet.csv")
    #pyg_html = pyg.walk(df, return_html=True, spec="config.json")

    # Embed the HTML into the Streamlit app
    #components.html(pyg_html, height=1000, scrolling=True)

    st.markdown('#')

    st.subheader('Conclusion')
    """
    L'analyse univariée et bivariée de nos données ne nous a pas permis d'identifier de manière évidente une corrélation entre nos variables et la variable cible. Nous devons donc sélectionner les variables les plus significatives.

    Au vu de notre quantité limité de données, nous allons avoir recours à des méthodes d'augmentation de nos données pour l'apprentissage de nos modèles de Machine Learning et de Deep Learning.
    """












st.markdown('###')

with variable_selection_container:
    st.header('Sélection de variables')
    """
    La bibliothèque que nous utilisons pour extraire nos données à partir des IRM nous fourni un très grand nombre de variables. Afin d'éviter le surentraînement de nos algorithmes, nous devions éliminer certaines variables.
    On peut noter que certaines variables forment des groupes où toutes sont fortement corrélées entre elles, comme nous pouvons le voir dans cette matrice de corrélation. Nous pouvons en déduire que ces variables sont redondantes et choisir un seuil de corrélation pour en éliminer une partie.
    Nous avons dans un premier temps utilisé une méthode de réduction de dimension de dimension : RFE (Recursive Feature Elimination).
    """

    st.subheader('Distribution et valeurs aberrantes')
    """
    La plupart des variables ont des moyennes nettement supérieures à la médiane, suggérant une asymétrie positive. Des variables telles que "LeastAxisLength", "Flatness", "TotalEnergy", "Maximum3DDiameter", "Energy", "Variance", etc. Les variables présentent une asymétrie positive avec des valeurs élevées. Cela peut compliquer l'analyse des données.
    """

    st.subheader('Valeurs nulles')
    """
    Des variables telles que "LeastAxisLength", "Flatness", "gldm_GrayLevelNonUniformityNormalized", "gldm_DependencePercentage" montrent une valeur minimale de 0 ou NaN, ce qui pourrait indiquer des données manquantes ou incorrectes.
    Nous supprimons désormais ces variables pour le reste du projet.
    """


    st.subheader('Redondance potentielle')
    """
    Un certain nombre de variables de notre jeu de données peuvent être redondantes. Afin de trouver celles-ci, nous avons regardé quelles variables étaient fortement corrélées entre elles.
    Ici nous pouvons voir la matrice de corrélation, où l'on peut voir uniquement les corrélations supérieures en valeur absolue à un seuil donné.
    """


    @st.cache_data
    def filter_correlation_matrix(correlation_matrix, correlation_threshold):
        """
        Filters a correlation matrix by keeping only the absolute values greater than or equal to the correlation threshold.

        Args:
            correlation_matrix (pd.DataFrame): The correlation matrix.
            correlation_threshold (float): The correlation threshold for filtering the matrix.

        Returns:
            pd.DataFrame: The filtered correlation matrix.

        """
        filtered_correlation_matrix = correlation_matrix.copy()
        filtered_correlation_matrix[abs(filtered_correlation_matrix) < correlation_threshold] = np.nan

        return filtered_correlation_matrix

    @st.cache_data
    def heatmap(df):
        fig = plt.figure(figsize=(50, 20))
        mask = np.triu(np.ones_like(df))
        sns.heatmap(df.corr(), annot=False, mask=mask, cmap="coolwarm")
        st.pyplot(fig)

    correlation_threshold = st.slider("Sélectionner un seuil de corrélation", min_value=0.5, max_value=0.95, step=0.05)
    # use the function as you need it
    updated_corr_df = filter_correlation_matrix(corr_df, correlation_threshold)
    heatmap(updated_corr_df)

    st.markdown('#')

    st.subheader('Conclusion')
    """
    Nous avons supprimé certaines variables dont les valeurs semblaient aberrantes. Notre analyse a également montré qu'il existait des variables fortement corrélées entre elles. Nous avons déterminé qu'à partir d'un certain seuil, ces variables pouvaient être considérées comme redondantes, et nous n'en avons gardé qu'une seule par groupe. Enfin, certaines variables, notamment liées à la taille du cerveau de fournissaient aucune information sur une possible corrélation avec notre variable cible.
    Nous avons donc pu réaliser une première étape du prétraitement des données, et supprimant un certain nombre de variables inutiles ou redondantes.
    """
