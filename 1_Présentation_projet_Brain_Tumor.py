''' 
This is another streamlit app that uses some more advanced functionality such as:
1. multipages
2. session states
3. callback functions
4. caching
5. using custom components (made by community)

'''

import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## 1. Multipages

st.set_page_config(
    page_title='Visualisation des données',
    layout='wide'
)

header_container = st.container()
intro_container = st.container()
tumor_presentation_container = st.container()
irm_container = st.container()

def int_part_file_name(file_name):
    m = re.search(r'\d+\.png', file_name)
    file_number = m.group(0)
    file_number = re.search(r'\d+', file_number)
    file_number = int(file_number.group(0))

    print(file_number)
    return file_number


def file_list_in_directory(path):
    filelist=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            filename=os.path.join(root, file)
            filelist.append(filename)

    filelist.sort(key=int_part_file_name)

    return(filelist)

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
	


with intro_container:

    st.header('Présentation de notre projet Brain Tumor')
    st.subheader('Glioblastome et méthylation du promoteur MGMT')
    """
    Le glioblastome est le type de cancer cérébral le plus agressif et le plus commun chez l'adulte. Il pose aujourd'hui de grande difficultés pour son diagnostic et son traitement. La méthylation du promoteur MGMT a été identifiée comme signe de réponse à la chimiothérapie dans le cadre du traitement pour le glioblastome.
    Sa détection requiert cependant des opérations et procédures lourdes pour le patient.
    Ce projet a pour objectif d'aider à détecter la méthylation du promoteur MGMT à partir d'IRM. Nous avions à notre disposition une base de données avec, pour un ensemble de patients, des fichiers DCM correspondant aux IRM mettant en évidence les tumeurs cérébrales, ainsi qu'un fichier CSV nous indiquant si pour chacun des patients, il y avait ou non méthylation du promoteur MGMT. Les fichiers DCM sont des fichiers standards dans le domaine de l'imagerie médicale. 
    Nous avons dans un premier temps voulu comprendre ce qu'était le glioblastome et méthylation du promoteur MGMT, comment cela pouvait se traduire sur des IRM, et comment passer d'un ensemble de fichiers d'imagerie médicale à un dataset que nous pouvions utiliser pour entraîner nos modèles de prédiction.
    """

    st.markdown('##')

with tumor_presentation_container:
    st.header('Les tumeurs cérébrales, glioblastome et méthylation du promoteur MGMT')
    st.markdown('#')

    col_tumor_presentation1, col_tumor_presentation2 = st.columns([5,3])

    with col_tumor_presentation1:
        st.subheader('Introduction aux tumeurs cérébrales')
        """
        Les tumeurs cérébrales sont un problème de santé majeur affectant des personnes de tous les groupes d'âge, et leur incidence semble être en augmentation. Chez les nourrissons, les enfants, les adolescents, les adultes jeunes et âgés, les tumeurs cérébrales sont une réalité qui touche un grand nombre de vies.
        Cette introduction met en évidence l'ampleur du problème et souligne la nécessité de recherches et d'outils avancés pour leur détection et leur traitement.

        Une tumeur est une masse anormale de cellules résultant d'une prolifération anormale. Dans le corps humain, il existe une grande variété de cellules, chacune ayant des fonctions spécifiques. Parfois, certaines cellules subissent des mutations qui les amènent à se multiplier de manière incontrôlée, formant ainsi une tumeur.
        """
    
    with col_tumor_presentation2:
        st.image("images/brain_tumor_plan.png")

    st.markdown('#')

    col_tumor_presentation3, col_tumor_presentation4 = st.columns([3,5])

    with col_tumor_presentation3:
        st.image("images/mgmt_plan.png")
    
    with col_tumor_presentation4:
        st.subheader('Méthylation du promoteur MGMT')
        """
        La Méthylguanine-DNA Méthyltransférase, abrégée MGMT, est une enzyme cruciale dans le mécanisme de réparation de l'ADN au sein des cellules. Son rôle est de protéger l'intégrité du matériel génétique en éliminant les groupes méthyles de l'ADN. Les groupes méthyles, lorsqu'ils s'attachent à des sites spécifiques de l'ADN, peuvent causer des mutations et des dommages à l'ADN, ce qui peut éventuellement conduire à la formation de tumeurs.
        
        En utilisant des données IRM, il est possible de détecter des caractéristiques spécifiques dans les tumeurs cérébrales qui peuvent être associées au statut de la MGMT, ce qui peut orienter les décisions cliniques.
        La présence ou l'absence de MGMT dans une tumeur cérébrale a des implications directes sur les choix de traitement. Les tumeurs cérébrales qui expriment activement la MGMT ont tendance à être résistantes à la témozolomide, ce qui rend le traitement moins efficace. En revanche, les tumeurs cérébrales qui présentent une faible expression ou une absence de MGMT sont plus sensibles à la témozolomide, ce qui peut améliorer les perspectives de traitement et de survie pour les patients.
        """

    st.markdown('##')

with irm_container:
    st.subheader('IRM à disposition pour le projet')
    """
    L'IRM, ou Imagerie par Résonance Magnétique, est réalisée à l'aide d'un appareil en forme de cylindre équipé d'un aimant très puissant. Cet aimant génère des ondes radio projetées sur le cerveau pour créer des images en coupe. Ces images sont ensuite assemblées par un ordinateur pour obtenir une représentation précise du cerveau. Pendant l'examen, un produit de contraste est souvent injecté dans une veine du bras pour mettre en évidence certains aspects du cerveau, tels que les vaisseaux sanguins, ce qui facilite l'interprétation des images. 
    
    Nous avions à notre disposition un ensemble d'IRM  pour chaque patient. Ces IRM représente l'ensemble du cerveau du patient par coupes en 2D. Nous avions des IRM correspondant à quatre modalités : FLAIR, T1w, T2 et T1wCE. Ces modalités sont utilisées en imagerie médicale pour mettre en évidence une zone spécifique d'une tumeur par exemple. 
    """
    st.markdown('#')

    col_tumor_viz1, col_tumor_viz2, col_tumor_viz3, col_tumor_viz4 = st.columns([2,2,2,2])

    flair_file_list = file_list_in_directory("FLAIR")
    flair_list_size = len(flair_file_list)

    t1w_file_list = file_list_in_directory("T1w")
    t1w_list_size = len(t1w_file_list)

    t2w_file_list = file_list_in_directory("T2w")
    t2w_list_size = len(t2w_file_list)

    t1wce_file_list = file_list_in_directory("T1wCE")
    t1wce_list_size = len(t1wce_file_list)

    # use st.slider to select
    index = st.slider('Faites défiler les IRM pour les quatre modalités', 1, 100, key="slider")

    with col_tumor_viz1:
        st.subheader('FLAIR')

        index_flair = index*(flair_list_size/100)
        index_flair = int(index_flair)
        
        print(flair_file_list[index_flair-1])
        st.image(flair_file_list[index_flair-1])

    # use st.pyplot
    with col_tumor_viz2:
        st.subheader('T1w')

        index_t1w = index*(t1w_list_size/100)
        index_t1w = int(index_t1w)

        print(t1w_file_list[index_t1w-1])
        st.image(t1w_file_list[index_t1w-1])

    with col_tumor_viz3:
        st.subheader('T2w')

        index_t2w = index*(t2w_list_size/100)
        index_t2w = int(index_t2w)

        print(t2w_file_list[index_t2w-1])
        st.image(t2w_file_list[index_t2w-1])

    # use st.pyplot
    with col_tumor_viz4:
        st.subheader('T1wCE')

        index_t1wce = index*(t1wce_list_size/100)
        index_t1wce = int(index_t1wce)

        print(t1wce_file_list[index_t1wce-1])
        st.image(t1wce_file_list[index_t1wce-1])


