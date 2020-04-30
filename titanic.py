import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time
import plotly.express as px



def main():
    """ RandomForestClassifier """
    
    # titre
 
    st.title("distribution des données / Titanic")

    my_placeholder = st.empty()

    my_placeholder.image('titanic.jpeg')
    
    st.sidebar.markdown("## présenté par Jasmine")

    st.sidebar.text('Titanic - 891 observations / 6 attributs')

    st.sidebar.markdown('choisir une option:')

    def open_csv_file(dataset):
        df = pd.read_csv(os.path.join(dataset))
        return df
    
    # ouvrir le dataset
    my_dataset = "train.csv"
    data = open_csv_file(my_dataset)
    data2 = data
    

    #supprimer les features inutiles

    data2.drop(['Cabin','Ticket', 'Parch', 'SibSp', 'Fare', 'PassengerId'], axis='columns', inplace=True)
    #remplir l'âge avec la moyenne

    #remplir Embarked avecle mode ? 




    #if st.sidebar.checkbox("voir le jeu de données complet"):      
     #   st.write("le voici")
      #  st.table(data)

    if st.sidebar.checkbox("voir le jeu de données traité"):
        st.write("le voici")
        st.table(data2)

    if st.sidebar.checkbox("distribution des données"):      
        x = st.sidebar.selectbox('Attributs: ', ['Survived', 'Pclass', 'Sex', 'Age', 'Embarked'])
        st.write("Histogramme:  ",x)
        if (x == 'Age'):
            fig1 = px.histogram(data, x=x, hover_data=data.columns, color=x, nbins=20)
            st.write("max: ",data.Age.max(),"moyenne: ",'{0:.2f}'.format(data.Age.mean()), "min: ",data.Idade.min() )
        else:
            fig1 = px.histogram(data, x=x, hover_data=data.columns, color=x)
        st.plotly_chart(fig1)
        

    

if __name__ == '__main__':
    main()