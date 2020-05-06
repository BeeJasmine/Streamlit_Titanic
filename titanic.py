import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time
import plotly.express as px

DATE_COLUMN = 'date/time'
DATA_TRAIN = 'train.csv'

@st.cache
def load_data():
    data = pd.read_csv(DATA_TRAIN)
    data_p = data.copy()
    data_p =data_p.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    values = {'Age': data_p['Age'].mean(), 'Cabin': 'NO CABIN', 'Embarked': data_p['Embarked'].mode()[0]}
    data_p = data_p.fillna(value=values)
    data_e = data_p.copy()
    data_e = pd.get_dummies(data_e, columns=['Sex'], prefix = ['Sex'], drop_first=True)
    data_e = pd.get_dummies(data_e, columns=['Embarked'], prefix = ['Embarked'], drop_first=True)
    data_e['Cabin Owner'] = np.where(data_e['Cabin'].str.contains('NO CABIN'), 0, 1)
    data_e = data_e.drop(['Cabin'], axis=1)
    return (data, data_p, data_e)

#data_load_state = st.text('Loading data...')
data = load_data()
#data_load_state.text("Done! (using st.cache)")

def main():
    """ RandomForestClassifier """

    # titre
 
    st.title("distribution des données / Titanic")

    my_placeholder = st.empty()

    my_placeholder.image('titanic.jpeg')
    
    st.sidebar.markdown("## présenté par Jasmine")

    st.sidebar.text('Titanic - 891 observations / 6 attributs')

    st.sidebar.markdown('choisir une option :')

    def open_csv_file(dataset):
        df = pd.read_csv(os.path.join(dataset))
        return df
    
    # ouvrir le dataset
    my_dataset = "train.csv"
    data = open_csv_file(my_dataset)
    data2 = data.copy()
    

    #supprimer les features inutiles/inexploitable
    #penser à transformer SiSp ?

    data.drop(['Cabin','Ticket', 'Parch', 'SibSp', 'PassengerId', 'Name'], axis='columns', inplace=True)
    #remplir l'âge avec la moyenne
    #Nora dit qu'il vaut mieux utiliser la médianne, car moins sensible aux outliers
    data['Age'].fillna((data['Age'].median()), inplace = True)



    #remplir Embarked avecle mode ? 
    #juste parce que je sais le faire
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    labelencoder1 = preprocessing.LabelEncoder()
    data['Embarked'] = labelencoder1.fit_transform(data['Embarked'])


    #il n'y en a que 2 sur les 891 
    #peut-être serait-ce plus judicieux de dropper les deux lignes ?


    #créer une catégorie 'None' me parait une mauvaise idée
    #attributes_with_na = ['Embarked'] # liste des features à remplir
    #for col in attributes_with_na:
    #   df[col].fillna('None',inplace=True

    # #transformer les variables catégoriques avec pandas
    # data = pd.get_dummies(data, columns=['Embarked'], prefix = ['Embarked'], drop_first=True)
    #data = pd.get_dummies(data, columns=['Survived'], prefix = ['Survived'])



    #data = pd.get_dummies(data, columns=['Sex'], prefix = ['Sex'], drop_first=True)
    data['Sex'] = data['Sex'].map({'male':0, 'female':1})


    df = pd.DataFrame(data)
    Y = df["Survived"]
    X = df.drop('Survived', axis=1)
    #X = df[["Age"]]




    if st.sidebar.checkbox("voir le jeu de données complet"):      
        st.write("le voici")
        st.table(data2)

    if st.sidebar.checkbox("voir le jeu de données traité"):
        st.write("le voici")
        st.table(data)

    if st.sidebar.checkbox("distribution des données"):      
        x = st.sidebar.selectbox('Attributs: ', ['Survived', 'Pclass', 'Sex', 'Age', 'Embarked', 'Fare'])
        st.write("Histogramme:  ",x)
        if (x == 'Age'):
            fig1 = px.histogram(data, x=x, hover_data=data.columns, color=x, nbins=20)
            st.write("max: ",data.Age.max(),"moyenne: ",'{0:.2f}'.format(data.Age.mean()), "min: ",data.Age.min() )
        else:
            fig1 = px.histogram(data, x=x, hover_data=data.columns, color=x)
        st.plotly_chart(fig1)
    






    if st.sidebar.checkbox('prediction de survie'):
        #@st.cache
        st.sidebar.markdown("caractéristiques du passager :")

        #classe
        classe = st.sidebar.radio('classe',('1ère classe', '2nde classe', '3e classe'))

        #sexe
        sexe = st.sidebar.radio('sexe',('homme', 'femme'))

        #embarked
        embarquement = st.sidebar.radio('embarquement',('Cherboug', 'Queenstown', 'Southampton'))



        # TO DO: gérer Embarked 
        #embarquement = st.sidebar.radio("embarquement", data['Embarked_Q'].unique())
        age = st.sidebar.slider('âge',min_value=1, max_value=80, value=20, step=1)



        tarif = st.sidebar.slider('tarif',min_value=0, max_value=512, value=100, step=10)



        st.markdown("## voici les caractéristiques sélectionnées")


        #transformer les variables catégoriques avec pandas
        data = pd.get_dummies(data, columns=['Embarked'], prefix = ['Embarked'], drop_first=True)

        data = [{'Pclass': classe, 'Sex': sexe, 'Age': age, 'Fare': tarif, 'Embarked':embarquement}]

        if sexe == "homme":
                st.write(sexe, '---',age,"ans", '--- embarqué en', classe, '--- à', embarquement, '--- pour','$', tarif,) #embarquement
        else:
                st.write(sexe, '---',age,"ans", '--- embarquée en', classe, '--- à', embarquement, '--- pour','$', tarif,) #embarquement

        

        #maintenant que l'utilisateur a lu cette petite synthèse

        #je transforme ses inputs
        #pour mon modèle :
        #je prépare des numériques 


        #sse-cla
        if classe == '1ère classe':
            classe = 1
        elif classe == '2nd class':
            classe = 2
        else:
            classe = 3        

        #sexe
        if sexe == 'homme':
            sexe = 1
        else:
            sexe = 0

        #embarquement
        if embarquement == 'Southampton':
            embarquement = 0
        elif embarquement == 'Cherboug':
            embarquement = 1
        else:
            embarquement = 2


        #une méthode plus pratique consiste à utiliser un dico avec .map :
        #data['Embarked'] = data['Embarked'].map({"S": 0, "C": 1, "Q": 2})
        #comment l'utiliser pour ces .radio ??



        # from sklearn import preprocessing
        # labelencoder1 = preprocessing.LabelEncoder()
        # data['Embarked'] = labelencoder1.fit_transform(data['Embarked'])

        print(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

        model = RandomForestClassifier(n_estimators=100) # plein de params 
        model.fit(X_train, Y_train) # faire le fit
        model.score(X_train, Y_train)
        #personne = np.array([[age]]) #ORDRE du DF respecté ds le commentaire ci-dessous
        #je dois convert classe, sexe, et embarquement avec un dico
        personne = np.array([[classe, sexe, age, tarif, embarquement]]).reshape(1, -1)

        model_pred = model.predict_proba(personne) # stocker la liste de prédictions
        st.write(model_pred)


        bar = st.progress(0)

        if st.button('prédire la survie de l`individu'):
            for i in range(11):
                bar.progress(i * 10)
                # wait
                time.sleep(0.07)

            st.markdown('#### mes prédictions :')
            st.write(model_pred)
            percent = model_pred[0][1] * 100
            st.write("l'individu a")
            st.write(percent,'%')
            st.write('de chance de survie')


            if (model_pred[0][1] < 0.5):
                st.markdown('🐋🐋🐋🐋')
                st.write("repose en paix")                
                
            else:
                st.write("sauvé des eaux ! 🐳")
                st.balloons()
        
        

    

if __name__ == '__main__':
    main()




#maintenant il faut créer le modèle
# from sklearn.ensemble import RandomForestClassifier # pour classif
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# model = RandomForestClassifier(n_estimators=100) # plein de params 
# model.fit(X_train, Y_train) # faire le fit avec rfc
# model.score(X_train, Y_train)
# model_pred = model.predict(X_test) # stocker la liste de prédictions





# import re
# deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
# data = [train_df, test_df]

# for dataset in data:
#     dataset['Cabin'] = dataset['Cabin'].fillna("U0")
#     dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
#     dataset['Deck'] = dataset['Deck'].map(deck)
#     dataset['Deck'] = dataset['Deck'].fillna(0)
#     dataset['Deck'] = dataset['Deck'].astype(int)# we can now drop the cabin feature
# train_df = train_df.drop(['Cabin'], axis=1)
# test_df = test_df.drop(['Cabin'], axis=1)







        # if classe == '1ère classe':
        #     classe = 1
        # elif classe == '2nd class':
        #     classe = 2
        # else:
        #     classe = 3



#selected_data = np.array([class_, sex_, age_, not_alone_, cabin_]).reshape(1, -1)