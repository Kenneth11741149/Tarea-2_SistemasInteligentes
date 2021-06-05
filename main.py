import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix



from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from sklearn.neighbors import KNeighborsClassifier

import time

from sklearn.tree import DecisionTreeClassifier


import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree





def k_means_clustering():
    datasetlist = ["datos_1.csv","datos_2.csv","datos_3.csv"]
    names = []
    i = 0
    for element in datasetlist:
        names.append(datasetlist[i].replace(".csv",""))
        i = i + 1
    j = 0
    for dataset in datasetlist:
        dataset = pd.read_csv(dataset)
        for i in range(1,6):  # 1 through 5 clusters
            k_means(i,dataset,names[j])
        j = j + 1


def k_means(number_of_clusers, dataset,datasetname):
    #Kmeans
    kmeans = KMeans(n_clusters=number_of_clusers)
    labels = kmeans.fit_predict(dataset)

    #Get Centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(dataset['x'], dataset['y'], c=kmeans.labels_.astype(float), s=10, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')

    #plt.savefig(r"C:\Users\Burni\Desktop\Tarea-2_SistemasInteligentes\kmeans"+"\k_means_"+str(datasetname)+"_clusters"+str(number_of_clusers))
    plt.savefig("Kmeans\\"+str(datasetname)+"_clusters"+str(number_of_clusers))
    plt.cla()
    plt.clf()

def agglomerative_clustering():
    datasetlist = ["datos_1.csv", "datos_2.csv", "datos_3.csv"]
    names = []
    i = 0
    for element in datasetlist:
        names.append(datasetlist[i].replace(".csv", ""))
        i = i + 1
    j = 0
    for dataset in datasetlist:
        dataset = pd.read_csv(dataset)
        for i in range(1, 6):  # 1 through 5 clusters
            agglomerative(i, None, dataset, names[j])
        distances = [0.25,0.50,0.75,1.0,1.5]
        for distance in distances:
            agglomerative(None,distance,dataset,names[j])
        j = j + 1


def agglomerative(number_of_clusters,distance,dataset,datasetname):
    save_location = ""
    if distance is None: #With clusters
        # Agglomerative Clustering "Ward"
        cluster = AgglomerativeClustering(n_clusters=number_of_clusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(dataset)
        save_location = "Agglomerative Clustering\\"+str(datasetname)+"_clusters"+str(number_of_clusters)
    else: #With distances
        # Agglomerative Clustering "Ward:
        cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=distance,affinity='euclidean',linkage="ward")
        cluster.fit_predict(dataset)
        distance = str(distance).replace(".","_")
        save_location = "Agglomerative Clustering\\" + str(datasetname) + "_distance_"+str(distance)
    plt.scatter(dataset['x'], dataset['y'], c=cluster.labels_, cmap='rainbow')
    plt.savefig(save_location)
    plt.cla()
    plt.clf()

def DBScan():
    datasetlist = ["datos_1.csv", "datos_2.csv", "datos_3.csv"]
    names = []
    i = 0
    for element in datasetlist:
        names.append(datasetlist[i].replace(".csv", ""))
        i = i + 1
    i = 0
    for dataset in datasetlist:
        dataset = pd.read_csv(dataset)
        neighbors_distance = [0.25,0.35,0.5]
        min_samples = [5, 10, 15]
        for distance in neighbors_distance:  # 1 through 5 clusters
            for sample in min_samples:
                DB(distance,sample,dataset,names[i])
                #agglomerative(None, distance, dataset, names[j])
        i = i + 1

def DB(distance_between_neighbors, min_samples_neighborhood, dataset, datasetname):
    #Db Scan
    label = DBSCAN(eps=distance_between_neighbors, min_samples=min_samples_neighborhood).fit(dataset)
    #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #core_samples_mask[db.core_sample_indices_] = True
    labels = label.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)



    #no_clusters = len(dataset.unique(labels))
    #no_noise = np.sum(dataset.array(labels) == -1, axis=0)
    plt.scatter(dataset['x'], dataset['y'],  c=label.labels_)

    distance_between_neighbors = str(distance_between_neighbors).replace(".","_")
    save_location = "DBScan\\"+str(datasetname)+"_eps_"+str(distance_between_neighbors)+"_min_s_"+str(min_samples_neighborhood)
    plt.savefig(save_location)
    plt.cla()
    plt.clf()




def knn():
    kays = [1,3,5,7,9,11,13,15]
    for k in kays:
        Ejercicio2(k)

def Ejercicio2(k):
    dataset = pd.read_csv("genero_peliculas_training.csv")

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, 10]
    labelEncoder_X = LabelEncoder()
    X = X.apply(LabelEncoder().fit_transform)
    Knn = KNeighborsClassifier(n_neighbors=k).fit(X,y)

    testing_dataset = pd.read_csv("genero_peliculas_testing.csv")
    X2 = testing_dataset.iloc[:, :-1]
    X2 = X2.apply(LabelEncoder().fit_transform)
    y2 = testing_dataset.iloc[:, 10]

    start = time.time()
    y_pred = Knn.predict(X2)
    prediction_time = time.time() - start
    print("Analitics for: "+"on k: "+str(k))

    print("\n")
    print("Prediction Time: " + str(prediction_time))
    print(confusion_matrix(y2, y_pred, ))
    print(classification_report(y2, y_pred, zero_division="warn"))

def decisicion_tree():
    modes = ["gini","entropy"]
    max_depths = [2,3,4,5,None]
    for mode in modes:
        for max_depth in max_depths:
            Ejercicio3(mode,max_depth)

def Ejercicio3(mode, max_depth):
    dataset = pd.read_csv("genero_peliculas_training.csv")


    #print(dataset)
    X = dataset.iloc[:, :-1]
    #print("X will be")
    #print(X)
    y = dataset.iloc[:,10]
    #print("Y will be")
    #print(y)

    labelEncoder_X = LabelEncoder()
    X = X.apply(LabelEncoder().fit_transform)
    #print(X)

    clf = DecisionTreeClassifier(criterion=mode,max_depth=max_depth).fit(X,y)
    testing_dataset = pd.read_csv("genero_peliculas_testing.csv")
    X2 = testing_dataset.iloc[:, :-1]
    X2 = X2.apply(LabelEncoder().fit_transform)
    y2 = testing_dataset.iloc[:,10]
    #print(X2)
    #print(y2)
    start = time.time()
    y_pred = clf.predict(X2)
    prediction_time = time.time() - start
    print("Analitics for: "+mode+" on depth: "+str(max_depth))
    print("Prediction Time: "+str(prediction_time))

    print("\n")

    print(confusion_matrix(y2,y_pred,))
    print(classification_report(y2, y_pred, zero_division="warn"))




menu = True
while menu:
    print("*****Menu*****")
    print("*Ejercicio 1*")
    print("1. Kmeans.")
    print("2. Agglomerative Clustering.")
    print("3. DBScan.")
    print("*Ejercicio 2*")
    print("4. Knn")
    print("*Ejercicio 3*")
    print("5. Decision Tree Classifier.")
    selection = int(input("Ingrese su eleccion: "))
    if selection == 1:
        k_means_clustering()
        print("Output guardado en la carpeta Kmeans.")
    elif selection == 2:
        agglomerative_clustering()
        print("Output guardado en la carpeta Agglomerative Clustering.")
    elif selection == 3:
        DBScan()
        print("Output guardado en la carpeta DBScan.")
    elif selection == 4:
        knn()
    elif selection == 5:
        decisicion_tree()

    else:
        menu = False



'''
    accuracy = accuracy_score(y2, y_pred)
    print("Accuracy: "+str(accuracy))
    the_average = "micro"
    precision = precision_score(y2, y_pred, average=the_average)
    print("Precision: "+str(precision))
    recall = recall_score(y2, y_pred, average=the_average)
    print("Recall: "+str(recall))
    f_score = f1_score(y2, y_pred, average=the_average)
    print("F1-Score: "+str(f_score))
'''