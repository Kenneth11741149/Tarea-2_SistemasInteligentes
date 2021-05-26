import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
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



menu = True
while menu:
    print("Menu")
    selection = int(input("Ingrese su eleccion: "))
    if selection == 1:
        k_means_clustering()
    elif selection == 2:
        agglomerative_clustering()
    elif selection == 3:
        DBScan()
    else:
        menu = False







'''
dataset = pd.read_csv("genero_peliculas_training.csv")
print(dataset)
X = dataset.iloc[:,:-1]
print("X will be")
#print(X)
y = dataset.iloc[:,5]
print("Y will be")
print(y)

from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X = X.apply(LabelEncoder().fit_transform)
print(X)

'''

from sklearn.tree import DecisionTreeClassifier

'''
iris = load_iris()
X,y = iris.data, iris.target
print("data")
print(X)
print("target")
print(y)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
tree.plot_tree(clf)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

'''