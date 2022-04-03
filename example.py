import streamlit as st
import pandas as pd
import altair as alt
from matplotlib import pyplot as plt
from urllib.error import URLError
import random
import numpy as np
from copy import deepcopy


st.write("--- Data Frame to cluster on ---")

data = pd.read_csv("xclara.csv")
st.dataframe(data, 500, 200)
f1 = list(data['V1'])
f2 = list(data['V2'])
X = np.array(list(zip(f1, f2)))

c = alt.Chart(data).mark_circle().encode(x='V1', y='V2')

st.altair_chart(c, use_container_width=True)
cluster_number = st.slider('Number of clusters', 0, 6 , 1)
st.write("Number of clusters : ", cluster_number)

def dist(a, b, ax=1):
     return  np.linalg.norm(a - b, axis=ax)
# # Number of clusters
k=cluster_number
# # X coordinates of random centroids
C_x = np.random.randint(0, np.max(f1), size=k)

# # Y coordinates of random centroids
C_y = np.random.randint(0, np.max(f2), size=k)
df = pd.DataFrame({"X":C_x,"Y": C_y})


fig = plt.figure(figsize=(10, 4))

plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
st.pyplot(fig)

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

# print(C)

# Initizie C_old which will store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Intitialize Cluster Lables

clusters = []
for i in range(k):
    clusters.append([])
# Error func. - Distance between new centroids and old centroids
error = dist(C_old, C, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assign each value to its closest cluster
    clusters = []
    for i in range(k):
        clusters.append([]) 
    for i in range(len(X)):
        distance = dist(X[i], C)
        label = np.argmin(distance)
        clusters[label].append(X[i])
        
    # Store the old centroid values
    C_old = deepcopy(C)
    # Find the new centroids by taking the average value
    means = []
    for i in range(len(clusters)):    
         means.append(list(np.mean(np.array(clusters[i]), axis=0)))
    C = np.array(means)
    error = dist(C_old, C, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig1, ax = plt.subplots()

# print(clusters)
for i in range(k):
        points = np.array(clusters[i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

st.pyplot(fig1)
