import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import base64
from sklearn.metrics import silhouette_samples, silhouette_score
# For Calinski-Harabasz Index quality metric
from sklearn.metrics import calinski_harabasz_score
# For Fowlkes-Mallows Index quality metric
from sklearn.metrics.cluster import fowlkes_mallows_score

# Image Arguments & Clustering Evaluation
df = pd.read_csv("clusteredData.csv")

true_labels = np.loadtxt(open("product_images.csv", "rb"), delimiter=",", skiprows=1)
true_labels = true_labels[:,0]

imageData = np.loadtxt(open("product_images.csv", "rb"), delimiter=",", skiprows=1)

cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

def formatImage(image): return np.array([image[i:i+28] for i in range(0, 784, 28)])

def plotImage(idx = int, data = np.array, colour = str):
    fig, ax = plt.subplots(figsize=(10, 8))
    chosenImage = ax.imshow(formatImage(data[idx]), cmap=colour)
    ax.axis('off') # remove the axis
    return st.pyplot(fig=chosenImage.axes.figure, clear_figure=True)

# Set Background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('image_file.jpeg')

# Products Page
def main():
    # Create a banner at the top with links to other pages
    with st.container():
        st.markdown(
            """
            <div style='background-color: white; padding: 1px;'>
            <h1 style='font-family:Optima;color: #8B4513; text-align: center;'>Jorge & Jeff</h1>
            <p style='font-family: Optima;color: #8B4513; text-align: center; font-size: 20px;'> 
            <a style='color: #8B4513; text-decoration: none;' href='https://charlotteg1224-scenarioweek-homepage-zb4jfq.streamlit.app/'target='_blank'>Home</a> | 
            <a style='color: #8B4513; text-decoration: none;' href='https://ewanyeo-search-searchpage-ga2mq2.streamlit.app/'target='_blank'>Search</a> | 
            <a style='color: #8B4513; text-decoration: none;' href='https://ewanyeo-readytowear-productspage-14irvc.streamlit.app/'target='_blank'>Ready To Wear</a> | 
            <a style='color: #8B4513; text-decoration: none;' href='https://rajatk21-sw-doggy-pe2mzu.streamlit.app/' target='_blank'>Team</a> |
            <a style='color: #8B4513; text-decoration: none;' href='https://georginapalmer-contactus-streamlitcontactus-49oqai.streamlit.app/'target='_blank'>Contact</a> 
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )
if __name__ == '__main__':
     main()

# leave space
st.title("")
st.text("")
#Create Cols
col1, col2 = st.columns(2)

#Display in Cols
with col1:
# Cluster 0: 6899 Tops
    st.image("J&Jshirt.png", width = 200)
    chosen_item = df.loc[6899, 'Segment K-means PCA']
    cluster0 = df.loc[df['Segment K-means PCA'] == chosen_item].index.values.tolist()
    if st.button('Tops'):
        for i in range(6):
             st.image(formatImage(imageData[random.choice(cluster0)]), width = 200, clamp = True)
    else:
        st.text("")
# Cluster 13: 3099 bag
    st.image("J&Jhandbags.png", width = 200)
    chosen_item = df.loc[3099, 'Segment K-means PCA']
    cluster13 = df.loc[df['Segment K-means PCA'] == chosen_item].index.values.tolist()
    if st.button('Bags'):
        for i in range(6):
            st.image(formatImage(imageData[random.choice(cluster13)]), width = 200, clamp = True)
    else:
        st.text("")
st.title("")
st.title("")
with col2:
    # Cluster 4: 100 Boots
    st.image("J&Jboots.png", width = 200)
    chosen_item = df.loc[100, 'Segment K-means PCA']
    cluster4 = df.loc[df['Segment K-means PCA'] == chosen_item].index.values.tolist()
    if st.button('Boots'):
        for i in range(6):
            st.image(formatImage(imageData[random.choice(cluster4)]), width = 200, clamp = True)
    else:
        st.text("")
    # Cluster 9: Footwear
    st.image("J&Jshoes.png", width = 200)
    chosen_item = df.loc[2001, 'Segment K-means PCA']
    cluster9 = df.loc[df['Segment K-means PCA'] == chosen_item].index.values.tolist()
    if st.button('Footwear'):
        for i in range(6):
            st.image(formatImage(imageData[random.choice(cluster9)]), width = 200, clamp = True)
    else:
        st.text("")


# leave space
st.title("")
st.text("")

# Cluster Evaluation Metrics
cluster_labels = df.iloc[:,-1]
scores_pca = df.iloc[:,:-1]

silhouette_score = str("Silhouette score: " + str(silhouette_score(scores_pca, cluster_labels)))
calinsky_harabasz = str("Calinsky-Harabasz: " + str(calinski_harabasz_score(scores_pca, cluster_labels)))
fowlkes_mallows = str("Fowlkes-Mallows: " + str(fowlkes_mallows_score(true_labels, cluster_labels)))

metrics_title = '<strong><p style="font-family:Optima; color:#8B4513; font-size: 23px;">Quality Metrics:</p></strong>'
st.markdown(metrics_title, unsafe_allow_html=True)
metrics = st.selectbox(
    '',
    ("Silhouette Score", "Calinsky-Harabasz", "Fowlkes-Mallows"))

if metrics == "Silhouette Score":
    st.write(silhouette_score)
elif metrics == "Calinsky-Harabasz":
    st.write(calinsky_harabasz)
elif metrics == "Fowlkes-Mallows":
    st.write(fowlkes_mallows)
else:
    st.write("")

# leave space
st.title("")
st.text("")