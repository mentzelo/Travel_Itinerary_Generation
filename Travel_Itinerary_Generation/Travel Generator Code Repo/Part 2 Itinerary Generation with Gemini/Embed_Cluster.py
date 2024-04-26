# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt




def embed_d2v(path, vector_size=100, window=5, min_count=2, workers=4):
    #### We need to select the vector size(how large should our vector be to capture data), window(number of context words before/behind each token)
    #### min_count(ignore all words with frequency lower than this) and workers(how many worker threads to train model)
    #### Here we read in the CSV file which has rows as reviews, and columns as review attributes(text, photos etc)
    df = pd.read_csv( path, low_memory = False)

    #### We now group reviews that are from the same location together and concatenate them together to form long form text
    grouped_reviews = df.groupby('placeInfo/name')
    concatenated_reviews = grouped_reviews['text'].apply(lambda x: ' '.join(x))
    reviews_by_entity = pd.DataFrame(concatenated_reviews)
    reviews_by_entity.index = reviews_by_entity.index.get_level_values('placeInfo/name')
    df = reviews_by_entity



    col = 'text'
    # Prepare a list of tagged documents for Doc2Vec
    documents = [TaggedDocument(doc.split(), [col + '_' + str(idx)]) for idx, doc in df[col].iteritems()]

    # Train a Doc2Vec model for this column
    model = Doc2Vec(documents, vector_size = vector_size, window = window, min_count = min_count, workers = workers, epochs=40)  # Adjust hyperparameters as needed

    # Embed each document in the column
    column_embeddings = [model.infer_vector(doc.words) for doc in documents]

    # Add a new column for the embeddings
    df[col + '_embeddings'] = column_embeddings

    return df, model

def kmeans_cluster(df, embedding_column, n_clusters, random_state):
    #### Specify the df column in which the embeddings are located e.g. df['text_embeddings']
    #### n_clusters, and random state
    # Extract embeddings as a NumPy array
    embeddings = np.vstack(embedding_column.tolist())
    embeddings = normalize(embeddings)
    # Calculate cosine similarity matrix (1 - cosine distance)
    dist_matrix = 1 - squareform(pdist(embeddings, metric='cosine'))


    ## K means below for example

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)  # Set a random state for reproducibility
    clusters = kmeans.fit_predict(embeddings)


    # Assign cluster labels to the DataFrame
    df['cluster'] = clusters

    return df


def kmeans_evaluate(df, max_clusters, max_iter):
    #### We evaluate different numbers of clusters by their inertia
    sse = {}
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, max_iter=max_iter).fit(embeddings)
        df["cluster"] = kmeans.labels_
        #print(data["clusters"])
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.show()



def vec_similarity(text, model):

    #### Function to give you top 10 document tags and their cosine similarity
    #### text is a string, model is the model

    tokens = text.split()
    new_vector = model.infer_vector(tokens)
    sims = model.dv.most_similar([new_vector])

    return sims



def find_matches(df, group, rating_column):
    ##This function inputs a specific component name, and dataframe with components and their respective cluster, ratings, and returns
    ## a list of components in the same cluster, ordered by rating.
    ## cluster column  = name of column that has cluster(usually 'cluster'),


    #group = df.loc[df['name'] == name, 'cluster'].iloc[0]  # Get the group of the given location
    other_locations = df.loc[(df['cluster'] == group)]  # Filter for other locations in the same group
    other_locations = other_locations.sort_values(by=rating_column, ascending=False)
    return list(zip(other_locations['name'].tolist(), other_locations[rating_column].tolist()))

def get_components(days, prefs, df, factor):
    ##Prefs are a list of the numerical clusters representing a user's preferences, days is the length of trip in days,
    ## factor is the scale factor on how many components per day we want to generate.
    ## max number of components is the maximum of either days*factor, or number of clusters selected by user
    max_components = max(days*factor, len(prefs))
    components = []
    top_components = {}
    #Creates a dictionary with keys = cluster #, and values are ordered list of components by rating
    for i in prefs:
        top_components[i] = find_matches(df, i, 'numberOfReviews') ####Can use rawRanking which is Tripadvisor quality measure,
        ## Or can use numberOfReviews which is a proxy for popularity
    #We iterate through our dictionary adding one representative from each cluster at a time to our final components list until
    #the list has reached our maximum
    j = 0
    while len(components) < max_components:
        for i in prefs:
            if top_components[i][j][0] not in components: ###add index 0 if you want to leave out the raw rating in output
                components.append(top_components[i][j][0]) ###add index 0 if you want to leave out the raw rating in output
            else:
                j = j+1
                components.append(top_components[i][j][0])  ###add index 0 if you want to leave out the raw rating in output
    return components