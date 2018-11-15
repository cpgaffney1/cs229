from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import math

n_clusters = 5
min_yr = 1815.
max_yr = 2011.

def load_data():
    # returns pandas dataframes
    return pd.read_csv('data/disputesCountryPairsCcode.csv')

def filter_data(data):
    drop_cols = ['dispnum3', 'styear', 'endyear', 'a_styear', 'a_endyear', 'b_styear', 
        'b_endyear', 'war_names', 'a_revtype2', 'b_revtype2']
    filtered_data = data.drop(drop_cols, axis=1)
    return filtered_data
    
def get_year_colors(years):
    # Normalize to 0-1
    colors = []
    for i in range(len(years)):
        yr = years[i]
        ratio = (yr-min_yr) / (max_yr - min_yr)
        assert(ratio >= 0 and ratio <= 1)
        b = int(max(0, 255*(1 - ratio)))
        r = int(max(0, 255*(ratio - 1)))
        g = 255 - b - r
        colors.append([r, g, b])
    return colors
    
def plot_clusters(x, clusters):
    assert(len(clusters) == len(x))
    assert(x.shape[1] == 2)
    fig, ax = plt.subplots()
    #plt.xscale('symlog')
    #plt.yscale('symlog')
    ax.scatter(x[:, 0], x[:, 1], c=clusters, s=1)
    
        
    plt.show()
    plt.clf()
    
def plot_wars(x, war_names):
    assert(x.shape[1] == 2)
    fig, ax = plt.subplots()
    ww1_points = []
    ww2_points = []
    for i, war in enumerate(war_names):
        label = None
        if war is np.nan or war == np.nan:
            color = 'k'
        elif war == 'WW1':
            color = 'b'
            ww1_points.append((x[i,0], x[i,1]))
        elif war == 'WW2':
            color = 'r'
            ww2_points.append((x[i,0], x[i,1]))
        else:
            print(war)
            color = 'g'
            label = war
        if color == 'k':
            ax.scatter(x[i, 0], x[i, 1], c=color, s=0.1)
        else:
            ax.scatter(x[i, 0], x[i, 1], c=color, s=10)
        #if label is not None:
            #ax.annotate(label, (x[i,0], x[i,1]))    
    
    ww1_points = np.array(ww1_points)
    ww2_points = np.array(ww2_points)
    
    pt = np.mean(ww1_points, axis=0)
    #ax.annotate('WW1', (pt[0], pt[1]))
    pt = np.mean(ww2_points, axis=0)
    #ax.annotate('WW2', (pt[0], pt[1]))
    
    #plt.xscale('symlog')
    #plt.yscale('symlog')
        
    plt.show()
    plt.clf()
    
    
def plot_year_pca(x, years):
    assert(len(x) == len(years))
    assert(x.shape[1] == 2)
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=min_yr, vmax=max_yr)
    #assert(len(colors) == len(years))
    #plt.xscale('symlog')
    #plt.yscale('symlog')
    plt.scatter(x[:, 0], x[:, 1], c=cmap(norm(years)))
    plt.show()
    

data = load_data()
filtered_data = filter_data(data)
x = filtered_data.as_matrix()
years = data['styear'].as_matrix()
x = np.log(x + 10)
assert(len(years) == filtered_data.shape[0])
assert(len(x) == filtered_data.shape[0])
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
clusters = kmeans.labels_

pca = PCA(n_components=2)
pca.fit(x)
x = pca.transform(x)
#x[:, 0] += np.abs(np.min(x)) + 1
#x[:, 1] += np.abs(np.min(x)) + 1
#x = np.log(x)
#x = x / float(10e5)

plot_clusters(x, clusters)
plot_year_pca(x, years)
plot_wars(x, data['war_names'])





    
