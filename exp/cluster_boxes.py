import cv2
import numpy as np
from sklearn.cluster import DBSCAN,KMeans
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('Connected Components', labeled_img)
    cv2.waitKey()

def get_component(stats, i):
    x = stats[i,cv2.CC_STAT_LEFT]
    y = stats[i,cv2.CC_STAT_TOP]
    w = stats[i,cv2.CC_STAT_WIDTH]
    h = stats[i,cv2.CC_STAT_HEIGHT]
    return x,y,w,h

def get_cc(image, size_increase=3, height_range=(30,500)):
    size = (image.shape[1] * size_increase, image.shape[0] * size_increase)

    img = cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,10)

    _, _,stats,_ = cv2.connectedComponentsWithStats(img)

    components = []
    for i in range(len(stats)):
        x,y,w,h = get_component(stats,i)
        x = int(x/size_increase)
        y = int(y/size_increase)
        w = int(w/size_increase)
        h = int(h/size_increase)
        if height_range[0] < h < height_range[1]:
            components.append ( [x,y,w,h] )
    components = sorted(components,key=lambda x: x[0])

    return np.array(components)


def get_clustered_boxes(image, all_points=True):
    
    cc = get_cc(image)

    viz_img = image.copy()
    for x,y,w,h in cc:
        cv2.rectangle(viz_img, (x,y), (x+w,y+h), color=(0,0,255 ), thickness=1)
    # cv2.imshow('Clusters',viz_img)
    # cv2.waitKey() 


    clusters = get_clusters(image,cc)
    # plot_clusters(image,clusters)

    if all_points:
        
        new_clusters = []
        for cluster in clusters:
            new_cluster = np.zeros( (cluster.shape[0],4,2) ,dtype='float32')
            new_cluster[:,0] = cluster[:,[0,1]] #tl
            new_cluster[:,1] = np.array([cluster[:,0] + cluster[:,2], cluster[:,1]]).T # tr
            new_cluster[:,2] =  np.array([cluster[:,0] + cluster[:,2], cluster[:,1]+cluster[:,3]]).T # br
            new_cluster[:,3] = np.array([cluster[:,0] , cluster[:,1]+cluster[:,3]]).T # bl    
            new_clusters.append(new_cluster)
        clusters = new_clusters
        
    
    return clusters

    
def get_clusters(image,components, n_clusters=3,min_samples=1,max_samples=20,height_range=(30,500)):
    X = components[:,3].reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters,tol=1e5,random_state=42)
    clusters = [components]
    if len(X) > n_clusters:
        kmeans.fit(X)

        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels,return_counts=True)
        unique_labels = [l for l,c in zip(unique_labels,counts) if min_samples <= c <= max_samples ]

        clusters = [components[labels==l] for l in unique_labels]

        clusters = sorted(clusters,key = lambda x: x[:,3].max() )

    return clusters

def plot_clusters(image,clusters):
    image = image.copy()
    colors = [[x * 255 for x in plt.cm.Spectral(each)] for each in np.linspace(0, 1, len(clusters))]
    
    for cluster,col in zip(clusters, colors):
        for x,y,w,h in cluster:
            cv2.rectangle(image, (x,y), (x+w,y+h), color=col[:-1], thickness=1)

    cv2.imshow('Clusters',image)
    cv2.waitKey() 
    


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default='../data/test_dataset/images/111.png',
        help="path to input image to be OCR'd")
    args = vars(ap.parse_args())

    print(args)
    path = Path(args["image"])
    if path.is_dir():
        for p in path.iterdir():
            img = cv2.imread(str(p))
            get_clustered_boxes(img)
    else:
        img = cv2.imread(str(path))
        get_clustered_boxes(img)