from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

##############################
#Part 1

def get_data_blobs(n_points = 100):
  x, y = make_blobs(n_samples = n_points, cluster_std = 0.2, random_state = 42)
  return x,y

def get_data_circles(n_points = 100):
  x, y = make_circles(n_samples = n_points, noise = 0.1, random_state = 42)
  return x,y

def build_kmeans(X = None, k = 10):
  kmeans = KMeans(n_clusters = k, random_state = 42)
  km = kmeans.fit(X)
  return km

def assign_kmeans(km = None, X = None):
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1 = None, ypred_2 = None):
  homo = homogeneity_score(ypred_1, ypred_2)
  comp = completeness_score(ypred_1, ypred_2)
  vscore = v_measure_score(ypred_1, ypred_2)
  return homo, comp, vscore

##############################
