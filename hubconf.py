from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc

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
#Part 2

def get_data_mnist():
  data = datasets.load_digits()
  x = data.data
  y = data.target
  return x, y

def build_lr_model(X = None, y = None):
  lr = LogisticRegression(random_state = 42)
  lr_model = lr.fit(X, y)
  return lr_model

def build_rf_model(X=None, y=None):
  rf = RandomForestClassifier(max_depth = 3, random_state = 42)
  rf_model = rf.fit(X, y)
  return rf_model

def get_metrics(model1 = None, X = None, y = None):
  y_pred = model1.predict(X)
  acc = accuracy_score(y, y_pred)
  prec = precision_score(y, y_pred, average = "weighted")
  rec = recall_score(y, y_pred, average = "weighted")
  f1 = f1_score(y, y_pred, average = "weighted")
  tpr, fpr, thres = roc_curve(y, y_pred, pos_label = 2)
  auc_ = auc(fpr, tpr)
  return acc, prec, rec, f1, auc_

##############################
