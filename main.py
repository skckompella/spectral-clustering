import scipy.io
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


temp = scipy.io.loadmat('graphData.mat')
graph_data = temp['data_struct'][0][0] #Coz of some weird representation of the data
#names_data = temp['data_struct'][0][1]
#names_data = []
names_data = ['Adler - T', 'Adrion - S', 'Allan - A', 'Avrunin - S', 'Barto - A', 'Barrington - T', 'Berger - S', 'Brock - A', 'Clarke - S', 'Corner - S', 'Croft - A',\
              'Ganesan - S', 'Grupen - A', 'Hanson - A', 'Immerman - T', 'Jensen - A', 'Kaplan - S', 'Kurose - S', 'Learned_Miller - A', 'Lehnert - A', 'Lesser - A',\
              'Levine - S', 'Mahadevan - A', 'Manmatha - A', 'McCallum - A', 'Moll - A', 'Moss - S', 'Osterweil S', 'Riseman - A', 'Rissland - A', 'Rosenberg - T',\
              'Schultz - A', 'Shenoy - S', 'Sitaraman - T', 'Towsley - S', 'Utgoff - A', 'Weems - S', 'Wilden - S', 'Woolf - S', 'Zilberstein - A']

category_data = ['T', 'S', 'A', 'S', 'A', 'T', 'S', 'A', 'S', 'S', 'A', 'S', 'A', 'A', 'T', 'A', 'S', 'S', 'A', 'A', 'A',\
              'S', 'A', 'A', 'A', 'A', 'S', 'S', 'A', 'A', 'T', 'A', 'S', 'T', 'S', 'A', 'S', 'S', 'S', 'A']

#affinity_mat = np.zeros(graph_data.shape)
#sigma = 1
#for i in range(graph_data.shape[0]):
#    for j in range(graph_data.shape[0]):
#        affinity_mat[i][j] = np.exp(-((np.sum(np.square(graph_data[i] - graph_data[j])))/(2*(sigma**2))))

affinity_mat = graph_data

np.fill_diagonal(affinity_mat, 0)

row_sums = affinity_mat.sum(axis=1)
row_sum_diag_mat = np.diag(row_sums)

laplacian_mat = np.sqrt(np.linalg.inv(row_sum_diag_mat)).dot(affinity_mat).dot(np.sqrt(np.linalg.inv(row_sum_diag_mat)))

eigen_vals, eigen_vecs = np.linalg.eig(laplacian_mat)
sorted_val_indices = np.argsort(eigen_vals)
sorted_val_indices = sorted_val_indices[::-1] #flipping the array for descending order

k = 6

eigen_mat = np.ndarray(shape=(eigen_vecs.shape[0], k))

for i in range (0,k):
    eigen_mat[:,i] = eigen_vecs[:,sorted_val_indices[i]]

eigen_row_sums = np.sqrt(np.sum(np.square(eigen_mat) ,axis=1) )
#print eigen_row_sums
normalized_eigen_mat = eigen_mat / eigen_row_sums[:, np.newaxis]  #row_sums[:, numpy.newaxis] reshapes row_sums from being (3,) to being (3, 1). When you do a / b, a and b are broadcast against each other.


kmeans = KMeans(n_clusters=k, random_state=0).fit(normalized_eigen_mat)

predictions = kmeans.labels_
mapping = defaultdict(list)
count = defaultdict(list)
for i in range(0,k):
    c = [0,0,0]
    vals = np.where(predictions == i)[0]
    profs = []
    for j in vals:
        profs.append(names_data[j])
        if category_data[j] == 'A':
            c[0]+=1
        elif category_data[j] == 'S':
            c[1]+=1
        else:
            c[2]+=1
    mapping[i] = profs
    count[i] = c

for i in range(k):
    #print count[i]
    print "Class: " + str(i)
    print mapping[i]







