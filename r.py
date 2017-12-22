# coding: utf8
from scipy.cluster.hierarchy import *
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA 
import pandas as pd
from pylab import *

rcParams['font.family'] = 'DejaVu Sans' 
rcParams['font.size'] = 16

data = pd.read_excel('data.xls')

labels =  list(data['name'])

data_for_clust =  data.ix[:,1:].as_matrix()

data_dist = pdist(data_for_clust, 'euclidean')

data_linkage = linkage(data_dist, method='average')

def getNames (namelist, datalist, k):
    
    rez = list()
    for i in range(0,k):
        rez.append('')
        
    for i in range(1,k+1):
        for j in range(0,10):
            if datalist[j] == i:
                rez[i-1] += namelist[j] + ' '
    return rez


# ------------- Кластеризация --------------
pca = PCA(n_components=2) 
Xt = pca.fit_transform(data_for_clust)
fig = figure()
for k in range(5, 11):
    groups = fcluster(data_linkage, k, criterion='maxclust')
    namel = getNames(list(data['name']), groups, k)
    ax = fig.add_subplot(3, 2, k-4)
    for j, m, c in zip(range(k), 'so><^vs><^', 'rgbmycgrmb'):
        ax.scatter(Xt[groups==(j+1), 0], Xt[groups==(j+1), 1], marker=m, s=30, label='%s'%namel[j], facecolors=c)
        ax.set_title('k=%s'%k)
        ax.legend(fontsize=8, loc="lower right")
fig.suptitle(u'Кластеризация')
# ---------------------------------------------------------------------

print(fcluster(data_linkage, k, criterion='maxclust'))
print(Xt)
print(data_for_clust)


show()









