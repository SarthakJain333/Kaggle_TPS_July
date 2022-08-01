import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import plotly.express as px

data = pd.read_csv(r'Programming\tabular-playground-series-jul-2022\data.csv')

print(data.head(10))
print(data.info()) # From here we observe
# that there are 7 discrete and 22 continuous features
print(data.shape)

fig = plt.figure(figsize=(15,14))

for i in range(7):
    plt.subplot(4,2,i+1)
    feature_no = i+7
    sns.countplot(x=data.iloc[:,feature_no])
    plt.title(f'feature_no :- {feature_no}')
    plt.xlim([-1,44])
    plt.ylim([0,11000])
plt.show()

continuous_features = [f'f_0{i}' for i in range(7)]
continuous_features = continuous_features + [f'f_{i}' for i in range(14,29)]
fig = plt.figure(figsize=(15,14))

for i,f in enumerate(continuous_features):
    plt.subplot(6,4,i+1)
    sns.histplot(x=data[f])
    plt.title(f'Feature :- {f}')
plt.show()

sns.heatmap(data.corr().abs(), vmin=0, vmax=1)
# data.corr() is used to find the pairwise correlation 
# of all the columns in the dataframe.Correlation of a 
# variable with itself is 1.
 
plt.show()

### Using elbow method we can easily conclude that preferred no of clusters is 7.
inertias = []
for k in range(1,30):
    km = KMeans(n_clusters=k)
    km.fit(data.iloc[:10000])
    inertias.append(km.inertia_)

plt.figure(figsize=(10,10))
plt.plot(range(1,30), inertias, 'bx-')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.title('ELBOW METHOD')
plt.show()
### Power Transformer is preferred as it converts the data more into gaussian as compared
### to other scaling methods.
pt_scaled_data = pd.DataFrame(PowerTransformer().fit_transform(data))
pt_scaled_data.columns = data.columns
print(data.head(10))
# model_km = KMeans(n_clusters = 7, random_state=0)
# preds_km = model_km.fit_predict(pt_scaled_data)

submission = pd.read_csv(r'Programming\tabular-playground-series-jul-2022\sample_submission.csv')
# submission['Predicted'] = preds_km
# submission.to_csv('submission_kmeans.csv', index=False)


### First let's do PCA and then apply GaussianMixtureModel.
pca = PCA(n_components=3)
components = pca.fit_transform(pt_scaled_data)

model_gm = GaussianMixture(n_components=7, random_state=0)
preds_gm = model_gm.fit_predict(pt_scaled_data)
submission['Predicted'] = preds_gm
submission.to_csv('sub_wqas.csv', index=False)



