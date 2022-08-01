import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.metrics import roc_auc_score


df = pd.read_csv(r'tabular-playground-series-jul-2022\data.csv', index_col=False)
df.drop(['id'], axis=1, inplace=True)
print(df.head(10))

categorical_col = [x for (x,y) in df.dtypes.items() if y == 'int64']
num_col = [x for (x,y) in df.dtypes.items() if y == 'float64']

num_data = df[num_col]
cat_data = df[categorical_col]

print('Categorical Columns', cat_data.columns)
print('Numerical data', num_data.columns)

sns.heatmap(df.corr().abs(), vmin=0, vmax=1)
plt.show()
for i in range(7):
    df.drop([f'f_0{i}'],axis=1, inplace=True)
for i in range(14,22):
    df.drop([f'f_{i}'],axis=1, inplace=True)

scaler = PowerTransformer()
x_train = scaler.fit_transform(df)

pca = PCA(n_components=3)
x_pca = pca.fit_transform(x_train)

km = KMeans(n_clusters=7)
km_result = km.fit_predict(x_train)


# km2 = KMeans(n_clusters=5)
# km_result2 = km2.fit_predict(x_train)

submission_sample_df = pd.read_csv(r'tabular-playground-series-jul-2022\sample_submission.csv')
# submission_sample_df['Predicted'] = np.round((km_result+km_result2)/2)
gm = GaussianMixture(n_components=6)
gm_result = gm.fit_predict(x_train)
submission_sample_df['Predicted'] = gm_result
submission_sample_df.to_csv('submission.csv', index=False)

