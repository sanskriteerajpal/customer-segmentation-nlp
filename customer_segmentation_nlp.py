import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample dataset
# Replace with actual marketing/customer data
df = pd.read_csv("sample_marketing_data.csv")

# Feature selection for segmentation
features = ['annual_income', 'spending_score']
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['segment'] = kmeans.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df['PC1'], df['PC2'] = components[:, 0], components[:, 1]

# Plot segments
sns.scatterplot(data=df, x='PC1', y='PC2', hue='segment')
plt.title('Customer Segments')
plt.show()

# Sentiment analysis on reviews
df['sentiment'] = df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
sns.histplot(df['sentiment'], kde=True)
plt.title('Sentiment Distribution')
plt.show()