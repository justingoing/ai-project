import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import folium
from folium import Choropleth

# Load the data
data = pd.read_csv('/mnt/c/Users/Justin Going/Projects/ai-project/data/mahopac-city.csv')

summary = data.describe()
print(summary)

# Scatter plot of Living area vs. Property price (USD)
plt.figure(figsize=(10, 6))
plt.scatter(data['Living area'], data['Property price (USD)'], alpha=0.5)
plt.title('Living Area vs. Property Price (USD)')
plt.xlabel('Living Area')
plt.ylabel('Property Price (USD)')
plt.savefig('/mnt/c/Users/Justin Going/Projects/ai-project/data/rockland/living_area_vs_price.png')
plt.close()

# Histogram of Property Price
plt.figure(figsize=(10, 6))
sns.histplot(data['Property price (USD)'], kde=True)
plt.title('Distribution of Property Prices')
plt.xlabel('Property Price (USD)')
plt.ylabel('Frequency')
plt.savefig('/mnt/c/Users/Justin Going/Projects/ai-project/data/rockland/price_distribution.png')
plt.close()

# Shapiro-Wilk test for normality
stat, p = shapiro(data['Property price (USD)'])
print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Property price follows a normal distribution')
else:
    print('Property price does not follow a normal distribution')

# # Time series property price over time
# plt.figure(figsize=(10, 6))
# data.set_index('Sold date')['Property price (USD)'].resample('M').mean().plot()
# plt.title('Average Property Price Over Time')
# plt.xlabel('Date')
# plt.ylabel('Average Property Price (USD)')
# plt.savefig('/mnt/c/Users/Justin Going/Projects/ai-project/data/rockland/property-price-over-time.png')
# plt.close()


# # Create a folium map
# m = folium.Map(location=[41.15, -74.02], zoom_start=10)

# # Add Choropleth
# Choropleth(
#     geo_data=geo_data,
#     data=geo_data,
#     columns=['Zip', 'Property price (USD)'],
#     key_on='feature.properties.Zip',
#     fill_color='YlOrRd',
#     fill_opacity=0.7,
#     line_opacity=0.2,
# ).add_to(m)

# m.save('rockland_property_prices_map.html')


# Draw an ellipse over 10 random points
subset = data.sample(10)

plt.figure(figsize=(10, 6))
plt.scatter(data['Living area'], data['Property price (USD)'], alpha=0.5)
plt.scatter(subset['Living area'], subset['Property price (USD)'], color='red')

ellipse = Ellipse(xy=(subset['Living area'].mean(), subset['Property price (USD)'].mean()),
                  width=subset['Living area'].std()*2, height=subset['Property price (USD)'].std()*2,
                  edgecolor='red', facecolor='none')
plt.gca().add_patch(ellipse)

plt.title('Living Area vs. Property Price (USD) with Ellipse')
plt.xlabel('Living Area')
plt.ylabel('Property Price (USD)')
plt.savefig('/mnt/c/Users/Justin Going/Projects/ai-project/data/rockland/living_area_vs_price_with_ellipse.png')
plt.close()

# Ensure that the 'City' column is treated as a categorical variable
data['City'] = data['City'].astype('category')

# Box Plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='City', y='Property price (USD)', data=data)
plt.title('Property Prices by City - Box Plot')
plt.xlabel('City')
plt.ylabel('Property Price (USD)')
plt.xticks(rotation=45)  # Rotate city names if they are long
plt.tight_layout()
plt.savefig('/mnt/c/Users/Justin Going/Projects/ai-project/data/rockland/property_prices_boxplot.png')
plt.show()

# Violin Plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='City', y='Property price (USD)', data=data)
plt.title('Property Prices by City - Violin Plot')
plt.xlabel('City')
plt.ylabel('Property Price (USD)')
plt.xticks(rotation=45)  # Rotate city names if they are long
plt.tight_layout()
plt.savefig('/mnt/c/Users/Justin Going/Projects/ai-project/data/rockland/property_prices_violinplot.png')
plt.show()

# PCA
features = ['Living area', 'Property price (USD)', 'Lot/land area', 'Bedrooms', 'Bathrooms']
x = data[features].dropna()

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

plt.figure(figsize=(10, 6))
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], alpha=0.5)
plt.title('PCA of Property Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('/mnt/c/Users/Justin Going/Projects/ai-project/data/rockland/pca.png')
plt.close()
