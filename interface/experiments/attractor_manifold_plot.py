import re
import json
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns
import pandas as pd
import umap


sns.set_theme(style='darkgrid')

with open(sys.argv[1], 'r') as f:
    contents = json.load(f)

regex_pattern = r'trial: (\d+), pattern: (\d+), distortion: (\d+\.*\d*)'

rows = []
for key, value in contents.items():
    if key == 'patterns':
        continue
    trial = float(re.search(regex_pattern, key).group(1))
    current_pattern = float(re.search(regex_pattern, key).group(2))
    distortion = float(re.search(regex_pattern, key).group(3))

    rows.append([trial, current_pattern, distortion] + value['firing_rates'])

patterns = contents['patterns']

neuron_cols = [str(i) for i in range(len(list(contents.values())[0]['firing_rates']))]
cols = ['trial', 'pattern', 'distortion'] + neuron_cols

df = pd.DataFrame(rows, columns=cols)

reducer = umap.UMAP(n_components=3)

data = df[neuron_cols].values
scaled_data = StandardScaler().fit_transform(data)

embedding = reducer.fit_transform(scaled_data)

def acc(patterns, num_patterns, desired_pattern_index, firing_data):
    correlation_coefficients = []
    for pattern_index in range(num_patterns):
        correlation_coefficients.append(
            np.corrcoef(patterns[pattern_index], firing_data)[0, 1]
        )
        
    return bool(desired_pattern_index == np.argmax(correlation_coefficients))

accs = {}
for n, i in df.iterrows():
    desired_pattern_index = i['pattern']
    accs[n] = acc(patterns, 3, desired_pattern_index, i[neuron_cols].to_numpy())

selected_df = df[df.index.isin([key for key, value in accs.items() if value])]
selected = selected_df[neuron_cols].values
scaled_selected = StandardScaler().fit_transform(selected)

selected_embedding = reducer.transform(scaled_selected)

custom_colors = ['#a83232', '#8b32a8', '#3a32a8']
pattern_to_color = {pattern: custom_colors[i % len(custom_colors)] for i, pattern in enumerate(df['pattern'].unique())}
colors = [
    custom_colors[pattern] for pattern in selected_df['pattern'].map({i: int(i) for i in range(3)})
]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(
    selected_embedding[:, 0],
    selected_embedding[:, 1],
    selected_embedding[:, 2],
    c=colors
)
plt.title('Attractor States')
plt.show()
