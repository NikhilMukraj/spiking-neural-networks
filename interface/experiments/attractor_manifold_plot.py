import re
import json
import toml
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns
import pandas as pd
from pipeline_setup import correlation_acc
import joblib
import umap


sns.set_theme(style='darkgrid')

with open(sys.argv[1], 'r') as f:
    args = toml.load(f)

if 'plot_args' not in args:
    raise ValueError('Requires plot_args table')
if 'firing_data' not in args['plot_args']:
    raise ValueError('plot_args requires firing_data argument')

with open(args['plot_args']['firing_data'], 'r') as f:
    contents = json.load(f)

patterns = contents['patterns']

if 'plot_all_data' not in args['plot_args']:
    args['plot_all_data'] = True
if 'plot_high_accuracy_only_bounded_data' not in args['plot_args']:
    args['plot_args']['plot_high_accuracy_only_bounded_data'] = False
if args['plot_args']['plot_high_accuracy_only_bounded_data']:
    if 'bounding_percent' not in args['plot_args']:
        args['plot_args']['bounding_percent'] = 0.5

patterns = contents['patterns']
num_patterns = len(contents['patterns'])

if 'colors' in args['plot_args']:
    pattern_colors = args['plot_args']['colors']
else:
    raise ValueError('plot_args requires colors argument')

if 'reducer_args' not in args:
    args['reducer_args'] = {'reducer_all_data' : None, 'reducer_high_accuracy_only_bounded' : None}
else:
    if 'reducer_all_data' not in args['reducer_args']:
        args['reducer_args']['reducer_all_data'] = None
    if 'reducer_all_data' not in args['reducer_args']:
        args['reducer_args']['reducer_all_data'] = None

regex_pattern = r'trial: (\d+), pattern: (\d+), distortion: (\d+\.*\d*)'

rows = []
for key, value in contents.items():
    if key == 'patterns':
        continue
    trial = float(re.search(regex_pattern, key).group(1))
    current_pattern = float(re.search(regex_pattern, key).group(2))
    distortion = float(re.search(regex_pattern, key).group(3))

    rows.append([trial, current_pattern, distortion] + value['firing_rates'])

neuron_cols = [str(i) for i in range(len(list(contents.values())[0]['firing_rates']))]
cols = ['trial', 'pattern', 'distortion'] + neuron_cols

df = pd.DataFrame(rows, columns=cols)

print('Loaded data...')

if args['plot_args']['plot_all_data']:
    reducer = umap.UMAP(n_components=3)

    data = df[neuron_cols].values
    scaled_data = StandardScaler().fit_transform(data)

    embedding = reducer.fit_transform(scaled_data)

    selected = df[neuron_cols].values
    scaled_selected = StandardScaler().fit_transform(selected)

    selected_embedding = reducer.transform(scaled_selected)

    pattern_to_color = {pattern: pattern_colors[i % len(pattern_colors)] for i, pattern in enumerate(df['pattern'].unique())}
    colors = [
        pattern_colors[pattern] for pattern in df['pattern'].map({i: int(i) for i in range(3)})
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

    if args['reducer_args']['reducer_all_data'] is not None:
        joblib.dump(reducer, args['reducer_args']['reducer_all_data'])

if args['plot_args']['plot_high_accuracy_only_bounded_data']:
    mean_firing_rate = np.array([i[neuron_cols].mean() for _, i in df.iterrows()]).mean()

    accs = {}
    for n, i in df.iterrows():
        desired_pattern_index = i['pattern']
        current_firing_rate = i[neuron_cols].mean()
        if current_firing_rate < mean_firing_rate * args['plot_args']['bounding_percent'] \
        or current_pattern > mean_firing_rate * (1 + args['plot_args']['bounding_percent']):
            continue
        accs[n] = correlation_acc(patterns, num_patterns, desired_pattern_index, i[neuron_cols].to_numpy())

    selected_df = df[df.index.isin([key for key, value in accs.items() if value])]
    selected = selected_df[neuron_cols].values
    scaled_selected = StandardScaler().fit_transform(selected)

    selected_reducer = umap.UMAP(n_components=3)
    selected_embedding = selected_reducer.fit_transform(scaled_selected)

    pattern_to_color = {pattern: pattern_colors[i % len(pattern_colors)] for i, pattern in enumerate(df['pattern'].unique())}
    colors = [
        pattern_colors[pattern] for pattern in selected_df['pattern'].map({i: int(i) for i in range(3)})
    ]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        selected_embedding[:, 0],
        selected_embedding[:, 1],
        selected_embedding[:, 2],
        c=colors
    )
    plt.show()

    if args['reducer_args']['reducer_high_accuracy_only_bounded'] is not None:
        joblib.dump(selected_reducer, args['reducer_args']['reducer_high_accuracy_only_bounded'])

print("\033[92mFinished plots\033[0m")
