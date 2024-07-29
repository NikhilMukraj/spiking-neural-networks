import lixirnet as ln
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numpy as np
import pandas as pd


sns.set_theme(style='darkgrid')

def connection_conditional(x, y):
    distance = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    return bool(distance <= 2 and np.random.rand() <= 0.8 and x != y)

def randomize_neuron(neuron):
    neuron.current_voltage = np.random.uniform(-65, 30)
    return neuron

n = 5

lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), n, n)
lattice.apply(randomize_neuron)
lattice.connect(connection_conditional)

lattice.update_grid_history = True
lattice.reset_timing()
lattice.reset_history()

lattice.run_lattice(10000)

def find_peaks_above_threshold(series, threshold):
    peaks, _ = scipy.signal.find_peaks(np.array(series))
    filtered_peaks = [index for index in peaks if series[index] > threshold]
    
    return filtered_peaks

peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]
selection_min, selection_max = 0, 2000
filtered_peaks = [[i for i in j if i >= selection_min and i <= selection_max] for j in peaks]

for n in range(len(filtered_peaks)):
    plt.scatter(filtered_peaks[n], [n for i in range(len(filtered_peaks[n]))], color='black')
    
plt.show()
