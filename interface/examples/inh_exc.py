import lixirnet as ln
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


sns.set_theme(style='darkgrid')

def randomize_neuron(neuron):
    neuron.current_voltage = np.random.uniform(-65, 30)
    return neuron

inh_lattice = ln.IzhikevichLattice(0)
inh_lattice.populate(ln.IzhikevichNeuron(), 5, 5)
inh_lattice.connect(lambda x, y: x != y, lambda x, y: -1)
inh_lattice.update_grid_history = True
inh_lattice.apply(randomize_neuron)

exc_lattice = ln.IzhikevichLattice(1)
exc_lattice.populate(ln.IzhikevichNeuron(), 10, 10)
exc_lattice.connect(lambda x, y: x != y)
exc_lattice.update_grid_history = True
exc_lattice.apply(randomize_neuron)

network = ln.IzhikevichNetwork()
network.add_lattice(exc_lattice)
network.add_lattice(inh_lattice)
network.connect(0, 1, lambda x, y: True, lambda x, y: -1)
network.connect(1, 0, lambda x, y: True)

for _ in tqdm(range(10_000)):
    network.run_lattices(1)

inh_history = network.get_lattice(0).history
exc_history = network.get_lattice(1).history

inh_history = np.array([np.array(i).mean() for i in inh_history])
exc_history = np.array([np.array(i).mean() for i in exc_history])

plt.plot(inh_history, label='inh')
plt.plot(exc_history, label='exc')
plt.legend()
plt.show()
