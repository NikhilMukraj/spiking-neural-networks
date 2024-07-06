import lixirnet as ln
import matplotlib.pyplot as plt
import seaborn as sns


izhikevich_neuron = ln.IFCell('izhikevich')

history = {'voltage' : [], 'spikes': []}
for i in range(1000):
    spike = izhikevich_neuron.iterate_and_return_spike(i=30, bayesian=False)
    history['voltage'].append(izhikevich_neuron.current_voltage)
    history['spike'].append(spike)

spike_times = [(i, history['voltage'][n]) for n, i in enumerate(history['spike'])]

sns.set_theme(style='darkgrid')

plt.plot(history['voltage'])
plt.scatter(*zip(*spike_times))
plt.show()
