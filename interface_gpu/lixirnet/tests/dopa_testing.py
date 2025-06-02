import numpy as np
from setup_functions import get_neuron_setup
import lixirnet as ln


def get_spike_train_setup(init_state):
    def setup_spike_train(pos, neuron):
        x, y = pos
        neuron.step = init_state[x][y]

        return neuron
    
    return setup_spike_train

exc_n1 = 4
e1 = 0
c1 = 1
c2 = 2

exc_neuron = ln.IzhikevichNeuron()
exc_neuron.gap_conductance = 10
exc_neuron.c_m = 25

glu_neuro = ln.BoundedNeurotransmitterKinetics()
exc_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Glutamate : glu_neuro}
# gaba_neuro = ln.BoundedNeurotransmitterKinetics()
# inh_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.GABA : gaba_neuro}
dopa_neuro = ln.BoundedNeurotransmitterKinetics()
dopa_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Dopamine : dopa_neuro}

glu = ln.GlutamateReceptor()
# gaba = ln.GABAReceptor()
dopa = ln.DopamineReceptor()
dopa.s_d1 = 1
dopa.s_d2 = 0
receptors = ln.DopaGluGABA()
receptors.insert(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)
# receptors.insert(ln.DopaGluGABANeurotransmitterType.GABA, gaba)
receptors.insert(ln.DopaGluGABANeurotransmitterType.Dopamine, dopa)

exc_neuron.set_synaptic_neurotransmitters(exc_neurotransmitters)
exc_neuron.set_receptors(receptors)

exc_spike_train = ln.RateSpikeTrain()
exc_spike_train.rate = 100

exc_spike_train.set_synaptic_neurotransmitters(exc_neurotransmitters)

dopa_spike_train = ln.RateSpikeTrain()
dopa_spike_train.rate = 100

dopa_spike_train.set_synaptic_neurotransmitters(dopa_neurotransmitters)

init_state1 = np.random.uniform(exc_neuron.c, exc_neuron.v_th, (exc_n1, exc_n1))

setup_neuron1 = get_neuron_setup(init_state1)

init_spike_train1 = np.random.uniform(0, 100, (exc_n1, exc_n1))
init_spike_train2 = np.random.uniform(0, 100, (exc_n1, exc_n1))

setup_spike_train1 = get_spike_train_setup(init_spike_train1)
setup_spike_train2 = get_spike_train_setup(init_spike_train2)

spike_train_lattice1 = ln.RateSpikeTrainLattice(c1)
spike_train_lattice1.populate(exc_spike_train, exc_n1, exc_n1)
spike_train_lattice1.apply_given_position(setup_spike_train1)
spike_train_lattice1.update_grid_history = True

spike_train_lattice2 = ln.RateSpikeTrainLattice(c2)
spike_train_lattice2.populate(dopa_spike_train, exc_n1, exc_n1)
spike_train_lattice2.apply_given_position(setup_spike_train2)
spike_train_lattice2.update_grid_history = True

lattice1 = ln.IzhikevichNeuronLattice(e1)
lattice1.populate(exc_neuron, exc_n1, exc_n1)
lattice1.apply_given_position(setup_neuron1)
lattice1.connect(lambda x, y: x != y, lambda x, y: 1)
lattice1.update_grid_history = True

network = ln.IzhikevichNeuronNetwork.generate_network([lattice1], [spike_train_lattice1, spike_train_lattice2])
# network = ln.IzhikevichNeuronNetwork.generate_network([lattice1], [spike_train_lattice1])

network.connect(c1, e1, lambda x, y: x == y, lambda x, y: 1)
network.connect(c2, e1, lambda x, y: x == y, lambda x, y: 1)

network.electrical_synapse = False
network.chemical_synapse = True

network.parallel = True
network.set_dt(1)

gpu_network = ln.IzhikevichNeuronNetworkGPU.from_network(network)

network.run_lattices(1000)
gpu_network.run_lattices(1000)

# cpu_history = np.array(network.get_lattice(e1).history)
# gpu_history = np.array(gpu_network.get_lattice(e1).history)

np.save(f'cpu_history_dopa_testing_e1.npy', np.array(network.get_lattice(e1).history))
np.save(f'gpu_history_dopa_testing_e1.npy', np.array(gpu_network.get_lattice(e1).history))

# seems to not engage in excitation as much as it should
# likely some error inherit to both electrical and chemical kernels
# iterate kernel liekly fine
