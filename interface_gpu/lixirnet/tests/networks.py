import unittest
import scipy
import numpy as np
from setup_functions import get_neuron_setup
import lixirnet as ln


e1 = 0
e2 = 1
i1 = 2
i2 = 3
c1 = 4
c2 = 5
exc_n1 = 3
exc_n2 = 2
iterations = 1000

def get_spike_train_setup(init_state):
    def setup_spike_train(pos, neuron):
        x, y = pos
        neuron.step = init_state[x][y]

        return neuron
    
    return setup_spike_train

def find_peaks_above_threshold(series, threshold):
    peaks, _ = scipy.signal.find_peaks(np.array(series))
    filtered_peaks = [index for index in peaks if series[index] > threshold]
    
    return filtered_peaks

class TestCPUGPUImpl(unittest.TestCase):
    def test_network_electrical_using_from(self):
        neuron = ln.IzhikevichNeuron()
        neuron.gap_conductance = 10
        neuron.c_m = 25

        init_state1 = np.random.uniform(neuron.c, neuron.v_th, (exc_n1, exc_n1))
        init_state2 = np.random.uniform(neuron.c, neuron.v_th, (exc_n2, exc_n2))

        setup_neuron1 = get_neuron_setup(init_state1)
        setup_neuron2 = get_neuron_setup(init_state2)

        lattice1 = ln.IzhikevichNeuronLattice(e1)
        lattice1.populate(neuron, exc_n1, exc_n1)
        lattice1.apply_given_position(setup_neuron1)
        lattice1.connect(lambda x, y: x != y, lambda x, y: 5)
        lattice1.update_grid_history = True

        lattice2 = ln.IzhikevichNeuronLattice(e2)
        lattice2.populate(neuron, exc_n2, exc_n2)
        lattice2.apply_given_position(setup_neuron2)
        lattice2.connect(lambda x, y: x != y, lambda x, y: 3)
        lattice2.update_grid_history = True

        network = ln.IzhikevichNeuronNetwork.generate_network([lattice1, lattice2], [])
        network.connect(e1, e2, lambda x, y: x == y, lambda x, y: 5)
        network.connect(e2, e1, lambda x, y: x == y, lambda x, y: 3)
        network.electrical_synapse = True
        network.chemical_synapse = False
        gpu_network = ln.IzhikevichNeuronNetworkGPU.from_network(network)

        for n1, n2 in zip(range(exc_n1), range(exc_n1)):
            for m1, m2 in zip(range(exc_n1), range(exc_n1)):
                self.assertTrue(
                    np.abs(network.get_lattice(e1).get_weight((n1, m1), (n2, m2)) - gpu_network.get_lattice(e1).get_weight((n1, m1), (n2, m2))) < 0.1
                )

        for n1, n2 in zip(range(exc_n1), range(exc_n1)):
            for m1, m2 in zip(range(exc_n1), range(exc_n1)):
                self.assertTrue(
                    np.abs(network.get_lattice(e1).get_neuron(n1, m1).current_voltage - gpu_network.get_lattice(e1).get_neuron(n2, m2).current_voltage) < 0.1,
                    f'{network.get_lattice(e1).get_neuron(n1, m1).current_voltage} != {gpu_network.get_lattice(e1).get_neuron(n2, m2).current_voltage}'
                )

        for n1, n2 in zip(range(exc_n2), range(exc_n2)):
            for m1, m2 in zip(range(exc_n2), range(exc_n2)):
                self.assertTrue(
                    np.abs(network.get_lattice(e2).get_weight((n1, m1), (n2, m2)) - gpu_network.get_lattice(e2).get_weight((n1, m1), (n2, m2))) < 0.1
                )

        for n1, n2 in zip(range(exc_n2), range(exc_n2)):
            for m1, m2 in zip(range(exc_n2), range(exc_n2)):
                self.assertTrue(
                    np.abs(network.get_lattice(e2).get_neuron(n1, m1).current_voltage - gpu_network.get_lattice(e2).get_neuron(n2, m2).current_voltage) < 0.1,
                    f'{network.get_lattice(e2).get_neuron(n1, m1).current_voltage} != {gpu_network.get_lattice(e2).get_neuron(n2, m2).current_voltage}'
                )

        self.assertTrue(network.connecting_weights, gpu_network.connecting_weights)

        network.run_lattices(iterations)
        gpu_network.run_lattices(iterations)

        np.save(f'cpu_history_{self._testMethodName}_e1.npy', np.array(network.get_lattice(e1).history))
        np.save(f'gpu_history_{self._testMethodName}_e1.npy', np.array(gpu_network.get_lattice(e1).history))

        np.save(f'cpu_history_{self._testMethodName}_e2.npy', np.array(network.get_lattice(e2).history))
        np.save(f'gpu_history_{self._testMethodName}_e2.npy', np.array(gpu_network.get_lattice(e2).history))

        self.assertTrue(np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)).sum()) < 0.1)
        
        self.assertTrue(np.abs((np.array(network.get_lattice(e2).history) - np.array(gpu_network.get_lattice(e2).history)).sum()) < 0.1)

        # for n, (cpu_grid, gpu_grid) in enumerate(zip(network.get_lattice(e1).history, gpu_network.get_lattice(e1).history)):
        #     for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
        #         for i, j in zip(cpu_row, gpu_row):
        #             if i > -80 and (i != neuron.c and j != neuron.c):
        #                 self.assertTrue(
        #                     np.abs(i - j) < 4,
        #                     f'{n} | {i} != {j}'
        #                 )

        # for n, (cpu_grid, gpu_grid) in enumerate(zip(network.get_lattice(e2).history, gpu_network.get_lattice(e2).history)):
        #     for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
        #         for i, j in zip(cpu_row, gpu_row):
        #             if i > -80 and (i != neuron.c and j != neuron.c):
        #                 self.assertTrue(
        #                     np.abs(i - j) < 4,
        #                     f'{n} | {i} != {j}'
        #                 )
    
    def test_network_chemical_using_from(self):
        neuron = ln.IzhikevichNeuron()
        neuron.c_m = 25

        glu_neuro = ln.BoundedNeurotransmitterKinetics()
        exc_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Glutamate : glu_neuro}
        glu = ln.GlutamateReceptor()
        receptors = ln.DopaGluGABA()
        receptors.insert(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)

        neuron.set_synaptic_neurotransmitters(exc_neurotransmitters)
        neuron.set_receptors(receptors)

        init_state1 = np.random.uniform(neuron.c, neuron.v_th, (exc_n1, exc_n1))
        init_state2 = np.random.uniform(neuron.c, neuron.v_th, (exc_n2, exc_n2))

        setup_neuron1 = get_neuron_setup(init_state1)
        setup_neuron2 = get_neuron_setup(init_state2)

        lattice1 = ln.IzhikevichNeuronLattice(e1)
        lattice1.populate(neuron, exc_n1, exc_n1)
        lattice1.apply_given_position(setup_neuron1)
        lattice1.connect(lambda x, y: x != y, lambda x, y: 2)
        lattice1.update_grid_history = True

        lattice2 = ln.IzhikevichNeuronLattice(e2)
        lattice2.populate(neuron, exc_n2, exc_n2)
        lattice2.apply_given_position(setup_neuron2)
        lattice2.connect(lambda x, y: x != y, lambda x, y: 0.5)
        lattice2.update_grid_history = True

        network = ln.IzhikevichNeuronNetwork.generate_network([lattice1, lattice2], [])
        network.connect(e1, e2, lambda x, y: x == y, lambda x, y: 1)
        network.connect(e2, e1, lambda x, y: x == y, lambda x, y: 1)
        network.electrical_synapse = False
        network.chemical_synapse = True
        gpu_network = ln.IzhikevichNeuronNetworkGPU.from_network(network)

        for n1, n2 in zip(range(exc_n1), range(exc_n1)):
            for m1, m2 in zip(range(exc_n1), range(exc_n1)):
                self.assertTrue(
                    np.abs(network.get_lattice(e1).get_weight((n1, m1), (n2, m2)) - gpu_network.get_lattice(e1).get_weight((n1, m1), (n2, m2))) < 0.1
                )

        for n1, n2 in zip(range(exc_n1), range(exc_n1)):
            for m1, m2 in zip(range(exc_n1), range(exc_n1)):
                self.assertTrue(
                    np.abs(network.get_lattice(e1).get_neuron(n1, m1).current_voltage - gpu_network.get_lattice(e1).get_neuron(n2, m2).current_voltage) < 0.1,
                    f'{network.get_lattice(e1).get_neuron(n1, m1).current_voltage} != {gpu_network.get_lattice(e1).get_neuron(n2, m2).current_voltage}'
                )

        for n1, n2 in zip(range(exc_n2), range(exc_n2)):
            for m1, m2 in zip(range(exc_n2), range(exc_n2)):
                self.assertTrue(
                    np.abs(network.get_lattice(e2).get_weight((n1, m1), (n2, m2)) - gpu_network.get_lattice(e2).get_weight((n1, m1), (n2, m2))) < 0.1
                )

        for n1, n2 in zip(range(exc_n2), range(exc_n2)):
            for m1, m2 in zip(range(exc_n2), range(exc_n2)):
                self.assertTrue(
                    np.abs(network.get_lattice(e2).get_neuron(n1, m1).current_voltage - gpu_network.get_lattice(e2).get_neuron(n2, m2).current_voltage) < 0.1,
                    f'{network.get_lattice(e2).get_neuron(n1, m1).current_voltage} != {gpu_network.get_lattice(e2).get_neuron(n2, m2).current_voltage}'
                )

        self.assertTrue(network.connecting_weights, gpu_network.connecting_weights)

        network.run_lattices(iterations)
        gpu_network.run_lattices(iterations)

        np.save(f'cpu_history_{self._testMethodName}_e1.npy', np.array(network.get_lattice(e1).history))
        np.save(f'gpu_history_{self._testMethodName}_e1.npy', np.array(gpu_network.get_lattice(e1).history))

        np.save(f'cpu_history_{self._testMethodName}_e2.npy', np.array(network.get_lattice(e2).history))
        np.save(f'gpu_history_{self._testMethodName}_e2.npy', np.array(gpu_network.get_lattice(e2).history))

        self.assertTrue(np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)).sum()) < 0.1)
        
        self.assertTrue(np.abs((np.array(network.get_lattice(e2).history) - np.array(gpu_network.get_lattice(e2).history)).sum()) < 0.1)

        # for n, (cpu_grid, gpu_grid) in enumerate(zip(network.get_lattice(e1).history, gpu_network.get_lattice(e1).history)):
        #     for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
        #         for i, j in zip(cpu_row, gpu_row):
        #             if i > -80 and (i != neuron.c and j != neuron.c):
        #                 self.assertTrue(
        #                     np.abs(i - j) < 5,
        #                     f'{n} | {i} != {j}'
        #                 )

        # for n, (cpu_grid, gpu_grid) in enumerate(zip(network.get_lattice(e2).history, gpu_network.get_lattice(e2).history)):
        #     for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
        #         for i, j in zip(cpu_row, gpu_row):
        #             if i > -80 and (i != neuron.c and j != neuron.c):
        #                 self.assertTrue(
        #                     np.abs(i - j) < 5,
        #                     f'{n} | {i} != {j}'
        #                 )

    def test_network_electrical_with_spike_trains(self):
        neuron = ln.IzhikevichNeuron()
        neuron.gap_conductance = 10
        neuron.c_m = 25

        init_state1 = np.random.uniform(neuron.c, neuron.v_th, (exc_n1, exc_n1))
        init_state2 = np.random.uniform(neuron.c, neuron.v_th, (exc_n2, exc_n2))

        setup_neuron1 = get_neuron_setup(init_state1)
        setup_neuron2 = get_neuron_setup(init_state2)

        spike_train = ln.RateSpikeTrain()
        spike_train.rate = 100

        init_spike_train = np.random.uniform(0, 100, (exc_n1, exc_n1))

        setup_spike_train = get_spike_train_setup(init_spike_train)

        spike_train_lattice = ln.RateSpikeTrainLattice(c1)
        spike_train_lattice.populate(spike_train, exc_n1, exc_n1)
        spike_train_lattice.apply_given_position(setup_spike_train)
        spike_train_lattice.update_grid_history = True

        lattice1 = ln.IzhikevichNeuronLattice(e1)
        lattice1.populate(neuron, exc_n1, exc_n1)
        lattice1.apply_given_position(setup_neuron1)
        lattice1.connect(lambda x, y: x != y, lambda x, y: 5)
        lattice1.update_grid_history = True

        lattice2 = ln.IzhikevichNeuronLattice(i1)
        lattice2.populate(neuron, exc_n2, exc_n2)
        lattice2.apply_given_position(setup_neuron2)
        lattice2.connect(lambda x, y: x != y, lambda x, y: 3)
        lattice2.update_grid_history = True

        network = ln.IzhikevichNeuronNetwork.generate_network([lattice1, lattice2], [spike_train_lattice])
        network.connect(e1, i1, lambda x, y: x == y, lambda x, y: 5)
        network.connect(i1, e1, lambda x, y: x == y, lambda x, y: -3)
        network.connect(c1, e1, lambda x, y: x == y, lambda x, y: 5)
        network.electrical_synapse = True
        network.chemical_synapse = False
        gpu_network = ln.IzhikevichNeuronNetworkGPU.from_network(network)

        self.assertTrue(network.connecting_weights, gpu_network.connecting_weights)

        network.run_lattices(iterations)
        gpu_network.run_lattices(iterations)

        np.save(f'cpu_history_{self._testMethodName}_e1.npy', np.array(network.get_lattice(e1).history))
        np.save(f'gpu_history_{self._testMethodName}_e1.npy', np.array(gpu_network.get_lattice(e1).history))

        np.save(f'cpu_history_{self._testMethodName}_i1.npy', np.array(network.get_lattice(i1).history))
        np.save(f'gpu_history_{self._testMethodName}_i1.npy', np.array(gpu_network.get_lattice(i1).history))

        self.assertTrue(np.abs((np.array(network.get_spike_train_lattice(c1).history) - np.array(gpu_network.get_spike_train_lattice(c1).history)).sum()) < 0.1)

        # self.assertTrue(np.abs((np.array(network.get_lattice(i1).history) - np.array(gpu_network.get_lattice(i1).history)).sum()) < 0.5)
        diff_hist = np.abs((np.array(network.get_lattice(i1).history) - np.array(gpu_network.get_lattice(i1).history)))
        self.assertTrue(np.sum((np.abs(diff_hist) < 0.1)) / np.prod(diff_hist.shape) > 0.95)

        # self.assertTrue(np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)).sum()) < 0.5)
        diff_hist = np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)))
        self.assertTrue(np.sum((np.abs(diff_hist) < 0.1)) / np.prod(diff_hist.shape) > 0.95)

        
    def test_network_chemical_with_spike_trains(self):
        neuron = ln.IzhikevichNeuron()
        neuron.gap_conductance = 5
        neuron.c_m = 25

        glu_neuro = ln.BoundedNeurotransmitterKinetics()
        exc_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Glutamate : glu_neuro}
        glu = ln.GlutamateReceptor()
        receptors = ln.DopaGluGABA()
        receptors.insert(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)

        neuron.set_synaptic_neurotransmitters(exc_neurotransmitters)
        neuron.set_receptors(receptors)

        init_state1 = np.random.uniform(neuron.c, neuron.v_th, (exc_n1, exc_n1))
        init_state2 = np.random.uniform(neuron.c, neuron.v_th, (exc_n2, exc_n2))

        setup_neuron1 = get_neuron_setup(init_state1)
        setup_neuron2 = get_neuron_setup(init_state2)

        spike_train = ln.RateSpikeTrain()
        spike_train.rate = 100

        spike_train.set_synaptic_neurotransmitters(exc_neurotransmitters)

        init_spike_train = np.random.uniform(0, 100, (exc_n1, exc_n1))

        setup_spike_train = get_spike_train_setup(init_spike_train)

        spike_train_lattice = ln.RateSpikeTrainLattice(c1)
        spike_train_lattice.populate(spike_train, exc_n1, exc_n1)
        spike_train_lattice.apply_given_position(setup_spike_train)
        spike_train_lattice.update_grid_history = True

        lattice1 = ln.IzhikevichNeuronLattice(e1)
        lattice1.populate(neuron, exc_n1, exc_n1)
        lattice1.apply_given_position(setup_neuron1)
        lattice1.connect(lambda x, y: x != y, lambda x, y: 5)
        lattice1.update_grid_history = True

        network = ln.IzhikevichNeuronNetwork.generate_network([lattice1], [spike_train_lattice])
        network.connect(c1, e1, lambda x, y: x == y, lambda x, y: 5)
        network.electrical_synapse = False
        network.chemical_synapse = True
        gpu_network = ln.IzhikevichNeuronNetworkGPU.from_network(network)

        self.assertTrue(network.connecting_weights, gpu_network.connecting_weights)

        network.run_lattices(iterations)
        gpu_network.run_lattices(iterations)

        np.save(f'cpu_history_{self._testMethodName}_e1.npy', np.array(network.get_lattice(e1).history))
        np.save(f'gpu_history_{self._testMethodName}_e1.npy', np.array(gpu_network.get_lattice(e1).history))
       
        self.assertTrue(np.abs((np.array(network.get_spike_train_lattice(c1).history) - np.array(gpu_network.get_spike_train_lattice(c1).history)).sum()) < 0.1)

        self.assertTrue(np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)).sum()) < 0.1)

    def test_network_chemical_multiple_lattices_and_spike_train(self):
        neuron = ln.IzhikevichNeuron()
        neuron.gap_conductance = 10
        neuron.c_m = 25

        glu_neuro = ln.BoundedNeurotransmitterKinetics()
        exc_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Glutamate : glu_neuro}
        glu = ln.GlutamateReceptor()
        receptors = ln.DopaGluGABA()
        receptors.insert(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)

        neuron.set_synaptic_neurotransmitters(exc_neurotransmitters)
        neuron.set_receptors(receptors)

        init_state1 = np.random.uniform(neuron.c, neuron.v_th, (exc_n1, exc_n1))
        init_state2 = np.random.uniform(neuron.c, neuron.v_th, (exc_n2, exc_n2))

        setup_neuron1 = get_neuron_setup(init_state1)
        setup_neuron2 = get_neuron_setup(init_state2)

        spike_train = ln.RateSpikeTrain()
        spike_train.rate = 100

        spike_train.set_synaptic_neurotransmitters(exc_neurotransmitters)

        init_spike_train = np.random.uniform(0, 100, (exc_n1, exc_n1))

        setup_spike_train = get_spike_train_setup(init_spike_train)

        spike_train_lattice = ln.RateSpikeTrainLattice(c1)
        spike_train_lattice.populate(spike_train, exc_n1, exc_n1)
        spike_train_lattice.apply_given_position(setup_spike_train)
        spike_train_lattice.update_grid_history = True

        lattice1 = ln.IzhikevichNeuronLattice(e1)
        lattice1.populate(neuron, exc_n1, exc_n1)
        lattice1.apply_given_position(setup_neuron1)
        lattice1.connect(lambda x, y: x != y, lambda x, y: 5)
        lattice1.update_grid_history = True

        lattice2 = ln.IzhikevichNeuronLattice(e2)
        lattice2.populate(neuron, exc_n2, exc_n2)
        lattice2.apply_given_position(setup_neuron2)
        lattice2.connect(lambda x, y: x != y, lambda x, y: 3)
        lattice2.update_grid_history = True

        network = ln.IzhikevichNeuronNetwork.generate_network([lattice1, lattice2], [spike_train_lattice])
        network.connect(e1, e2, lambda x, y: x == y, lambda x, y: 5)
        network.connect(e2, e1, lambda x, y: x == y, lambda x, y: 3)
        network.connect(c1, e1, lambda x, y: x == y, lambda x, y: 5)
        network.electrical_synapse = False
        network.chemical_synapse = True
        gpu_network = ln.IzhikevichNeuronNetworkGPU.from_network(network)

        self.assertTrue(network.connecting_weights, gpu_network.connecting_weights)

        network.run_lattices(iterations)
        gpu_network.run_lattices(iterations)

        np.save(f'cpu_history_{self._testMethodName}_e1.npy', np.array(network.get_lattice(e1).history))
        np.save(f'gpu_history_{self._testMethodName}_e1.npy', np.array(gpu_network.get_lattice(e1).history))

        np.save(f'cpu_history_{self._testMethodName}_e2.npy', np.array(network.get_lattice(e2).history))
        np.save(f'gpu_history_{self._testMethodName}_e2.npy', np.array(gpu_network.get_lattice(e2).history))

        self.assertTrue(np.abs((np.array(network.get_spike_train_lattice(c1).history) - np.array(gpu_network.get_spike_train_lattice(c1).history)).sum()) < 0.1)

        self.assertTrue(np.abs((np.array(network.get_lattice(e2).history) - np.array(gpu_network.get_lattice(e2).history)).sum()) < 0.1)

        self.assertTrue(np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)).sum()) < 0.1)

    def test_network_multiple_spike_trains_and_multiple_lattices(self):
        neuron = ln.IzhikevichNeuron()
        neuron.gap_conductance = 10
        neuron.c_m = 25

        glu_neuro = ln.BoundedNeurotransmitterKinetics()
        exc_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Glutamate : glu_neuro}
        glu = ln.GlutamateReceptor()
        receptors = ln.DopaGluGABA()
        receptors.insert(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)

        neuron.set_synaptic_neurotransmitters(exc_neurotransmitters)
        neuron.set_receptors(receptors)

        init_state1 = np.random.uniform(neuron.c, neuron.v_th, (exc_n1, exc_n1))
        init_state2 = np.random.uniform(neuron.c, neuron.v_th, (exc_n2, exc_n2))

        setup_neuron1 = get_neuron_setup(init_state1)
        setup_neuron2 = get_neuron_setup(init_state2)

        spike_train = ln.RateSpikeTrain()
        spike_train.rate = 100

        spike_train.set_synaptic_neurotransmitters(exc_neurotransmitters)

        init_spike_train1 = np.random.uniform(0, 100, (exc_n1, exc_n1))
        init_spike_train2 = np.random.uniform(0, 100, (exc_n1, exc_n1))

        setup_spike_train1 = get_spike_train_setup(init_spike_train1)
        setup_spike_train2 = get_spike_train_setup(init_spike_train2)

        spike_train_lattice1 = ln.RateSpikeTrainLattice(c1)
        spike_train_lattice1.populate(spike_train, exc_n1, exc_n1)
        spike_train_lattice1.apply_given_position(setup_spike_train1)
        spike_train_lattice1.update_grid_history = True

        spike_train_lattice2 = ln.RateSpikeTrainLattice(c2)
        spike_train_lattice2.populate(spike_train, exc_n1, exc_n1)
        spike_train_lattice2.apply_given_position(setup_spike_train2)
        spike_train_lattice2.update_grid_history = True

        lattice1 = ln.IzhikevichNeuronLattice(e1)
        lattice1.populate(neuron, exc_n1, exc_n1)
        lattice1.apply_given_position(setup_neuron1)
        lattice1.connect(lambda x, y: x != y, lambda x, y: 5)
        lattice1.update_grid_history = True

        lattice2 = ln.IzhikevichNeuronLattice(e2)
        lattice2.populate(neuron, exc_n2, exc_n2)
        lattice2.apply_given_position(setup_neuron2)
        lattice2.connect(lambda x, y: x != y, lambda x, y: 3)
        lattice2.update_grid_history = True

        network = ln.IzhikevichNeuronNetwork.generate_network([lattice1, lattice2], [spike_train_lattice1, spike_train_lattice2])
        network.connect(e1, e2, lambda x, y: x == y, lambda x, y: 5)
        network.connect(e2, e1, lambda x, y: x == y, lambda x, y: 3)
        network.connect(c1, e1, lambda x, y: x == y, lambda x, y: 5)
        network.connect(c2, e1, lambda x, y: x == y, lambda x, y: 4)
        network.electrical_synapse = False
        network.chemical_synapse = True
        gpu_network = ln.IzhikevichNeuronNetworkGPU.from_network(network)

        self.assertTrue(network.connecting_weights, gpu_network.connecting_weights)

        network.run_lattices(iterations)
        gpu_network.run_lattices(iterations)

        np.save(f'cpu_history_{self._testMethodName}_e1.npy', np.array(network.get_lattice(e1).history))
        np.save(f'gpu_history_{self._testMethodName}_e1.npy', np.array(gpu_network.get_lattice(e1).history))

        np.save(f'cpu_history_{self._testMethodName}_e2.npy', np.array(network.get_lattice(e2).history))
        np.save(f'gpu_history_{self._testMethodName}_e2.npy', np.array(gpu_network.get_lattice(e2).history))

        self.assertTrue(np.abs((np.array(network.get_spike_train_lattice(c1).history) - np.array(gpu_network.get_spike_train_lattice(c1).history)).sum()) < 0.1)

        self.assertTrue(np.abs((np.array(network.get_spike_train_lattice(c2).history) - np.array(gpu_network.get_spike_train_lattice(c2).history)).sum()) < 0.1)

        # self.assertTrue(np.abs((np.array(network.get_lattice(e2).history) - np.array(gpu_network.get_lattice(e2).history)).sum()) < 0.1)
        diff_hist = np.abs((np.array(network.get_lattice(e2).history) - np.array(gpu_network.get_lattice(e2).history)))
        self.assertTrue(np.sum((np.abs(diff_hist) < 0.1)) / np.prod(diff_hist.shape) > 0.9)

        # self.assertTrue(np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)).sum()) < 0.1)
        diff_hist = np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)))
        self.assertTrue(np.sum((np.abs(diff_hist) < 0.1)) / np.prod(diff_hist.shape) > 0.9)

    def test_network_chemical_various_neurotransmitters(self):
        exc_neuron = ln.IzhikevichNeuron()
        exc_neuron.gap_conductance = 10
        exc_neuron.c_m = 25

        inh_neuron = ln.IzhikevichNeuron()
        inh_neuron.gap_conductance = 10
        inh_neuron.c_m = 25

        glu_neuro = ln.BoundedNeurotransmitterKinetics()
        exc_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Glutamate : glu_neuro}
        gaba_neuro = ln.BoundedNeurotransmitterKinetics()
        inh_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.GABA : gaba_neuro}
        dopa_neuro = ln.BoundedNeurotransmitterKinetics()
        dopa_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Dopamine : dopa_neuro}

        glu = ln.GlutamateReceptor()
        gaba = ln.GABAReceptor()
        dopa = ln.DopamineReceptor()
        dopa.s_d1 = 1
        dopa.s_d2 = 0
        receptors = ln.DopaGluGABA()
        receptors.insert(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)
        receptors.insert(ln.DopaGluGABANeurotransmitterType.GABA, gaba)
        receptors.insert(ln.DopaGluGABANeurotransmitterType.Dopamine, dopa)

        exc_neuron.set_synaptic_neurotransmitters(exc_neurotransmitters)
        exc_neuron.set_receptors(receptors)

        inh_neuron.set_synaptic_neurotransmitters(inh_neurotransmitters)
        inh_neuron.set_receptors(receptors)

        exc_spike_train = ln.RateSpikeTrain()
        exc_spike_train.rate = 100

        exc_spike_train.set_synaptic_neurotransmitters(exc_neurotransmitters)

        dopa_spike_train = ln.RateSpikeTrain()
        dopa_spike_train.rate = 100

        dopa_spike_train.set_synaptic_neurotransmitters(dopa_neurotransmitters)

        init_state1 = np.random.uniform(exc_neuron.c, exc_neuron.v_th, (exc_n1, exc_n1))
        init_state2 = np.random.uniform(inh_neuron.c, inh_neuron.v_th, (exc_n2, exc_n2))

        setup_neuron1 = get_neuron_setup(init_state1)
        setup_neuron2 = get_neuron_setup(init_state2)

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

        lattice2 = ln.IzhikevichNeuronLattice(i1)
        lattice2.populate(inh_neuron, exc_n2, exc_n2)
        lattice2.apply_given_position(setup_neuron2)
        lattice2.connect(lambda x, y: x != y, lambda x, y: 0.5)
        lattice2.update_grid_history = True

        network = ln.IzhikevichNeuronNetwork.generate_network([lattice1, lattice2], [spike_train_lattice1, spike_train_lattice2])
        network.connect(e1, i1, lambda x, y: x == y, lambda x, y: 2)
        network.connect(i1, e1, lambda x, y: x == y, lambda x, y: 1)
        network.connect(c1, e1, lambda x, y: x == y, lambda x, y: 3)
        network.connect(c2, e1, lambda x, y: x == y, lambda x, y: 1)
        network.electrical_synapse = False
        network.chemical_synapse = True
        gpu_network = ln.IzhikevichNeuronNetworkGPU.from_network(network)

        self.assertTrue(network.connecting_weights, gpu_network.connecting_weights)

        network.run_lattices(iterations)
        gpu_network.run_lattices(iterations)

        np.save(f'cpu_history_{self._testMethodName}_e1.npy', np.array(network.get_lattice(e1).history))
        np.save(f'gpu_history_{self._testMethodName}_e1.npy', np.array(gpu_network.get_lattice(e1).history))

        np.save(f'cpu_history_{self._testMethodName}_i1.npy', np.array(network.get_lattice(i1).history))
        np.save(f'gpu_history_{self._testMethodName}_i1.npy', np.array(gpu_network.get_lattice(i1).history))

        self.assertTrue(np.abs((np.array(network.get_spike_train_lattice(c1).history) - np.array(gpu_network.get_spike_train_lattice(c1).history)).sum()) < 0.1)

        self.assertTrue(np.abs((np.array(network.get_spike_train_lattice(c2).history) - np.array(gpu_network.get_spike_train_lattice(c2).history)).sum()) < 0.1)

        # self.assertTrue(np.abs((np.array(network.get_lattice(i1).history) - np.array(gpu_network.get_lattice(i1).history)).sum()) < 0.1)
        # diff_hist = np.abs((np.array(network.get_lattice(i1).history) - np.array(gpu_network.get_lattice(i1).history)))
        # self.assertTrue(np.sum((np.abs(diff_hist) < 25)) / np.prod(diff_hist.shape) > 0.9)

        cpu_history = np.array(network.get_lattice(i1).history)
        gpu_history = np.array(gpu_network.get_lattice(i1).history)

        for n in range(exc_n2):
            for m in range(exc_n2):
                self.assertTrue(all(abs(i - j) < 100 for i, j in zip(find_peaks_above_threshold(cpu_history[:, n, m], 20), find_peaks_above_threshold(gpu_history[:, n, m], 20))))

        # self.assertTrue(np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)).sum()) < 0.1)
        # diff_hist = np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)))
        # self.assertTrue(np.sum((np.abs(diff_hist) < 25)) / np.prod(diff_hist.shape) > 0.9)

        cpu_history = np.array(network.get_lattice(e1).history)
        gpu_history = np.array(gpu_network.get_lattice(e1).history)

        for n in range(exc_n1):
            for m in range(exc_n1):
                self.assertTrue(all(abs(i - j) < 100 for i, j in zip(find_peaks_above_threshold(cpu_history[:, n, m], 20), find_peaks_above_threshold(gpu_history[:, n, m], 20))))

if __name__ == '__main__':
    unittest.main()