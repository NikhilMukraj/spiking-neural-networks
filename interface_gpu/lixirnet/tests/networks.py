import unittest
import numpy as np
from setup_functions import get_neuron_setup
import lixirnet as ln


e1 = 0
e2 = 1
exc_n1 = 3
exc_n2 = 2
iterations = 1000

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

        # self.assertTrue(np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)).sum()) < 0.1)
        
        # self.assertTrue(np.abs((np.array(network.get_lattice(e2).history) - np.array(gpu_network.get_lattice(e2).history)).sum()) < 0.1)

        for n, (cpu_grid, gpu_grid) in enumerate(zip(network.get_lattice(e1).history, gpu_network.get_lattice(e1).history)):
            for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
                for i, j in zip(cpu_row, gpu_row):
                    if i > -80 and (i != neuron.c and j != neuron.c):
                        self.assertTrue(
                            np.abs(i - j) < 4,
                            f'{n} | {i} != {j}'
                        )

        for n, (cpu_grid, gpu_grid) in enumerate(zip(network.get_lattice(e2).history, gpu_network.get_lattice(e2).history)):
            for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
                for i, j in zip(cpu_row, gpu_row):
                    if i > -80 and (i != neuron.c and j != neuron.c):
                        self.assertTrue(
                            np.abs(i - j) < 4,
                            f'{n} | {i} != {j}'
                        )
    
    def test_isolated_network_chemical_using_from(self):
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

        # self.assertTrue(np.abs((np.array(network.get_lattice(e1).history) - np.array(gpu_network.get_lattice(e1).history)).sum()) < 0.1)
        
        # self.assertTrue(np.abs((np.array(network.get_lattice(e2).history) - np.array(gpu_network.get_lattice(e2).history)).sum()) < 0.1)

        for n, (cpu_grid, gpu_grid) in enumerate(zip(network.get_lattice(e1).history, gpu_network.get_lattice(e1).history)):
            for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
                for i, j in zip(cpu_row, gpu_row):
                    if i > -80 and (i != neuron.c and j != neuron.c):
                        self.assertTrue(
                            np.abs(i - j) < 5,
                            f'{n} | {i} != {j}'
                        )

        for n, (cpu_grid, gpu_grid) in enumerate(zip(network.get_lattice(e2).history, gpu_network.get_lattice(e2).history)):
            for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
                for i, j in zip(cpu_row, gpu_row):
                    if i > -80 and (i != neuron.c and j != neuron.c):
                        self.assertTrue(
                            np.abs(i - j) < 5,
                            f'{n} | {i} != {j}'
                        )

if __name__ == '__main__':
    unittest.main()