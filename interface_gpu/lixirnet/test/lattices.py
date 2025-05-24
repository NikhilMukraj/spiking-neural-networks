import unittest
import numpy as np
import lixirnet as ln


e1 = 0
exc_n = 3

def get_setup(init_state):
    def setup_neuron(pos, neuron):
        x, y = pos
        neuron.current_voltage = init_state[x][y]

        return neuron
    
    return setup_neuron

class TestCPUGPUImpl(unittest.TestCase):
    def test_single_lattice_electrical_using_from(self):
        neuron = ln.IzhikevichNeuron()
        neuron.gap_conductance = 10
        neuron.c_m = 25

        init_state = np.random.uniform(neuron.c, neuron.v_th, (exc_n, exc_n))
        setup_neuron = get_setup(init_state)

        lattice = ln.IzhikevichNeuronLattice(e1)
        lattice.populate(neuron, exc_n, exc_n)
        lattice.apply_given_position(setup_neuron)
        lattice.connect(lambda x, y: True, lambda x, y: 5)
        lattice.update_grid_history = True
        lattice.electrical_synapse = True
        lattice.chemical_synapse = False

        gpu_lattice = ln.IzhikevichNeuronLatticeGPU.from_lattice(lattice)

        for n1, n2 in zip(range(exc_n), range(exc_n)):
            for m1, m2 in zip(range(exc_n), range(exc_n)):
                self.assertTrue(
                    np.abs(lattice.get_weight((n1, m1), (n2, m2)) - gpu_lattice.get_weight((n1, m1), (n2, m2))) < 0.1
                )

        for n1, n2 in zip(range(exc_n), range(exc_n)):
            for m1, m2 in zip(range(exc_n), range(exc_n)):
                self.assertTrue(
                    np.abs(lattice.get_neuron(n1, m1).current_voltage - gpu_lattice.get_neuron(n2, m2).current_voltage) < 0.1,
                    f'{lattice.get_neuron(n1, m1).current_voltage} != {gpu_lattice.get_neuron(n2, m2).current_voltage}'
                )

        lattice.run_lattice(1000)
        gpu_lattice.run_lattice(1000)

        np.save(f'cpu_history_{self._testMethodName}.npy', np.array(lattice.history))
        np.save(f'gpu_history_{self._testMethodName}.npy', np.array(gpu_lattice.history))

        for n, (cpu_grid, gpu_grid) in enumerate(zip(lattice.history, gpu_lattice.history)):
            for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
                for i, j in zip(cpu_row, gpu_row):
                    if i > -80:
                        self.assertTrue(
                            np.abs(i - j) < 2,
                            f'{n} | {i} != {j}'
                        )

    def test_single_lattice_electrical(self):
        neuron = ln.IzhikevichNeuron()
        neuron.gap_conductance = 10
        neuron.c_m = 25

        init_state = np.random.uniform(neuron.c, neuron.v_th, (exc_n, exc_n))
        setup_neuron = get_setup(init_state)

        lattice = ln.IzhikevichNeuronLattice(e1)
        lattice.populate(neuron, exc_n, exc_n)
        lattice.apply_given_position(setup_neuron)
        lattice.connect(lambda x, y: True, lambda x, y: 1)
        lattice.update_grid_history = True
        lattice.electrical_synapse = True
        lattice.chemical_synapse = False

        gpu_lattice = ln.IzhikevichNeuronLatticeGPU(e1)
        gpu_lattice.populate(neuron, exc_n, exc_n)
        gpu_lattice.apply_given_position(setup_neuron)
        gpu_lattice.connect(lambda x, y: True, lambda x, y: 1)
        gpu_lattice.update_grid_history = True
        gpu_lattice.electrical_synapse = True
        gpu_lattice.chemical_synapse = False        

        for n1, n2 in zip(range(exc_n), range(exc_n)):
            for m1, m2 in zip(range(exc_n), range(exc_n)):
                self.assertTrue(
                    np.abs(lattice.get_weight((n1, m1), (n2, m2)) - gpu_lattice.get_weight((n1, m1), (n2, m2))) < 0.1
                )

        for n1, n2 in zip(range(exc_n), range(exc_n)):
            for m1, m2 in zip(range(exc_n), range(exc_n)):
                self.assertTrue(
                    np.abs(lattice.get_neuron(n1, m1).current_voltage - gpu_lattice.get_neuron(n2, m2).current_voltage) < 0.1,
                    f'{lattice.get_neuron(n1, m1).current_voltage} != {gpu_lattice.get_neuron(n2, m2).current_voltage}'
                )

        lattice.run_lattice(1000)
        gpu_lattice.run_lattice(1000)

        np.save(f'cpu_history_{self._testMethodName}.npy', np.array(lattice.history))
        np.save(f'gpu_history_{self._testMethodName}.npy', np.array(gpu_lattice.history))

        for n, (cpu_grid, gpu_grid) in enumerate(zip(lattice.history, gpu_lattice.history)):
            for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
                for i, j in zip(cpu_row, gpu_row):
                    if i > -80:
                        self.assertTrue(
                            np.abs(i - j) < 2,
                            f'{n} | {i} != {j}'
                        )
    
    def test_single_lattice_chemical(self):
        neuron = ln.IzhikevichNeuron()

        glu_neuro = ln.BoundedNeurotransmitterKinetics()
        exc_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Glutamate : glu_neuro}
        glu = ln.GlutamateReceptor()
        receptors = ln.DopaGluGABA()
        receptors.insert(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)

        neuron.set_synaptic_neurotransmitters(exc_neurotransmitters)
        neuron.set_receptors(receptors)

        init_state = np.random.uniform(neuron.c, neuron.v_th, (exc_n, exc_n))
        setup_neuron = get_setup(init_state)

        lattice = ln.IzhikevichNeuronLattice(e1)
        lattice.populate(neuron, exc_n, exc_n)
        lattice.apply_given_position(setup_neuron)
        lattice.connect(lambda x, y: True, lambda x, y: 5)
        lattice.update_grid_history = True
        lattice.electrical_synapse = False
        lattice.chemical_synapse = True

        gpu_lattice = ln.IzhikevichNeuronLatticeGPU(e1)
        gpu_lattice.populate(neuron, exc_n, exc_n)
        gpu_lattice.apply_given_position(setup_neuron)
        gpu_lattice.connect(lambda x, y: True, lambda x, y: 5)
        gpu_lattice.update_grid_history = True
        gpu_lattice.electrical_synapse = True
        gpu_lattice.chemical_synapse = False

        for n1, n2 in zip(range(exc_n), range(exc_n)):
            for m1, m2 in zip(range(exc_n), range(exc_n)):
                self.assertTrue(
                    np.abs(lattice.get_weight((n1, m1), (n2, m2)) - gpu_lattice.get_weight((n1, m1), (n2, m2))) < 0.1
                )

        for n1, n2 in zip(range(exc_n), range(exc_n)):
            for m1, m2 in zip(range(exc_n), range(exc_n)):
                self.assertTrue(
                    np.abs(lattice.get_neuron(n1, m1).current_voltage - gpu_lattice.get_neuron(n2, m2).current_voltage) < 0.1,
                    f'{lattice.get_neuron(n1, m1).current_voltage} != {gpu_lattice.get_neuron(n2, m2).current_voltage}'
                )

        lattice.run_lattice(1000)
        gpu_lattice.run_lattice(1000)

        np.save(f'cpu_history_{self._testMethodName}.npy', np.array(lattice.history))
        np.save(f'gpu_history_{self._testMethodName}.npy', np.array(gpu_lattice.history))

        for n, (cpu_grid, gpu_grid) in enumerate(zip(lattice.history, gpu_lattice.history)):
            for cpu_row, gpu_row in zip(cpu_grid, gpu_grid):
                for i, j in zip(cpu_row, gpu_row):
                    if i > -80:
                        self.assertTrue(
                            np.abs(i - j) < 2,
                            f'{n} | {i} != {j}'
                        )

if __name__ == '__main__':
    unittest.main()
