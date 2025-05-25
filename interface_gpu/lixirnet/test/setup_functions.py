def get_neuron_setup(init_state):
    def setup_neuron(pos, neuron):
        x, y = pos
        neuron.current_voltage = init_state[x][y]

        return neuron
    
    return setup_neuron
    