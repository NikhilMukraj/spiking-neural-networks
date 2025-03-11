# regular disturbing spike train group to liquid
# second spike train group to liquid or inh
# option to turn on d1 or d2
# option to connect second spike train group to either liquid or inh
# vary parameters to determine effect of stability based on neuromodulation


e1 = 0
i1 = 1
c1 = 2
c2 = 3

# add option to change freq of dopamine spike train

for current_state in tqdm(all_states):
    for trial in range(parsed_toml['simulation_parameters']['trials']):
        w = generate_liquid_weights(
            num, connectivity=current_state['connectivity'], scalar=current_state['internal_scalar']
        )

        if not parsed_toml['simulation_parameters']['exc_only']:
            w_inh = generate_liquid_weights(
                inh_num, connectivity=current_state['inh_connectivity'], scalar=current_state['inh_internal_scalar']
            )

        start_firing = generate_start_firing(current_state['cue_firing_rate'])

        glu_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['glu_clearance'])
        exc_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
        exc_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.Glutamate, glu_neuro)

        gaba_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['gabaa_clearance'])
        inh_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
        inh_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.GABA, gaba_neuro)

        dopa_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['dopamine_clearance'])
        dopamine_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
        dopamine_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.Dopamine, dopa_neuro)

        glu = ln.GlutamateReceptor()
        
        glu.ampa_g = current_state['nmda_g']
        glu.nmda_g = current_state['ampa_g']

        gaba = ln.GABAReceptor()
        gaba.g = current_state['gabaa_g']

        dopamine_rs = ln.DopamineReceptor()

        dopamine_rs.d1_enabled = parsed_toml['simulation_parameters']['d1']
        dopamine_rs.d2_enabled = parsed_toml['simulation_parameters']['d2']

        dopamine_rs.s_d1 = current_state['s_d1']
        dopamine_rs.s_d2 = current_state['s_d2']

        receptors = ln.DopaGluGABAReceptors()

        receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)
        receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.GABA, gaba)
        receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.Dopamine, dopamine_rs)

        exc_neuron = ln.DopaIzhikevichNeuron()
        exc_neuron.set_neurotransmitters(exc_neurotransmitters)
        exc_neuron.set_receptors(receptors)

        inh_neuron = ln.DopaIzhikevichNeuron()
        inh_neuron.set_neurotransmitters(inh_neurotransmitters)
        inh_neuron.set_receptors(receptors)

        poisson_neuron = ln.DopaPoissonNeuron()
        poisson_neuron.set_neurotransmitters(exc_neurotransmitters)

        exc_lattice = ln.DopaIzhikevichLattice(e1)
        exc_lattice.populate(exc_neuron, exc_n, exc_n)
        exc_lattice.apply(setup_neuron)
        position_to_index = exc_lattice.position_to_index
        exc_lattice.connect(
            lambda x, y: bool(float(w[position_to_index[x]][position_to_index[y]]) != 0), 
            lambda x, y: float(w[position_to_index[x]][position_to_index[y]]),
        )
        exc_lattice.update_grid_history = True

        spike_train_lattice = ln.DopaPoissonLattice(c1)
        spike_train_lattice.populate(poisson_neuron, exc_n, exc_n)

        dopamine_poisson_neuron = ln.DopaPoissonNeuron()
        dopamine_poisson_neuron.set_neurotransmitters(dopamine_neurotransmitters)

        spike_train_lattice = ln.DopaPoissonLattice(c2)
        spike_train_lattice.populate(dopamine_poisson_neuron, exc_n, exc_n)

        if not parsed_toml['simulation_parameters']['exc_only']:
            inh_lattice = ln.DopaIzhikevichLattice(i1)
            inh_lattice.populate(inh_neuron, inh_n, inh_n)
            inh_lattice.apply(setup_neuron)
            position_to_index = inh_lattice.position_to_index
            inh_lattice.connect(
                lambda x, y: bool(float(w_inh[position_to_index[x]][position_to_index[y]]) != 0), 
                lambda x, y: float(w_inh[position_to_index[x]][position_to_index[y]]),
            )
            # inh_lattice.update_grid_history = True

            network = ln.DopaIzhikevichNetwork.generate_network(
                [exc_lattice, inh_lattice], [spike_train_lattice],
            )
        else:
            network = ln.DopaIzhikevichNetwork.generate_network(
                [exc_lattice], [spike_train_lattice],
            )

        network.set_dt(parsed_toml['simulation_parameters']['dt'])
        network.parallel = True

        if not parsed_toml['simulation_parameters']['exc_only']:
            network.connect(
                i1, 
                e1, 
                lambda x, y: np.random.uniform(0, 1) < current_state['inh_to_exc_connectivity'], 
                lambda x, y: current_state['inh_to_exc_weight'],
            )
            network.connect(
                i1, 
                e1, 
                lambda x, y: np.random.uniform(0, 1) < current_state['exc_to_inh_connectivity'],
                lambda x, y: current_state['exc_to_inh_weight'],
            )

        network.connect(
            c1, 
            e1, 
            lambda x, y: np.random.uniform(0, 1) < current_state['spike_train_connectivity'], 
            lambda x, y: current_state['spike_train_to_exc']
        )

        network.connect(
            c2, 
            e1, 
            lambda x, y: np.random.uniform(0, 1) < current_state['spike_train_connectivity'], 
            lambda x, y: current_state['spike_train_to_exc']
        )

        network.electrical_synapse = False
        network.chemical_synapse = True

        network.apply_spike_train_lattice(
            1,
            stop_firing
        )
        network.run_lattices(parsed_toml['simulation_parameters']['off_phase'])

        network.apply_spike_train_lattice(
            1,
            start_firing
        )
        network.run_lattices(parsed_toml['simulation_parameters']['on_phase'])

        network.apply_spike_train_lattice(
            1,
            stop_firing
        )
        network.run_lattices(parsed_toml['simulation_parameters']['off_phase'])

        hist = network.get_lattice(0).history
        voltages = [float(np.array(i).mean()) for i in hist]

        return_to_baseline = determine_return_to_baseline(
            voltages,
            parsed_toml['simulation_parameters']['settling_period'],
            parsed_toml['simulation_parameters']['on_phase'],
            parsed_toml['simulation_parameters']['off_phase'],
            parsed_toml['simulation_parameters']['tolerance'],
        )

        current_value = {}

        if parsed_toml['simulation_parameters']['measure_snr']:
            current_value['first_snr'] = float(
                signal_to_noise(voltages[
                        parsed_toml['simulation_parameters']['settling_period']:parsed_toml['simulation_parameters']['off_phase']
                    ]
                )
            )
            current_value['second_snr'] = float(
                signal_to_noise(voltages[
                        parsed_toml['simulation_parameters']['on_phase'] + parsed_toml['simulation_parameters']['off_phase']:
                    ]
                )
            )
            current_value['during_disturbance'] = float(
                signal_to_noise(voltages[
                        parsed_toml['simulation_parameters']['on_phase']:
                        parsed_toml['simulation_parameters']['on_phase'] + parsed_toml['simulation_parameters']['off_phase']:
                    ]
                )
            )

        current_value['return_to_baseline'] = return_to_baseline
        current_value['voltages'] = voltages

        current_state['trial'] = trial

        key = generate_key(parsed_toml, current_state)

        simulation_output[key] = current_value

with open(parsed_toml['simulation_parameters']['filename'], 'w') as file:
    json.dump(simulation_output, file, indent=4)

print("\033[92mFinished simulation\033[0m")
