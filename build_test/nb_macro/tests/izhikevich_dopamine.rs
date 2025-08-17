use nb_macro::neuron_builder;


neuron_builder!("
    [neurotransmitter_kinetics]
        type: BoundedNeurotransmitterKinetics
        vars: t_max = 1, clearance_constant = 0.001, conc = 0
        on_iteration:
            [if] is_spiking [then]
                conc = t_max
            [else]
                conc = 0
            [end]

            t = t + dt * -clearance_constant * t + conc

            t = min(max(t, 0), t_max)
    [end]

    [receptor_kinetics]
        type: BoundedReceptorKinetics
        vars: r_max = 1
        on_iteration:
            r = min(max(t, 0), r_max)
    [end]

    [receptors]
        type: DopaGluGABA
        kinetics: BoundedReceptorKinetics
        vars: inh_modifier = 1, nmda_modifier = 1
        neurotransmitter: Glutamate
        receptors: ampa_r, nmda_r
        vars: current = 0, g_ampa = 1, g_nmda = 0.6, e_ampa = 0, e_nmda = 0, mg = 0.3
        on_iteration:
            current = inh_modifier * g_ampa * ampa_r * (v - e_ampa) + (1 / (1 + (exp(-0.062 * v) * mg / 3.75))) * inh_modifier * g_nmda * (nmda_r r^ nmda_modifier) * (v - e_nmda)
        neurotransmitter: GABA
        vars: current = 0, g = 1.2, e = -80
        on_iteration:
            current = g * r * (v - e)
        neurotransmitter: Dopamine
        receptors: r_d1, r_d2
        vars: s_d2 = 0, s_d1 = 0
        on_iteration:
            inh_modifier = 1 - (r_d2 * s_d2)
            nmda_modifier = 1 - (r_d1 * s_d1)
    [end]

    [neuron]
        type: IzhikevichNeuron
        kinetics: BoundedNeurotransmitterKinetics, BoundedReceptorKinetics
        receptors: DopaGluGABA
        vars: u = 30, a = 0.02, b = 0.2, c = -55, d = 8, v_th = 30, tau_m = 1
        on_spike: 
            v = c
            u += d
        spike_detection: v >= v_th
        on_iteration:
            du/dt = (a * (b * v - u)) / tau_m
            dv/dt = (0.04 * v * v + 5 * v + 140 - u + i) / c_m
    [end]"
);