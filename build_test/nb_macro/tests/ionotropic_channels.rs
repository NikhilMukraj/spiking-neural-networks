use nb_macro::neuron_builder;


neuron_builder!("
    [receptors]
        type: Ionotropic
        neurotransmitter: AMPA
        vars: current = 0, g = 1, e = 0
        on_iteration:
            current = g * r * (v - e)
        neurotransmitter: NMDA
        vars: current = 0, g = 0.6, mg = 0.3, e = 0
        on_iteration:
            current = 1 / (1 + (exp(-0.062 * v) * mg / 3.75)) * g * r * (v - e)
        neurotransmitter: GABA
        vars: current = 0, g = 1.2, e = -80
        on_iteration:
            current = g * r * (v - e)
    [end]
");