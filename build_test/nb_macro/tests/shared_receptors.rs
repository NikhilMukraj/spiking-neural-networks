use nb_macro::neuron_builder;


neuron_builder!(r#"
[receptors]
    type: MultipleReceptors
    neurotransmitter: A
    vars: current = 0, g = 1, e = 0
    on_iteration:
        current = g * r * (v - e)
    neurotransmitter: B
    vars: current = 0, g = 1, e = 0
    on_iteration:
        current = 2 * g * r * (v - e)
[end]

[receptors]
        type: MixedReceptors
        vars: m = 0
        neurotransmitter: Iono
        vars: current = 0, g = 1, e = 0
        on_iteration:
            current = g * m * r * (v - e)
        neurotransmitter: Meta
        vars: s = 1
        on_iteration:
            m = s * r
    [end]
"#);
