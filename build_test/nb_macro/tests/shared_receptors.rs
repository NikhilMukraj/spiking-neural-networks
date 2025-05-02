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

[receptors]
    type: CombinedReceptors
    neurotransmitter: Combined
    receptors: r1, r2
    vars: current = 0, g1 = 2, e1 = 0, g2 = 2, e2 = 0
    on_iteration:
        current = g1 * r1 * (v - e1) + g2 * r2 * (v - e2)
[end]
"#);
