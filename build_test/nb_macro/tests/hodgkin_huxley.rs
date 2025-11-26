use nb_macro::neuron_builder;


neuron_builder!(r#"
    [ion_channel]
        type: NaIonChannel
        vars: e = 50, g = 120
        gating_vars: m, h
        on_iteration:
            m.alpha = 0.1 * ((v + 40.) / exp(1. - (-(v + 40.) / 10.)))
            m.beta = 4. * exp(-(v + 65.) / 18.);
            h.alpha = 0.07 * exp(-(v + 65.) / 20.);
            h.beta = 1. / (exp(-(v + 35.) / 10.) + 1.);

            m.update(dt)
            h.update(dt)

            current = g * m.state ^ 3 * h.state * (v - e)
    [end]
    
    [ion_channel]
        type: KIonChannel
        vars: e = -77, g = 36
        gating_vars: n
        on_iteration:
            n.alpha = 0.01 * ((v + 55.) / exp(1. - (-(v + 55.) / 10.)))
            n.beta = 0.125 * exp(-(v + 65.) / 80.);

            n.update(dt)

            current = g * n.state ^ 4 * (v - e)
    [end]

    [ion_channel]
        type: LeakIonChannel
        vars: e = -55, g = 0.3
        on_iteration:
            current = g * (v - e)
    [end]

    [neuron]
        type: HodgkinHuxley
        ion_channels: k = KIonChannel, na = NaIonChannel, leak = LeakIonChannel
        spike_detection: continuous()
        on_iteration:
            na.update_current(v)
            k.update_current(v)
            leak.update_current(v)

            dv/dt = -(na.current + k.current + leak.current) + i
    [end]
    "#
);
