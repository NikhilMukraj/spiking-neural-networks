use nb_macro::neuron_builder;


neuron_builder!(r#"
    [ion_channel]
        type: ReducedCalciumChannel
        vars: m_ss = 0, e_ca = 120, g_ca = 4, v1 = -1.2, v2 = 18
        on_iteration:
            m_inf = 0.5 * (1. + tanh((v - v1) / v2))
            current = g_ca * m_ss * (v - e_ca)
    [end]

    [ion_channel]
        type: KSteadyStateChannel
        vars: g_k = 8, v_k = -84, n = 0, n_ss = 0, t_n = 0, phi = 0.067, v_3 = 12, v_4 = 17.4
        on_iteration:
            n_ss = 0.5 * (1. + tanh((v - v_3) / v_4))
            t_n = 1. / (phi * cosh((v - v_3) / (2. * v_4)))

            n += ((n_ss - n) / t_n) * dt

            current = g_k * n * (v - v_k)
    [end]

    [ion_channel]
        type: LeakIonChannel
        vars: e = -55, g = 0.3
        on_iteration:
            current = g * (v - e)
    [end]

    [neuron]
        type: MorrisLecarNeuron
        ion_channels: ca_channel = ReducedCalciumChannel, k_channel = KSteadyStateChannel, leak_channel = LeakIonChannel
        vars: v_init = -70, v_th = 25
        spike_detection: continuous()
        on_iteration:
            ca_channel.update_current(v)
            k_channel.update_current(v)
            leak_channel.update_current(v)

            dv/dt = (-ca_channel.current - k_channel.current - leak_channel.current + i + gap_conductance * (v - v_init) ) / c_m
    [end]
"#);