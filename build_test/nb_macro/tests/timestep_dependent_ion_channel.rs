#[cfg(test)]
mod test {
    // use nb_macro::neuron_builder; 

    
    // neuron_builder!(r#"
    //     [ion_channel]
    //         type: CalciumIonChannel 
    //         vars: e = 0, g = 1,
    //         gating_vars: s
    //         on_iteration:
    //             s.alpha = 1.6 / exp(1 + (-0.072 * (v - 5)));
    //             s.beta = (0.02 * (v + 8.9)) / ((exp(v + 8.9) / 5) - 1);
    //             current = g * (v - e)
    //     [end]
    // "#);

    // #[test]
    // pub fn test_current() {
    //     let mut leak = CalciumIonChannel::default();

    //     let voltages = [-50., -40., -30., -20., -10., 0., 10., 20., 30.];

    //     for i in voltages {
    //         leak.update_current(i);

    //         assert_eq!(leak.current, i);
    //     }

    //     leak.g = 2.;

    //     for i in voltages {
    //         leak.update_current(i);

    //         assert_eq!(leak.current, 2. * i);
    //     }

    //     leak.e = -10.;

    //     for i in voltages {
    //         leak.update_current(i);

    //         assert_eq!(leak.current, 2. * (i - -10.));
    //     }
    // }
}