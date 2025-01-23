#[cfg(test)]
mod test {
    use nb_macro::neuron_builder; 

    
    neuron_builder!(r#"
        [ion_channel]
            type: TestChannel
            vars: e = 0, g = 1
            gating_vars: n
            on_iteration:
                current = g * n.alpha * n.beta * n.state * (v - e)
        [end]
    "#);

    #[test]
    pub fn test_current() {
        let mut leak = TestChannel::default();
        leak.n.alpha = 1.;
        leak.n.beta = 1.;
        leak.n.state = 1.;

        let voltages = [-50., -40., -30., -20., -10., 0., 10., 20., 30.];

        for i in voltages {
            leak.update_current(i);

            assert_eq!(leak.current, i);
        }

        leak.g = 2.;

        for i in voltages {
            leak.update_current(i);

            assert_eq!(leak.current, 2. * i);
        }

        leak.e = -10.;

        for i in voltages {
            leak.update_current(i);

            assert_eq!(leak.current, 2. * (i - -10.));
        }

        leak.n.state = 2.;

        for i in voltages {
            leak.update_current(i);

            assert_eq!(leak.current, 2. * 2. * (i - -10.));
        }

        leak.n.alpha = 4.;
        leak.n.beta = 3.;

        
        for i in voltages {
            leak.update_current(i);

            assert_eq!(leak.current, 2. * 4. * 3. * 2. * (i - -10.));
        }
    }
}