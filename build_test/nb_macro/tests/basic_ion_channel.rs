#[cfg(test)]
mod test {
    use nb_macro::neuron_builder; 

    
    neuron_builder!(r#"
        [ion_channel]
            type: TestLeak
            vars: e = 0, g = 1,
            on_iteration:
                current = g * (v - e)
        [end]
    "#);

    #[test]
    pub fn test_current() {
        let mut leak = TestLeak::default();

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
    }
}