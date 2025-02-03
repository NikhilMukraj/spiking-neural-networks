#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;


    neuron_builder!(r#"
    [receptor_kinetics]
        type: BoundedReceptorKinetics
        vars: r_max = 1
        on_iteration:
            r = min(max(t, 0), r_max)
    [end]
    "#);

    #[test]
    pub fn test_bounded_receptor_kinetics() {
        let mut to_test = BoundedReceptorKinetics::default();

        let ts = [-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2.];

        for i in ts {
            to_test.apply_r_change(i, 0.1);
            assert_eq!(to_test.r, to_test.r.clamp(0., 1.));
        }
    }
}