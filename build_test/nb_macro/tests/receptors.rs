#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;


    neuron_builder!(r#"
    [receptors]
        type: BasicReceptors
        neurotransmitter: X
        vars: g = 1, e = 0
        on_iteration:
            current = g * r * (v - e)
    [end]
    "#);
}