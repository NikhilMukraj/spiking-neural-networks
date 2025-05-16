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
}