#[cfg(test)]
mod test {
    /// ```compile_fail
    /// use nb_macro::neuron_builder;
    /// 
    /// 
    /// neuron_builder!(r#"
    ///     [neuron]
    ///         type: BasicIntegrateAndFire
    ///         vars: e = 0, e = 0, v_reset = -75, v_th = -55
    ///         on_spike:
    ///             v = v_reset
    ///         spike_detection: v >= v_th
    ///         on_iteration:
    ///             dv/dt = (v - e) + i
    ///     [end]
    /// "#);
    /// ```
    pub fn _test_placeholder() {

    }
}
