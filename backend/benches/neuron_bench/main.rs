#![feature(test)]
extern crate test;


mod tests {
    use test::Bencher;
    extern crate spiking_neural_networks;
    use spiking_neural_networks::neuron::{
        hodgkin_huxley::HodgkinHuxleyNeuron, integrate_and_fire::{
            IzhikevichNeuron, 
            QuadraticIntegrateAndFireNeuron,
        }, iterate_and_spike::IterateAndSpike
    };

    #[bench]
    fn bench_izhikevich(b: &mut Bencher) {
        let mut neuron = IzhikevichNeuron::default_impl();

        b.iter(|| {
            neuron.iterate_and_spike(30.)
        })
    }

    #[bench]
    fn bench_quad(b: &mut Bencher) {
        let mut neuron = QuadraticIntegrateAndFireNeuron::default_impl();

        b.iter(|| {
            neuron.iterate_and_spike(30.)
        })
    }

    #[bench]
    fn bench_hodgkin_huxley(b: &mut Bencher) {
        let mut neuron = HodgkinHuxleyNeuron::default_impl();

        b.iter(|| {
            neuron.iterate_and_spike(30.)
        })
    }
}
