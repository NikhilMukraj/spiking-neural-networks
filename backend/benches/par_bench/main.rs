#![feature(test)]
extern crate test;


mod tests {
    use test::Bencher;
    extern crate rand;
    extern crate spiking_neural_networks;
    use rand::Rng;
    use spiking_neural_networks::neuron::{
        integrate_and_fire::IzhikevichNeuron,
        Lattice, RunLattice
    };

    fn sparse_connection(x: (usize, usize), y: (usize, usize)) -> bool {
        ((x.0 as f32 - y.0 as f32).powf(2.) + (x.1 as f32 - y.1 as f32).powf(2.)).sqrt() <= 3. && 
        x != y
    }

    #[bench]
    fn bench_reg_5x5(b: &mut Bencher) {
        let mut grid5x5 = Lattice::default_impl();
        grid5x5.populate(&IzhikevichNeuron::default_impl(), 5, 5).unwrap();
        grid5x5.connect(&|x, y| x != y, None);

        b.iter(|| {
            grid5x5.apply(
                |neuron| {
                    let mut rng = rand::thread_rng();
                    neuron.current_voltage = rng.gen_range(neuron.v_init..neuron.v_th);
                }
            );
            grid5x5.run_lattice(1).expect("Could not run lattice");
        })
    }

    #[bench]
    fn bench_par_5x5(b: &mut Bencher) {
        let mut grid5x5 = Lattice::default_impl();
        grid5x5.populate(&IzhikevichNeuron::default_impl(), 5, 5).unwrap();
        grid5x5.connect(&|x, y| x != y, None);
        grid5x5.parallel = true;

        b.iter(|| {
            grid5x5.apply(
                |neuron| {
                    let mut rng = rand::thread_rng();
                    neuron.current_voltage = rng.gen_range(neuron.v_init..neuron.v_th);
                }
            );
            grid5x5.run_lattice(1).expect("Could not run lattice");
        })
    }

    #[bench]
    fn bench_reg_10x10(b: &mut Bencher) {
        let mut grid10x10 = Lattice::default_impl();
        grid10x10.populate(&IzhikevichNeuron::default_impl(), 10, 10).unwrap();
        grid10x10.connect(&|x, y| x != y, None);

        b.iter(|| {
            grid10x10.apply(
                |neuron| {
                    let mut rng = rand::thread_rng();
                    neuron.current_voltage = rng.gen_range(neuron.v_init..neuron.v_th);
                }
            );
            grid10x10.run_lattice(1).expect("Could not run lattice");
        })
    }

    #[bench]
    fn bench_par_10x10(b: &mut Bencher) {
        let mut grid10x10 = Lattice::default_impl();
        grid10x10.populate(&IzhikevichNeuron::default_impl(), 10, 10).unwrap();
        grid10x10.connect(&|x, y| x != y, None);
        grid10x10.parallel = true;

        b.iter(|| {
            grid10x10.apply(
                |neuron| {
                    let mut rng = rand::thread_rng();
                    neuron.current_voltage = rng.gen_range(neuron.v_init..neuron.v_th);
                }
            );
            grid10x10.run_lattice(1).expect("Could not run lattice");
        })
    }

    #[bench]
    fn bench_sparse_reg_5x5(b: &mut Bencher) {
        let mut grid5x5 = Lattice::default_impl();
        grid5x5.populate(&IzhikevichNeuron::default_impl(), 5, 5).unwrap();
        grid5x5.connect(&sparse_connection, None);

        b.iter(|| {
            grid5x5.apply(
                |neuron| {
                    let mut rng = rand::thread_rng();
                    neuron.current_voltage = rng.gen_range(neuron.v_init..neuron.v_th);
                }
            );
            grid5x5.run_lattice(1).expect("Could not run lattice");
        })
    }

    #[bench]
    fn bench_sparse_par_5x5(b: &mut Bencher) {
        let mut grid5x5 = Lattice::default_impl();
        grid5x5.populate(&IzhikevichNeuron::default_impl(), 5, 5).unwrap();
        grid5x5.connect(&sparse_connection, None);
        grid5x5.parallel = true;

        b.iter(|| {
            grid5x5.apply(
                |neuron| {
                    let mut rng = rand::thread_rng();
                    neuron.current_voltage = rng.gen_range(neuron.v_init..neuron.v_th);
                }
            );
            grid5x5.run_lattice(1).expect("Could not run lattice");
        })
    }

    #[bench]
    fn bench_sparse_reg_10x10(b: &mut Bencher) {
        let mut grid10x10 = Lattice::default_impl();
        grid10x10.populate(&IzhikevichNeuron::default_impl(), 10, 10).unwrap();
        grid10x10.connect(&sparse_connection, None);

        b.iter(|| {
            grid10x10.apply(
                |neuron| {
                    let mut rng = rand::thread_rng();
                    neuron.current_voltage = rng.gen_range(neuron.v_init..neuron.v_th);
                }
            );
            grid10x10.run_lattice(1).expect("Could not run lattice");
        })
    }

    #[bench]
    fn bench_sparse_par_10x10(b: &mut Bencher) {
        let mut grid10x10 = Lattice::default_impl();
        grid10x10.populate(&IzhikevichNeuron::default_impl(), 10, 10).unwrap();
        grid10x10.connect(&sparse_connection, None);
        grid10x10.parallel = true;

        b.iter(|| {
            grid10x10.apply(
                |neuron| {
                    let mut rng = rand::thread_rng();
                    neuron.current_voltage = rng.gen_range(neuron.v_init..neuron.v_th);
                }
            );
            grid10x10.run_lattice(1).expect("Could not run lattice");
        })
    }
}
