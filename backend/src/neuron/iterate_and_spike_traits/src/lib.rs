use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};


/// Derive macro to automatically implement many necessary traits for the `IterateAndSpike` trait,
/// including `CurrentVoltage`, `GapConductance`, `Potentiation`, `GaussianFactor`, `LastFiringTime`,
/// and `STDP`
#[proc_macro_derive(IterateAndSpikeBase)]
pub fn derive_iterate_and_spike_traits(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    // Get the name of the struct we are deriving the trait for
    let name = input.ident;

    // Generate the implementation of the trait
    let expanded = quote! {
        impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> CurrentVoltage for #name<T, R> {
            fn get_current_voltage(&self) -> f64 {
                self.current_voltage
            }
        }

        impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> GapConductance for #name<T, R> {
            fn get_gap_conductance(&self) -> f64 {
                self.gap_conductance
            }
        }

        impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Potentiation for #name<T, R> {
            fn get_potentiation_type(&self) -> PotentiationType {
                self.potentiation_type
            }
        }

        impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> GaussianFactor for #name<T, R> {
            fn get_gaussian_factor(&self) -> f64 {
                crate::distribution::limited_distr(
                    self.gaussian_params.mean, 
                    self.gaussian_params.std, 
                    self.gaussian_params.min, 
                    self.gaussian_params.max,
                )
            }
        }

        impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> LastFiringTime for #name<T, R> {
            fn set_last_firing_time(&mut self, timestep: Option<usize>) {
                self.last_firing_time = timestep;
            }
        
            fn get_last_firing_time(&self) -> Option<usize> {
                self.last_firing_time
            }
        }

        impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> STDP for #name<T, R> {        
            fn get_stdp_params(&self) -> &STDPParameters {
                &self.stdp_params
            }
        }
    };

    TokenStream::from(expanded)
}