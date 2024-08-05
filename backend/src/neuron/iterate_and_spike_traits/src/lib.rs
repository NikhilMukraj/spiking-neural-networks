use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};


/// Derive macro to automatically implement many necessary traits for the `IterateAndSpike` trait,
/// including `CurrentVoltage`, `GapConductance`, `Timestep`, `GaussianFactor`, 
/// `LastFiringTime`, and `IsSpiking`
#[proc_macro_derive(IterateAndSpikeBase)]
pub fn derive_iterate_and_spike_traits(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics CurrentVoltage for #name #ty_generics #where_clause {
            fn get_current_voltage(&self) -> f32 {
                self.current_voltage
            }
        }

        impl #impl_generics GapConductance for #name #ty_generics #where_clause {
            fn get_gap_conductance(&self) -> f32 {
                self.gap_conductance
            }
        }

        impl #impl_generics GaussianFactor for #name #ty_generics #where_clause {
            fn get_gaussian_factor(&self) -> f32 {
                self.gaussian_params.get_random_number()
            }
        }

        impl #impl_generics Timestep for #name #ty_generics #where_clause {
            fn get_dt(&self) -> f32 {
                self.dt
            }

            fn set_dt(&mut self, dt: f32) {
                self.dt = dt;
            }
        }

        impl #impl_generics IsSpiking for #name #ty_generics #where_clause {
            fn is_spiking(&self) -> bool {
                self.is_spiking
            }
        }

        impl #impl_generics LastFiringTime for #name #ty_generics #where_clause {
            fn set_last_firing_time(&mut self, timestep: Option<usize>) {
                self.last_firing_time = timestep;
            }
        
            fn get_last_firing_time(&self) -> Option<usize> {
                self.last_firing_time
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro to automatically implement the `Timestep` trait assuming the 
/// struct has field `dt`
#[proc_macro_derive(Timestep)]
pub fn derive_timestep_trait(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics Timestep for #name #ty_generics #where_clause {
            fn get_dt(&self) -> f32 {
                self.dt
            }

            fn set_dt(&mut self, dt: f32) {
                self.dt = dt;
            }
        }
    };

    TokenStream::from(expanded)
}
