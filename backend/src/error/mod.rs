use std::fmt::{Display, Debug, Formatter, Result};


macro_rules! impl_debug_default {
    ($name:ident) => {
        impl Debug for $name {
            fn fmt(&self, f: &mut Formatter) -> Result {
                write!(f, "file: {}, line: {}, error: {}", file!(), line!(), self)
            }
        }
    };
}

/// Error set for potential graph errors
#[derive(Clone, Copy)]
pub enum GraphError {
    /// Presynaptic position cannot be found
    PresynapticNotFound,
    /// Postsynaptic position cannot be found
    PostsynapticNotFound,
    /// Position cannot be found
    PositionNotFound,
}

impl Display for GraphError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let err_msg = match self {
            GraphError::PostsynapticNotFound => "Postsynaptic position not found",
            GraphError::PresynapticNotFound => "Presynaptic position not found",
            GraphError::PositionNotFound => "Position not found",
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(GraphError);

/// Error set for potential lattice network errors
#[derive(Clone, Copy)]
pub enum LatticeNetworkError {
    /// Graph id already present in network (network must have graphs with unique identifiers)
    GraphIDAlreadyPresent,
    /// Postsynaptic id cannot be found
    PostsynapticIDNotFound,
    /// Presynaptic id cannot be found
    PresynapticIDNotFound,
    /// When connecting network, postsynaptic lattice cannot be a spike train lattice because spike trains
    /// cannot take inputs
    PostsynapticLatticeCannotBeSpikeTrain,
}

impl Display for LatticeNetworkError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let err_msg = match self {
            LatticeNetworkError::GraphIDAlreadyPresent => "Graph id already present in network",
            LatticeNetworkError::PostsynapticIDNotFound => "Postsynaptic id not present in network",
            LatticeNetworkError::PresynapticIDNotFound => "Postsynaptic id not present in network",
            LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain => "Postsynaptic lattice cannot be a spike train lattice because spike trains cannot take inputs",
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(LatticeNetworkError);

/// A set of errors for potential pattern errors
#[derive(Clone, Copy)]
pub enum PatternError {
    /// Pattern is not bipolar (`-1` or `1`)
    PatternIsNotBipolar,
    /// Pattern does not have the same dimensions throughout
    PatternDimensionsAreNotEqual,
}

impl Display for PatternError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let err_msg = match self {
            PatternError::PatternIsNotBipolar => "Pattern must be bipolar (-1 or 1)",
            PatternError::PatternDimensionsAreNotEqual => "Patterns must have the same dimensions",
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(PatternError);

/// A set of potential errors when using the genetic algorithm
#[derive(Clone)]
pub enum GeneticAlgorithmError {
    /// Non binary found in binary bitstring
    NonBinaryInBitstring(String),
    /// Bounds length is not compatible with `n_bits`
    InvalidBoundsLength,
    /// Bitstring length is not compatable with `n_bits`
    InvalidBitstringLength,
    /// Decoding bitstring cannot be completed due to non binary in string or overflow error
    DecodingBitstringFailure(String),
    /// Population parameter in genetic algorithm must be even
    PopulationMustBeEven,
    /// Objective function failed
    ObjectiveFunctionFailure(String),
}

impl Display for GeneticAlgorithmError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let err_msg = match self {
            GeneticAlgorithmError::NonBinaryInBitstring(string) => format!(
                "Non binary found in bitstring: {}", string
            ),
            GeneticAlgorithmError::InvalidBoundsLength => String::from("Bounds length does not match n_bits"),
            GeneticAlgorithmError::InvalidBitstringLength => String::from("String length is indivisible by n_bits"),
            GeneticAlgorithmError::DecodingBitstringFailure(string) => format!(
                "Bitstring could not be decoded from binary (non binary found or integer overflow): {}", 
                string,
            ),
            GeneticAlgorithmError::PopulationMustBeEven => String::from("n_pop should be even"),
            GeneticAlgorithmError::ObjectiveFunctionFailure(string) => format!(
                "Genetic algorithm objective function failed: {}", 
                string,
            )
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(GeneticAlgorithmError);

/// A set of errors that may occur when using the library
#[derive(Clone)]
pub enum SpikingNeuralNetworksError {
    /// Errors related to genetic algorithm
    GeneticAlgorithmRelatedErrors(GeneticAlgorithmError),
    /// Errors related to graph processing
    GraphRelatedError(GraphError),
    /// Errors related to lattice networks
    LatticeNetworkRelatedError(LatticeNetworkError),
    /// Errors related to patterns
    PatternRelatedError(PatternError),
}

impl Display for SpikingNeuralNetworksError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            Self::GeneticAlgorithmRelatedErrors(err) => write!(f, "{}", err),
            SpikingNeuralNetworksError::GraphRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksError::LatticeNetworkRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksError::PatternRelatedError(err) => write!(f, "{}", err),
        }
    }
}

impl_debug_default!(SpikingNeuralNetworksError);

impl From<GeneticAlgorithmError> for SpikingNeuralNetworksError {
    fn from(err: GeneticAlgorithmError) -> SpikingNeuralNetworksError {
        SpikingNeuralNetworksError::GeneticAlgorithmRelatedErrors(err)
    }
}

impl From<GraphError> for SpikingNeuralNetworksError {
    fn from(err: GraphError) -> SpikingNeuralNetworksError {
        SpikingNeuralNetworksError::GraphRelatedError(err)
    }
}

impl From<LatticeNetworkError> for SpikingNeuralNetworksError {
    fn from(err: LatticeNetworkError) -> SpikingNeuralNetworksError {
        SpikingNeuralNetworksError::LatticeNetworkRelatedError(err)
    }
}

impl From<PatternError> for SpikingNeuralNetworksError {
    fn from(err: PatternError) -> SpikingNeuralNetworksError {
        SpikingNeuralNetworksError::PatternRelatedError(err)
    }
}
