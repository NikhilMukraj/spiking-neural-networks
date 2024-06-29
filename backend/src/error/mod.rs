use std::fmt::{Display, Debug, Formatter, Result};


macro_rules! impl_debug_default {
    ($name:ident) => {
        impl Debug for $name {
            fn fmt(&self, f: &mut Formatter) -> Result {
                write!(f, "file: {}, line: {}, error: {}", self.file, self.line, self.error)
            }
        }
    };
}

macro_rules! impl_error_new {
    ($name:ident, $error_set:ident) => {
        impl $name {
            pub fn new(error: $error_set, file: &'static str, line: u32) -> Self {
                $name {
                    file: file,
                    line: line,
                    error: error,
                }
            }
        }
    };
}

/// Error set for potential graph errors
#[derive(Clone, Copy)]
pub enum GraphErrorKind {
    /// Presynaptic position cannot be found
    PresynapticNotFound,
    /// Postsynaptic position cannot be found
    PostsynapticNotFound,
    /// Position cannot be found
    PositionNotFound,
}

#[derive(Clone)]
pub struct GraphError {
    file: &'static str,
    line: u32,
    error: GraphErrorKind,
}

impl Display for GraphErrorKind {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let err_msg = match self {
            GraphErrorKind::PostsynapticNotFound => "Postsynaptic position not found",
            GraphErrorKind::PresynapticNotFound => "Presynaptic position not found",
            GraphErrorKind::PositionNotFound => "Position not found",
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(GraphError);
impl_error_new!(GraphError, GraphErrorKind);

/// Error set for potential lattice network errors
#[derive(Clone, Copy)]
pub enum LatticeNetworkErrorKind {
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

#[derive(Clone)]
pub struct LatticeNetworkError {
    file: &'static str,
    line: u32,
    error: LatticeNetworkErrorKind,
}

impl Display for LatticeNetworkErrorKind {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let err_msg = match self {
            LatticeNetworkErrorKind::GraphIDAlreadyPresent => "Graph id already present in network",
            LatticeNetworkErrorKind::PostsynapticIDNotFound => "Postsynaptic id not present in network",
            LatticeNetworkErrorKind::PresynapticIDNotFound => "Postsynaptic id not present in network",
            LatticeNetworkErrorKind::PostsynapticLatticeCannotBeSpikeTrain => "Postsynaptic lattice cannot be a spike train lattice because spike trains cannot take inputs",
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(LatticeNetworkError);
impl_error_new!(LatticeNetworkError, LatticeNetworkErrorKind);

/// A set of errors for potential pattern errors
#[derive(Clone, Copy)]
pub enum PatternErrorKind {
    /// Pattern is not bipolar (`-1` or `1`)
    PatternIsNotBipolar,
    /// Pattern does not have the same dimensions throughout
    PatternDimensionsAreNotEqual,
}

#[derive(Clone)]
pub struct PatternError {
    file: &'static str,
    line: u32,
    error: PatternErrorKind,
}

impl Display for PatternErrorKind {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let err_msg = match self {
            PatternErrorKind::PatternIsNotBipolar => "Pattern must be bipolar (-1 or 1)",
            PatternErrorKind::PatternDimensionsAreNotEqual => "Patterns must have the same dimensions",
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(PatternError);
impl_error_new!(PatternError, PatternErrorKind);

/// A set of potential errors when using the genetic algorithm
#[derive(Clone)]
pub enum GeneticAlgorithmErrorKind {
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

#[derive(Clone)]
pub struct GeneticAlgorithmError {
    file: &'static str,
    line: u32,
    error: GeneticAlgorithmErrorKind,
}

impl Display for GeneticAlgorithmErrorKind {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let err_msg = match self {
            GeneticAlgorithmErrorKind::NonBinaryInBitstring(string) => format!(
                "Non binary found in bitstring: {}", string
            ),
            GeneticAlgorithmErrorKind::InvalidBoundsLength => String::from("Bounds length does not match n_bits"),
            GeneticAlgorithmErrorKind::InvalidBitstringLength => String::from("String length is indivisible by n_bits"),
            GeneticAlgorithmErrorKind::DecodingBitstringFailure(string) => format!(
                "Bitstring could not be decoded from binary (non binary found or integer overflow): {}", 
                string,
            ),
            GeneticAlgorithmErrorKind::PopulationMustBeEven => String::from("n_pop should be even"),
            GeneticAlgorithmErrorKind::ObjectiveFunctionFailure(string) => format!(
                "Genetic algorithm objective function failed: {}", 
                string,
            )
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(GeneticAlgorithmError);
impl_error_new!(GeneticAlgorithmError, GeneticAlgorithmErrorKind);

/// A set of potential errors when using series processing tools
#[derive(Clone, Copy)]
pub enum TimeSeriesProcessingErrorKind {
    /// Series must be the same length
    SeriesAreNotSameLength
}

#[derive(Clone)]
pub struct TimeSeriesProcessingError {
    file: &'static str,
    line: u32,
    error: TimeSeriesProcessingErrorKind,
}

impl Display for TimeSeriesProcessingErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let err_msg = match self {
            TimeSeriesProcessingErrorKind::SeriesAreNotSameLength => "Lengths of input series must match"
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(TimeSeriesProcessingError);
impl_error_new!(TimeSeriesProcessingError, TimeSeriesProcessingErrorKind);

/// A set of errors that may occur when using the library
#[derive(Clone)]
pub enum SpikingNeuralNetworksErrorKind {
    /// Errors related to EEG processing
    SeriesProcessingRelatedError(TimeSeriesProcessingErrorKind),
    /// Errors related to genetic algorithm
    GeneticAlgorithmRelatedErrors(GeneticAlgorithmErrorKind),
    /// Errors related to graph processing
    GraphRelatedError(GraphErrorKind),
    /// Errors related to lattice networks
    LatticeNetworkRelatedError(LatticeNetworkErrorKind),
    /// Errors related to patterns
    PatternRelatedError(PatternErrorKind),
}

#[derive(Clone)]
pub struct SpikingNeuralNetworksError {
    file: &'static str,
    line: u32,
    error: SpikingNeuralNetworksErrorKind,
}

impl Display for SpikingNeuralNetworksErrorKind {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            SpikingNeuralNetworksErrorKind::SeriesProcessingRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksErrorKind::GeneticAlgorithmRelatedErrors(err) => write!(f, "{}", err),
            SpikingNeuralNetworksErrorKind::GraphRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksErrorKind::LatticeNetworkRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksErrorKind::PatternRelatedError(err) => write!(f, "{}", err),
        }
    }
}

impl_debug_default!(SpikingNeuralNetworksError);
impl_error_new!(SpikingNeuralNetworksError, SpikingNeuralNetworksErrorKind);

macro_rules! impl_from_error_default {
    ($error_name:ident, $variant_name:ident) => {
        impl From<$error_name> for SpikingNeuralNetworksError {
            fn from(err: $error_name) -> SpikingNeuralNetworksError {
                SpikingNeuralNetworksError {
                    error: SpikingNeuralNetworksErrorKind::$variant_name(err.error),
                    file: err.file,
                    line: err.line,
                }
            }
        }
    };
}

impl_from_error_default!(TimeSeriesProcessingError, SeriesProcessingRelatedError);
impl_from_error_default!(GeneticAlgorithmError, GeneticAlgorithmRelatedErrors);
impl_from_error_default!(GraphError, GraphRelatedError);
impl_from_error_default!(LatticeNetworkError, LatticeNetworkRelatedError);
impl_from_error_default!(PatternError, PatternRelatedError);
