use std::fmt::{Display, Debug, Formatter, Result};


macro_rules! impl_debug_default {
    ($name:ident) => {
        impl Debug for $name {
            fn fmt(&self, f: &mut Formatter) -> Result {
                write!(f, "{}", self)
            }
        }
    };
}

/// Error set for potential graph errors
#[derive(Clone, PartialEq, Eq)]
pub enum GraphError {
    /// Presynaptic position cannot be found
    PresynapticNotFound(String),
    /// Postsynaptic position cannot be found
    PostsynapticNotFound(String),
    /// Position cannot be found
    PositionNotFound(String),
}

impl Display for GraphError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let err_msg = match self {
            GraphError::PostsynapticNotFound(value) => format!("Postsynaptic position not found, position: {:#?}", value),
            GraphError::PresynapticNotFound(value) => format!("Presynaptic position not found, position: {:#?}", value),
            GraphError::PositionNotFound(value) => format!("Position not found, position: {:#?}", value),
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(GraphError);

/// Error set for potential lattice network errors
#[derive(Clone, PartialEq, Eq)]
pub enum LatticeNetworkError {
    /// Graph id already present in network (network must have graphs with unique identifiers)
    GraphIDAlreadyPresent(usize),
    /// Postsynaptic id cannot be found
    PostsynapticIDNotFound(usize),
    /// Presynaptic id cannot be found
    PresynapticIDNotFound(usize),
    /// Lattice id cannot be found, (non spike train lattice)
    IDNotFoundInLattices(usize),
    /// When connecting network, postsynaptic lattice cannot be a spike train lattice because spike trains
    /// cannot take inputs
    PostsynapticLatticeCannotBeSpikeTrain,
    /// When connecting reward modulated network, at least one lattice has to be reward modulated
    CannotConnectWithRewardModulatedConnection,
    /// When connecting reward modulated lattice, RewardModulatedConnection cannot be used to connect a
    /// reward modulated lattice internally
    RewardModulatedConnectionNotCompatibleInternally,
    /// Connect function must have non reward modulated lattices, connect with reward modulation instead
    ConnectFunctionMustHaveNonRewardModulatedLattice,
    /// Connection function failed
    ConnectionFailure(String)
}

impl Display for LatticeNetworkError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let err_msg = match self {
            LatticeNetworkError::GraphIDAlreadyPresent(value) => format!(
                "Graph id already present in network, id: {}", value
            ),
            LatticeNetworkError::PostsynapticIDNotFound(value) => format!(
                "Postsynaptic id not present in network, id: {}", value
            ),
            LatticeNetworkError::PresynapticIDNotFound(value) => format!(
                "Postsynaptic id not present in network, id: {}", value
            ),
            LatticeNetworkError::IDNotFoundInLattices(value) => format!(
                "Id not present in lattices, id: {}", value
            ),
            LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain => String::from(
                "Postsynaptic lattice cannot be a spike train lattice because spike trains cannot take inputs"
            ),
            LatticeNetworkError::CannotConnectWithRewardModulatedConnection => String::from(
                "When connecting reward modulated network, at least one lattice has to be reward modulated"
            ),
            LatticeNetworkError::RewardModulatedConnectionNotCompatibleInternally => String::from(
                "When connecting reward modulated lattice, RewardModulatedConnection cannot be used to connect a reward modulated lattice internally"
            ),
            LatticeNetworkError::ConnectFunctionMustHaveNonRewardModulatedLattice => String::from(
                "Connect function must have non reward modulated lattices, connect with reward modulation instead",
            ),
            LatticeNetworkError::ConnectionFailure(value) => {
                format!("Failed to connect: {}", value)
            },
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(LatticeNetworkError);

/// A set of errors for potential pattern errors
#[derive(Clone, Copy, PartialEq, Eq)]
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
#[derive(Clone, PartialEq, Eq)]
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

/// A set of potential errors when using series processing tools
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TimeSeriesProcessingError {
    /// Series must be the same length
    SeriesAreNotSameLength
}

impl Display for TimeSeriesProcessingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let err_msg = match self {
            TimeSeriesProcessingError::SeriesAreNotSameLength => "Lengths of input series must match"
        };

        write!(f, "{}", err_msg)
    }
}

impl_debug_default!(TimeSeriesProcessingError);

#[derive(Clone, PartialEq, Eq)]
pub enum AgentError {
    AgentIterationFailure(String)
}

impl Display for AgentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let AgentError::AgentIterationFailure(err_msg) = self;

        write!(f, "{}", err_msg)
    }
}

/// A set of errors that may occur when using the library
#[derive(Clone, PartialEq, Eq)]
pub enum SpikingNeuralNetworksError {
    /// Errors related to EEG processing
    SeriesProcessingRelatedError(TimeSeriesProcessingError),
    /// Errors related to genetic algorithm
    GeneticAlgorithmRelatedErrors(GeneticAlgorithmError),
    /// Errors related to graph processing
    GraphRelatedError(GraphError),
    /// Errors related to lattice networks
    LatticeNetworkRelatedError(LatticeNetworkError),
    /// Errors related to patterns
    PatternRelatedError(PatternError),
    /// Errors related to agent
    AgentRelatedError(AgentError),
}

impl Display for SpikingNeuralNetworksError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            SpikingNeuralNetworksError::SeriesProcessingRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksError::GeneticAlgorithmRelatedErrors(err) => write!(f, "{}", err),
            SpikingNeuralNetworksError::GraphRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksError::LatticeNetworkRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksError::PatternRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksError::AgentRelatedError(err) => write!(f, "{}", err)
        }
    }
}

impl_debug_default!(SpikingNeuralNetworksError);

macro_rules! impl_from_error_default {
    ($error_name:ident, $variant_name:ident) => {
        impl From<$error_name> for SpikingNeuralNetworksError {
            fn from(err: $error_name) -> SpikingNeuralNetworksError {
                SpikingNeuralNetworksError::$variant_name(err)
            }
        }
    };
}

impl_from_error_default!(TimeSeriesProcessingError, SeriesProcessingRelatedError);
impl_from_error_default!(GeneticAlgorithmError, GeneticAlgorithmRelatedErrors);
impl_from_error_default!(GraphError, GraphRelatedError);
impl_from_error_default!(LatticeNetworkError, LatticeNetworkRelatedError);
impl_from_error_default!(PatternError, PatternRelatedError);
impl_from_error_default!(AgentError, AgentRelatedError);
