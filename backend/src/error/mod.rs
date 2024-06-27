use std::fmt::{Display, Debug, Formatter, Result};


/// Error set for potential graph errors
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

impl Debug for GraphError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "file: {}, line: {}, error: {}", file!(), line!(), self)
    }
}

/// Error set for potential lattice network errors
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

impl Debug for LatticeNetworkError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "file: {}, line: {}, error: {}", file!(), line!(), self)
    }
}

/// A set of errors for potential pattern errors
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

impl Debug for PatternError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "file: {}, line: {}, error: {}", file!(), line!(), self)
    }
}

/// A set of errors that may occur when using the library
pub enum SpikingNeuralNetworksError {
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
            SpikingNeuralNetworksError::GraphRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksError::LatticeNetworkRelatedError(err) => write!(f, "{}", err),
            SpikingNeuralNetworksError::PatternRelatedError(err) => write!(f, "{}", err),
        }
    }
}

impl Debug for SpikingNeuralNetworksError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "file: {}, line: {}, error: {}", file!(), line!(), self)
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
