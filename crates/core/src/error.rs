use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReedError {
    #[error("Vector error: {0}")]
    Vector(String),

    #[error("Basis error: {0}")]
    Basis(String),

    #[error("ElemRestriction error: {0}")]
    ElemRestriction(String),

    #[error("QFunction error: {0}")]
    QFunction(String),

    #[error("Operator error: {0}")]
    Operator(String),

    #[error("Backend not supported: {0}")]
    BackendNotSupported(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

pub type ReedResult<T> = Result<T, ReedError>;
