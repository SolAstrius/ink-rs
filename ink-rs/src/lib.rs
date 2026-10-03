//! Native inference for the Latin/Cyrillic IndyLSTM handwriting models.
//!
//! Inputs are time-major, row-major `[T, 10]` curve features. The model reader
//! loads the existing TFLite files directly; no interpreter or Python runtime
//! is needed. `features` fits raw strokes into curves; `decoder` provides
//! CompactLmFst-backed CTC beam search for the matching language packs.

pub mod decoder;
pub mod features;
mod flatbuffer;
mod kernels;
mod model;
pub mod npy;
mod session;
pub mod spec;

pub use kernels::Backend;
pub use model::Model;
pub use session::Session;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self(value.to_string())
    }
}

pub(crate) fn error(message: impl Into<String>) -> Error {
    Error(message.into())
}
