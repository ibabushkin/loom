//! Mock implementation of `std::atomic`.

mod arc;
mod causal;
pub mod atomic;
mod condvar;
mod mutex;

pub use self::arc::Arc;
pub use self::causal::CausalCell;
pub use self::condvar::{Condvar, WaitTimeoutResult};
pub use self::mutex::{Mutex, MutexGuard};

pub use std::sync::{LockResult, TryLockResult};
