use super::Atomic;

use std::sync::atomic::Ordering;

/// Mock implementation of `std::sync::atomic::AtomicPtr`.
#[derive(Debug)]
pub struct AtomicPtr<T>(Atomic<*mut T>);

impl<T> AtomicPtr<T> {
    /// Creates a new instance of `AtomicPtr`.
    pub fn new(v: *mut T) -> AtomicPtr<T> {
        AtomicPtr(Atomic::new(v))
    }

    /// Loads a value from the pointer.
    pub fn load(&self, order: Ordering) -> *mut T {
        self.0.load(order)
    }

    /// Stores a value into the pointer.
    pub fn store(&self, val: *mut T, order: Ordering) {
        self.0.store(val, order)
    }

    /// Stores a value into the pointer, returning the previous value.
    pub fn swap(&self, val: *mut T, order: Ordering) -> *mut T {
        self.0.swap(val, order)
    }

    /// Stores a value into the pointer if the current value is the same as the `current` value.
    pub fn compare_and_swap(&self, current: *mut T, new: *mut T, order: Ordering) -> *mut T {
        self.0.compare_and_swap(current, new, order)
    }

    /// Stores a value into the pointer if the current value is the same as the `current` value.
    pub fn compare_exchange(
        &self,
        current: *mut T,
        new: *mut T,
        success: Ordering,
        failure: Ordering
    ) -> Result<*mut T, *mut T>
    {
        self.0.compare_exchange(current, new, success, failure)
    }
}
