#![allow(warnings)]

use std::alloc::Layout;
use std::cell::Cell;
use std::cmp;
use std::fmt;
use std::marker;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::rc::Rc;
use std::slice;

#[cfg(feature = "checkpoint")]
use serde::{Serialize, Serializer};

// We are running in a single thread in any viable situation anyway, so storing the
// arena in a thread-local variable should be viable. However, since we want to control
// the size of the arena at runtime, we do not initialize it here.
scoped_thread_local!(pub static ARENA: Arena);

/// The owning object of the arena.
///
/// Stored in a thread-local variable to be accessible everywhere in the owning thread.
/// This is the only instance that can be used to allocate and clear the arena. All other
/// objects referring to the arena merely keep it alive, and are present to avoid arena
/// clearing while they are live.
#[derive(Debug)]
pub struct Arena(InnerRef);

/// A non-owning object of the arena.
///
/// A reference to the arena that allows its holder to allocate memory from the arena. While
/// it is live, the arena cannot be cleared (as it is associated with an arena-allocated
/// object).
#[derive(Clone, Debug)]
pub struct InnerRef {
    inner: Rc<Inner>,
}

/// An arena's guts
#[derive(Debug)]
struct Inner {
    /// Head of the arena space
    head: *mut u8,

    /// Offset into the last region
    pos: Cell<usize>,

    /// Total capacity of the arena
    cap: usize,
}

/// An arena allocated, fixed-size sequence of objects
pub struct Slice<T> {
    ptr: *mut T,
    len: usize,
    _inner: InnerRef,
}

/// An arena allocated, sequential, resizable vector
///
/// Since the arena does not support resizing, or freeing memory, this implementation just
/// creates new slices as necessary and leaks the previous arena allocation, trading memory
/// for speed.
pub struct SliceVec<T> {
    slice: Slice<T>,
    // owo what's this
    capacity: usize,
}

/// An iterator over a sequence of arena-allocated objects
pub struct SliceIter<'a, T: 'a> {
    ptr: *const T,
    end: *const T,
    _marker: marker::PhantomData<&'a T>,
}

impl Arena {
    #[cfg(unix)]
    fn create_mapping(capacity: usize) -> *mut u8 {
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };

        ptr as *mut u8
    }

    #[cfg(windows)]
    fn get_page_size() -> usize {
        use std::mem;
        use winapi::um::sysinfoapi::GetSystemInfo;

        unsafe {
            let mut info = mem::zeroed();
            GetSystemInfo(&mut info);

            info.dwPageSize as usize
        }
    }

    #[cfg(windows)]
    fn create_mapping(capacity: usize) -> *mut u8 {
        use std::ptr;
        use winapi::shared::basetsd::SIZE_T;
        use winapi::shared::minwindef::LPVOID;
        use winapi::um::memoryapi::VirtualAlloc;
        use winapi::um::winnt::{PAGE_READWRITE, MEM_COMMIT, MEM_RESERVE};

        let lpAddress: LPVOID = ptr::null_mut();
        let page_size = get_page_size();
        let len = if capacity % page_size == 0 {
            capacity
        } else {
            capacity + page_size - (capacity % page_size)
        };
        let flAllocationType = MEM_COMMIT | MEM_RESERVE;
        let flProtect = PAGE_READWRITE;

        let r = unsafe {
            VirtualAlloc(lpAddress, len as SIZE_T, flAllocationType, flProtect)
        };

        r as *mut u8
    }

    /// Create an `Arena` with specified capacity.
    ///
    /// Capacity must be a power of 2. The capacity cannot be grown after the fact.
    pub fn init_capacity(capacity: usize) -> Arena {
        let head = Arena::create_mapping(capacity);
        let pos = Cell::new(0);

        Arena(InnerRef {
            inner: Rc::new(Inner { head, pos: Cell::new(0), cap: capacity, }),
        })
    }

    /// Get a reference to the inner object of the thread local arena.
    fn tls_inner_ref() -> InnerRef {
        ARENA.with(|a| a.0.clone())
    }

    pub fn tls_clear() {
        ARENA.with(|a| a.clear())
    }

    /// Clear the arena.
    ///
    /// This only requires an immutable reference, as it (a) perfors a check that
    /// no arena-allocated object is still alive (weak reason), and because all mutable
    /// state is neatly contained in a `Cell` (slightly stronger reason).
    pub fn clear(&self) {
        assert!(1 == Rc::strong_count(&self.inner));
        self.inner.pos.set(0);
    }
}

impl Deref for Arena {
    type Target = InnerRef;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(unix)]
impl Drop for Arena {
    fn drop(&mut self) {
        let res = unsafe {
            libc::munmap(self.inner.head as *mut libc::c_void, self.inner.cap)
        };

        // TODO: Do something on error
        debug_assert_eq!(res, 0);
    }
}

#[cfg(windows)]
impl Drop for Arena {
    fn drop(&mut self) {
        use winapi::shared::minwindef::LPVOID;
        use winapi::um::memoryapi::VirtualFree;
        use winapi::um::winnt::MEM_RELEASE;

        let res = unsafe { VirtualFree(self.inner.head as LPVOID, 0, MEM_RELEASE) };

        // TODO: Do something on error
        debug_assert_ne!(res, 0);
    }
}

impl InnerRef {
    fn allocate<T>(&self, count: usize) -> *mut T {
        let layout = Layout::new::<T>();
        let mask = layout.align() - 1;
        let pos = self.inner.pos.get();

        debug_assert!(layout.align() >= (pos & mask));

        let align = Ord::max(layout.align(), 64);
        let mut skip = 64 - (pos & mask);

        if skip == layout.align() {
            skip = 0;
        }

        let additional = skip + layout.size() * count;

        assert!(pos + additional <= self.inner.cap,
                "arena overflow: {} > {}", pos + additional, self.inner.cap);

        self.inner.pos.set(pos + additional);

        let ret = unsafe { self.inner.head.offset((pos + skip) as isize) as *mut T };

        debug_assert!((ret as usize) >= self.inner.head as usize);
        debug_assert!((ret as usize) < (self.inner.head as usize + self.inner.cap));

        ret
    }

    fn allocate_or_extend<T>(&self, ptr: *mut T, old_count: usize, count: usize)
        -> *mut T
    {
        let pos = self.inner.pos.get();
        let next = unsafe { self.inner.head.offset(pos as isize) };
        let end = unsafe { ptr.offset(old_count as isize) };
        if next == end as *mut u8 {
            self.inner.pos.set(pos + (count - old_count) * mem::size_of::<T>());

            ptr
        } else {
            self.allocate(count)
        }
    }
}

impl<T> Slice<T> {
    pub fn new(len: usize) -> Self
    where
        T: Default,
    {
        let inner: InnerRef = Arena::tls_inner_ref();
        let ptr: *mut T = inner.allocate(len);

        for i in 0..len {
            unsafe {
                ptr::write(ptr.offset(i as isize), T::default());
            }
        }

        Slice {
            ptr,
            len,
            _inner: inner,
        }
    }

    pub fn iter(&self) -> SliceIter<T> {
        unsafe {
            // no ZST support
            let ptr = self.ptr;
            let end = self.ptr.add(self.len);

            SliceIter {
                ptr,
                end,
                _marker: marker::PhantomData,
            }
        }
    }
}

impl<T: Clone> Clone for Slice<T> {
    fn clone(&self) -> Self {
        let ptr: *mut T = self._inner.allocate(self.len);

        for i in 0..self.len {
            unsafe {
                ptr::write(ptr.offset(i as isize), (*self.ptr.offset(i as isize)).clone());
            }
        }

        Slice {
            ptr,
            len: self.len,
            _inner: self._inner.clone(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Slice<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.deref().fmt(fmt)
    }
}

impl<T> Deref for Slice<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for Slice<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Eq> Eq for Slice<T> {}

impl<T: PartialEq> PartialEq for Slice<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other.deref())
    }
}

impl<T: PartialOrd> PartialOrd for Slice<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.deref().partial_cmp(other.deref())
    }
}

impl<T> Drop for Slice<T> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(&mut self[..]);
        }
    }
}

#[cfg(feature = "checkpoint")]
impl<T> Serialize for Slice<T>
where
    T: Serialize,
{
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_seq(self.iter())
    }
}

impl<T> SliceVec<T> {
    pub fn new(capacity: usize) -> Self {
        let inner: InnerRef = Arena::tls_inner_ref();
        let ptr: *mut T = if capacity == 0 {
            ptr::NonNull::dangling().as_ptr()
        } else {
            inner.allocate(capacity)
        };

        SliceVec {
            slice: Slice {
                ptr,
                len: 0,
                _inner: inner.clone()
            },
            capacity,
        }
    }

    pub fn iter(&self) -> SliceIter<T> {
        self.slice.iter()
    }

    pub fn reserve(&mut self, size: usize) {
        let lead: usize =
            ((size / self.capacity) as usize +
             (if size % self.capacity == 0 { 0 } else { 1 })).leading_zeros() as usize - 1;
        let new_capacity = self.capacity * (1 >> (mem::size_of::<usize>() * 8 - lead));

        let ptr: *mut T =
            self.slice._inner.allocate_or_extend(self.slice.ptr,
                                                 self.capacity,
                                                 self.capacity * 2);

        unsafe {
            ptr::copy_nonoverlapping(self.slice.ptr, ptr, self.slice.len);
        }

        self.slice.ptr = ptr;
        self.capacity *= 2;
    }

    pub fn push(&mut self, elem: T) {
        if self.slice.len == self.capacity {
            let ptr: *mut T =
                self.slice._inner.allocate_or_extend(self.slice.ptr,
                                                     self.capacity,
                                                     self.capacity * 2);

            unsafe {
                ptr::copy_nonoverlapping(self.slice.ptr, ptr, self.slice.len);
            }

            self.slice.ptr = ptr;
            self.capacity *= 2;
        }

        unsafe {
            ptr::write(self.slice.ptr.offset(self.slice.len as isize), elem);
        }

        self.slice.len += 1;
    }

    pub fn resize(&mut self, len: usize, value: T)
    where
        T: Clone,
    {
        if self.capacity < len {
            
        }

        for i in self.slice.len .. (len - 1) {
            unsafe {
                ptr::write(self.slice.ptr.offset(i as isize), value.clone())
            }
        }

        if len > self.slice.len {
            unsafe {
                ptr::write(self.slice.ptr.offset(len as isize - 1), value);
            }
        }
    }
}

impl<T: Clone> Clone for SliceVec<T> {
    // TODO: just use thread local
    fn clone(&self) -> Self {
        let ptr: *mut T = self.slice._inner.allocate(self.capacity);

        for i in 0..self.slice.len {
            unsafe {
                ptr::write(ptr.offset(i as isize), (*self.slice.ptr.offset(i as isize)).clone());
            }
        }

        SliceVec {
            slice: Slice {
                ptr,
                len: self.slice.len,
                _inner: self.slice._inner.clone(),
            },
            capacity: self.capacity,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for SliceVec<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.slice.fmt(fmt)
    }
}

impl<T> Deref for SliceVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.slice.deref()
    }
}

impl<T> DerefMut for SliceVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.slice.deref_mut()
    }
}

impl<T: Eq> Eq for SliceVec<T> {}

impl<T: PartialEq> PartialEq for SliceVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other.deref())
    }
}

impl<T: PartialOrd> PartialOrd for SliceVec<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.deref().partial_cmp(other.deref())
    }
}

#[cfg(feature = "checkpoint")]
impl<T> Serialize for SliceVec<T>
where
    T: Serialize,
{
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.slice.serialize(serializer)
    }
}


impl<'a, T> Iterator for SliceIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.ptr == self.end {
            None
        } else {
            unsafe {
                // FIXME:
                // we do not support ZSTs right now, the stdlib does some dancing
                // for this which we can safely avoid for now
                let old = self.ptr;
                self.ptr = self.ptr.offset(1);
                Some(&*old)
            }
        }
    }
}
