//! A counting global allocator with a hard cap on live heap bytes.
//!
//! When the requested live bytes exceed the cap, the process prints
//! `golden: OOM` to stderr and aborts; the driver records the running
//! `(case, op)` as `OOM` and restarts at the next op.  The count is of bytes
//! *requested* (not RSS), summed over all threads.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

pub struct Capped;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static LIMIT: AtomicUsize = AtomicUsize::new(4 << 30);
static FIRED: AtomicBool = AtomicBool::new(false);

pub fn set_limit_mb(mb: u64) {
    LIMIT.store(usize::try_from(mb).unwrap_or(usize::MAX / 2) << 20, Relaxed);
}

/// Called with the new live total; aborts once (re-entrant allocations made
/// while reporting pass through).
#[cold]
#[inline(never)]
fn over_cap() {
    if !FIRED.swap(true, Relaxed) {
        use std::io::Write;
        let _ = std::io::stderr().write_all(b"golden: OOM: live heap above the --mem-mb cap\n");
        std::process::abort();
    }
}

#[inline]
fn grow(by: usize) {
    let live = LIVE.fetch_add(by, Relaxed).wrapping_add(by);
    if live > LIMIT.load(Relaxed) {
        over_cap();
    }
}

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        grow(l.size());
        let p = unsafe { System.alloc(l) };
        if p.is_null() {
            LIVE.fetch_sub(l.size(), Relaxed);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        grow(l.size());
        let p = unsafe { System.alloc_zeroed(l) };
        if p.is_null() {
            LIVE.fetch_sub(l.size(), Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let old = l.size();
        if new_size > old {
            grow(new_size - old);
        }
        let q = unsafe { System.realloc(p, l, new_size) };
        if q.is_null() {
            if new_size > old {
                LIVE.fetch_sub(new_size - old, Relaxed);
            }
        } else if new_size < old {
            LIVE.fetch_sub(old - new_size, Relaxed);
        }
        q
    }
}
