//! CPU time of the calling thread, for the optional `--cpu` column.
//!
//! `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` is declared here by hand: the
//! harness depends on nothing but symplex and must build against any tree.
//! On other targets there is no such clock and the column is `-`.

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
mod imp {
    /// `struct timespec` of 64-bit Linux (`time_t` and `long` are 64 bits).
    #[repr(C)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }

    /// `CLOCK_THREAD_CPUTIME_ID` of Linux (`<time.h>`).
    const CLOCK_THREAD_CPUTIME_ID: i32 = 3;

    unsafe extern "C" {
        fn clock_gettime(clock_id: i32, tp: *mut Timespec) -> i32;
    }

    /// Nanoseconds of CPU time the calling thread has used so far.
    pub fn thread_ns() -> Option<u64> {
        let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: `ts` is valid, writable storage with the layout of the C
        // `timespec` of this target, and `clock_gettime` only writes it.
        let rc = unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &raw mut ts) };
        if rc != 0 {
            return None;
        }
        let secs = u64::try_from(ts.tv_sec).ok()?;
        let nanos = u64::try_from(ts.tv_nsec).ok()?;
        secs.checked_mul(1_000_000_000)?.checked_add(nanos)
    }
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
mod imp {
    /// No thread CPU clock is declared for this target.
    pub fn thread_ns() -> Option<u64> {
        None
    }
}

pub use imp::thread_ns;
