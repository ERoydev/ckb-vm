//! Runtime tap that logs 16/32/64-bit simulated loads and stores whose
//! address is not naturally aligned to the access width.
//!
//! The tap is invoked from the Rust interpreter's load/store call sites in
//! `crate::instructions::common`, which is the single choke point above both
//! `FlatMemory` and `SparseMemory`. The asm path has its own lbu/sb
//! decomposition and is intentionally not covered here.

use core::sync::atomic::{AtomicU64, Ordering};

pub static UNALIGNED_LD16: AtomicU64 = AtomicU64::new(0);
pub static UNALIGNED_LD32: AtomicU64 = AtomicU64::new(0);
pub static UNALIGNED_LD64: AtomicU64 = AtomicU64::new(0);
pub static UNALIGNED_ST16: AtomicU64 = AtomicU64::new(0);
pub static UNALIGNED_ST32: AtomicU64 = AtomicU64::new(0);
pub static UNALIGNED_ST64: AtomicU64 = AtomicU64::new(0);

#[inline(always)]
fn low_bits(width_bytes: u64, addr: u64) -> u64 {
    addr & (width_bytes - 1)
}

#[inline(always)]
pub fn check_load(width_bytes: u64, addr: u64, pc: u64) {
    let low = low_bits(width_bytes, addr);
    if low == 0 {
        return;
    }
    let counter = match width_bytes {
        2 => &UNALIGNED_LD16,
        4 => &UNALIGNED_LD32,
        8 => &UNALIGNED_LD64,
        _ => return,
    };
    counter.fetch_add(1, Ordering::Relaxed);
    #[cfg(feature = "std")]
    {
        std::eprintln!(
            "UNALIGNED LD{} @ addr=0x{:x}  pc=0x{:x}  (low_bits={})",
            width_bytes * 8,
            addr,
            pc,
            low
        );
    }
    #[cfg(not(feature = "std"))]
    let _ = pc;
}

#[inline(always)]
pub fn check_store(width_bytes: u64, addr: u64, pc: u64) {
    let low = low_bits(width_bytes, addr);
    if low == 0 {
        return;
    }
    let counter = match width_bytes {
        2 => &UNALIGNED_ST16,
        4 => &UNALIGNED_ST32,
        8 => &UNALIGNED_ST64,
        _ => return,
    };
    counter.fetch_add(1, Ordering::Relaxed);
    #[cfg(feature = "std")]
    {
        std::eprintln!(
            "UNALIGNED ST{} @ addr=0x{:x}  pc=0x{:x}  (low_bits={})",
            width_bytes * 8,
            addr,
            pc,
            low
        );
    }
    #[cfg(not(feature = "std"))]
    let _ = pc;
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Summary {
    pub ld16: u64,
    pub ld32: u64,
    pub ld64: u64,
    pub st16: u64,
    pub st32: u64,
    pub st64: u64,
}

impl Summary {
    pub fn total(&self) -> u64 {
        self.ld16 + self.ld32 + self.ld64 + self.st16 + self.st32 + self.st64
    }
}

pub fn snapshot() -> Summary {
    Summary {
        ld16: UNALIGNED_LD16.load(Ordering::Relaxed),
        ld32: UNALIGNED_LD32.load(Ordering::Relaxed),
        ld64: UNALIGNED_LD64.load(Ordering::Relaxed),
        st16: UNALIGNED_ST16.load(Ordering::Relaxed),
        st32: UNALIGNED_ST32.load(Ordering::Relaxed),
        st64: UNALIGNED_ST64.load(Ordering::Relaxed),
    }
}

pub fn reset() {
    UNALIGNED_LD16.store(0, Ordering::Relaxed);
    UNALIGNED_LD32.store(0, Ordering::Relaxed);
    UNALIGNED_LD64.store(0, Ordering::Relaxed);
    UNALIGNED_ST16.store(0, Ordering::Relaxed);
    UNALIGNED_ST32.store(0, Ordering::Relaxed);
    UNALIGNED_ST64.store(0, Ordering::Relaxed);
}

#[cfg(feature = "std")]
pub fn print_summary() {
    let s = snapshot();
    std::eprintln!("=== ckb-vm unaligned access summary ===");
    std::eprintln!("  LD16: {}", s.ld16);
    std::eprintln!("  LD32: {}", s.ld32);
    std::eprintln!("  LD64: {}", s.ld64);
    std::eprintln!("  ST16: {}", s.st16);
    std::eprintln!("  ST32: {}", s.st32);
    std::eprintln!("  ST64: {}", s.st64);
    std::eprintln!("  TOTAL: {}", s.total());
}

/// RAII helper: prints the summary when dropped. Useful for guaranteeing a
/// summary fires even on early return or panic unwind.
#[cfg(feature = "std")]
pub struct SummaryGuard;

#[cfg(feature = "std")]
impl Drop for SummaryGuard {
    fn drop(&mut self) {
        print_summary();
    }
}
