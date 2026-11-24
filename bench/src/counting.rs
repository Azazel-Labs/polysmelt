//! A counting global allocator: the peak heap during a load and the heap its result keeps, and an
//! optional ceiling past which allocations fail (the resilience run's guard against a hostile size
//! field taking the machine's memory).

#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

pub struct Counting;

pub static CURRENT: AtomicUsize = AtomicUsize::new(0);
pub static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Allocations that would take the heap past this fail (return null).
pub static LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);

fn grow(by: usize) -> bool {
    let now = CURRENT.fetch_add(by, Relaxed) + by;
    if now > LIMIT.load(Relaxed) {
        CURRENT.fetch_sub(by, Relaxed);
        return false;
    }
    PEAK.fetch_max(now, Relaxed);
    true
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !grow(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc(layout) };
        if p.is_null() {
            CURRENT.fetch_sub(layout.size(), Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !grow(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc_zeroed(layout) };
        if p.is_null() {
            CURRENT.fetch_sub(layout.size(), Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        CURRENT.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        if new > layout.size() && !grow(new - layout.size()) {
            return std::ptr::null_mut();
        }
        let q = unsafe { System.realloc(p, layout, new) };
        if q.is_null() {
            if new > layout.size() {
                CURRENT.fetch_sub(new - layout.size(), Relaxed);
            }
        } else if new < layout.size() {
            CURRENT.fetch_sub(layout.size() - new, Relaxed);
        }
        q
    }
}
