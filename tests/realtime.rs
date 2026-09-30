use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
struct CountAlloc;
static ENABLED: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;
#[test]
fn processing_and_gap_reset_do_not_allocate() {
    let (mut input, mut worker) = pulseweave::stream::audio_channel(48000, 1024).unwrap();
    let audio = [0.1; 2048];
    let mut frames = 0;
    let mut evidence = pulseweave::period::PeriodEvidence::new();
    let mut decoder = pulseweave::decoder::PeriodDecoder::new();
    let mut phase = pulseweave::decoder::PhaseDecoder::new();
    let accents: [f64; 86] = std::array::from_fn(|i| (i as f64 * 0.31).sin());
    let (mut meter_input, mut meter_worker) =
        pulseweave::stream::meter_channel(48000, 2048).unwrap();
    let mut updates = 0;
    ENABLED.store(true, Ordering::SeqCst);
    for _ in 0..80 {
        meter_input.push(&audio);
        meter_worker.poll(2048, |_, _| updates += 1);
    }
    meter_input.push(&audio);
    meter_input.push(&audio);
    meter_worker.poll(2048, |_, _| {});
    meter_input.push(&audio);
    meter_worker.poll(2048, |_, _| {});
    for step in 1..=3 {
        let candidates = evidence.process([&accents; 4]);
        let periods = decoder.process(candidates);
        phase.process(&evidence, periods, step * 86);
    }
    phase.reset();
    decoder.reset();
    evidence.reset();
    for _ in 0..16 {
        input.push(&audio[..1024]);
        worker.poll(2048, |_, _| frames += 1);
    }
    for _ in 0..16 {
        input.push(&audio); // Forces an overrun and a reset on the following pass.
        worker.poll(2048, |_, _| frames += 1);
    }
    ENABLED.store(false, Ordering::SeqCst);
    assert_eq!(COUNT.load(Ordering::SeqCst), 0);
    assert!(frames > 0);
    assert!(updates > 0);
}
