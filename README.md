# pulseweave

A synchronous Rust library for streaming bar, beat, and tatum analysis of mono
audio. It uses rubato for resampling, rtrb for an optional bounded audio handoff,
and no async runtime.

## Use

```rust
use pulseweave::stream::meter_channel;

let (mut input, mut worker) = meter_channel(48_000, 8192)?;
// Move input into your audio callback. Supply mono samples.
let dropped = input.push(&[0.0_f32; 512]);
// Poll from an ordinary worker thread; the library does not spawn one.
worker.poll(8192, |generation, update| {
    let bpm = 60.0 / update.meter.periods.beat;
    let bar_seconds = update.meter.periods.bar;
    let atom_seconds = update.meter.periods.tatum;
    let beat_time = update.meter.beat_time;
});
# Ok::<(), pulseweave::stream::StreamError>(())
```

For synchronous processing without a queue, use `stream::MeterStream::new(rate)`
and `process(&[f32], callback)`. Supported input rates are 8–192 kHz.
`tracker::Tracker` accepts mono f64 directly at 44.1 kHz.
`FeatureStream` and `audio_channel` expose the intermediate spectral accents.
Constructors return `StreamError`, with separate variants for unsupported sample
rates, zero queue capacity, and resampler initialization failures.