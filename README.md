# pulseweave

A synchronous Rust library for streaming bar, beat, and tatum analysis of mono
audio. It uses rubato for resampling, rtrb for an optional bounded audio handoff,
and no async runtime.

## Use

```rust
use pulseweave::stream::meter_channel;

fn main() -> Result<(), pulseweave::stream::StreamError> {
    let (mut input, mut worker) = meter_channel(48_000, 8192)?;
    // Move input into your audio callback. Supply mono samples.
    let dropped = input.push(&[0.0_f32; 512]);
    assert_eq!(dropped, 0);
    // Poll from an ordinary worker thread; the library does not spawn one.
    worker.poll(8192, |generation, update| {
        let bpm = 60.0 / update.meter.periods.beat;
        println!("generation {generation}: {bpm:.1} BPM at {:.3}s", update.meter.beat_time);
    });
    Ok(())
}
```

For synchronous processing without a queue, use `stream::MeterStream::new(rate)`
and `process(&[f32], callback)`. Supported input rates are 8–192 kHz.
`tracker::Tracker` accepts mono f64 directly at 44.1 kHz.
`stream::FeatureStream` and `stream::audio_channel` expose the intermediate spectral accents.
Constructors return `StreamError`, with separate variants for unsupported sample
rates, zero queue capacity, and resampler initialization failures.

## Processing and timing

Construct streams before entering the audio callback: construction allocates the
analysis and resampling buffers. Processing and reset use preallocated storage;
your result callback is responsible for any allocations it performs. For an audio
callback, use `AudioInput::push` and run the analysis worker on a separate thread.

Meter updates arrive roughly twice per second after the initial analysis window,
so a single short block may produce no update. Periods and meter timestamps are
in seconds on the analysis timeline. `TrackerUpdate::audio_samples` counts samples
at the internal 44.1 kHz rate, including when input is resampled. Subtract
`resampler_delay() / 44_100.0` from analysis timestamps to account for resampling
delay when mapping back to source audio.

When the queue fills, `push` returns the number of dropped samples. The worker
resets analysis at the next detected gap and increments the generation; timestamps
restart within that generation. Nonfinite input samples become zero. Partial
blocks remain buffered; there is no end-of-stream flush API.

The lower-level `dsp`, `onset`, `period`, and `decoder` modules expose the analysis
stages. `pulse::reconstruct` interpolates pulse times from a history of anchors.

## License

Licensed under either the [MIT license](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option.
