//! Synchronous resampling and an optional bounded SPSC audio handoff.
use crate::onset::{AccentFrame, OnsetAnalyzer, SAMPLE_RATE};
use crate::tracker::{Tracker, TrackerUpdate};
use rtrb::{Consumer, Producer, RingBuffer};
use rubato::{FftFixedIn, Resampler};

/// Failure to construct an audio stream or queue.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StreamError {
    /// The input sample rate is outside the supported range.
    #[error("sample rate {0} Hz is outside the supported range 8000..=192000 Hz")]
    UnsupportedSampleRate(usize),
    /// A queue needs at least one sample of storage.
    #[error("audio queue capacity must be positive")]
    ZeroCapacity,
    /// The resampler could not be initialized.
    #[error("could not initialize resampler: {0}")]
    Resampler(#[from] rubato::ResamplerConstructionError),
}

const CHUNK: usize = 512;

/// Mono input adapter. All resampler buffers are allocated at construction.
struct AudioAdapter {
    resampler: Option<FftFixedIn<f64>>,
    input: [f64; CHUNK],
    filled: usize,
    output: Vec<f64>,
}
impl AudioAdapter {
    /// Create an adapter for mono input at 8–192 kHz; reject rates outside that range.
    /// Allocates all resampling and analysis buffers.
    fn new(sample_rate: usize) -> Result<Self, StreamError> {
        if !(8_000..=192_000).contains(&sample_rate) {
            return Err(StreamError::UnsupportedSampleRate(sample_rate));
        }
        let resampler = if sample_rate == SAMPLE_RATE {
            None
        } else {
            Some(FftFixedIn::new(sample_rate, SAMPLE_RATE, CHUNK, 2, 1)?)
        };
        let output = vec![0.; resampler.as_ref().map_or(0, Resampler::output_frames_max)];
        Ok(Self {
            resampler,
            input: [0.; CHUNK],
            filled: 0,
            output,
        })
    }
    /// Resampler delay in 44.1 kHz output samples. Does not include onset latency.
    fn resampler_delay(&self) -> usize {
        self.resampler.as_ref().map_or(0, Resampler::output_delay)
    }
    /// Clear buffered input and analysis state; restart the analysis timeline.
    fn reset(&mut self) {
        self.filled = 0;
        self.input.fill(0.);
        if let Some(r) = &mut self.resampler {
            r.reset();
        }
    }
    /// Consume mono input and emit resampled audio blocks.
    /// Nonfinite samples become zero. Partial resampling blocks remain buffered.
    fn process(&mut self, samples: &[f32], mut emit: impl FnMut(&[f64])) {
        for &sample in samples {
            self.input[self.filled] = if sample.is_finite() {
                sample as f64
            } else {
                0.0
            };
            self.filled += 1;
            if self.filled != CHUNK {
                continue;
            }
            if let Some(r) = &mut self.resampler {
                // Fixed channel count, input length and preallocated output size
                // make buffer validation an invariant established by construction.
                let (_, count) = r
                    .process_into_buffer(&[&self.input[..]], &mut [&mut self.output[..]], None)
                    .expect("fixed resampler buffers must remain valid");
                emit(&self.output[..count]);
            } else {
                emit(&self.input);
            }
            self.filled = 0;
        }
    }
}

/// Synchronous mono f32 adapter with preallocated rubato resampling.
pub struct FeatureStream {
    adapter: AudioAdapter,
    onset: OnsetAnalyzer,
}
impl FeatureStream {
    /// Create an adapter for mono input at 8–192 kHz; reject rates outside that range.
    /// Allocates all resampling and analysis buffers.
    pub fn new(sample_rate: usize) -> Result<Self, StreamError> {
        Ok(Self {
            adapter: AudioAdapter::new(sample_rate)?,
            onset: OnsetAnalyzer::new(),
        })
    }
    /// Resampling delay in 44.1 kHz output samples; subtract from timestamps
    /// when mapping the analysis timeline back to the source audio.
    pub fn resampler_delay(&self) -> usize {
        self.adapter.resampler_delay()
    }
    /// Clear buffered input and analysis state; restart the analysis timeline.
    pub fn reset(&mut self) {
        self.adapter.reset();
        self.onset.reset();
    }
    /// Consume mono input and emit spectral accent frames.
    /// Nonfinite samples become zero. Partial resampling blocks remain buffered.
    pub fn process(&mut self, samples: &[f32], mut emit: impl FnMut(AccentFrame)) {
        let engine = &mut self.onset;
        self.adapter
            .process(samples, |audio| engine.process(audio, &mut emit));
    }
}
/// Synchronous mono f32 adapter with preallocated rubato resampling.
pub struct MeterStream {
    adapter: AudioAdapter,
    tracker: Tracker,
}
impl MeterStream {
    /// Create an adapter for mono input at 8–192 kHz; reject rates outside that range.
    /// Allocates all resampling and analysis buffers.
    pub fn new(sample_rate: usize) -> Result<Self, StreamError> {
        Ok(Self {
            adapter: AudioAdapter::new(sample_rate)?,
            tracker: Tracker::new(),
        })
    }
    /// Resampling delay in 44.1 kHz output samples; subtract from timestamps
    /// when mapping the analysis timeline back to the source audio.
    pub fn resampler_delay(&self) -> usize {
        self.adapter.resampler_delay()
    }
    /// Clear buffered input and analysis state; restart the analysis timeline.
    pub fn reset(&mut self) {
        self.adapter.reset();
        self.tracker.reset();
    }
    /// Consume mono input and emit meter estimates.
    /// Nonfinite samples become zero. Partial resampling blocks remain buffered.
    pub fn process(&mut self, samples: &[f32], mut emit: impl FnMut(TrackerUpdate)) {
        let engine = &mut self.tracker;
        self.adapter
            .process(samples, |audio| engine.process(audio, &mut emit));
    }
}
#[derive(Clone, Copy)]
struct Sample {
    sequence: u64,
    value: f32,
}

/// Audio-callback endpoint. Never waits, allocates, or runs analysis.
pub struct AudioInput {
    producer: Producer<Sample>,
    sequence: u64,
}
impl AudioInput {
    /// Returns the number of dropped samples when the bounded queue is full.
    /// Sequence numbers preserve discontinuities so the worker can reset safely.
    pub fn push(&mut self, mono: &[f32]) -> usize {
        let mut dropped = 0;
        for &value in mono {
            if self
                .producer
                .push(Sample {
                    sequence: self.sequence,
                    value,
                })
                .is_err()
            {
                dropped += 1;
            }
            self.sequence = self.sequence.wrapping_add(1);
        }
        dropped
    }
}

struct AudioQueue {
    consumer: Consumer<Sample>,
    expected: u64,
    generation: u64,
}

impl AudioQueue {
    fn new(capacity: usize) -> (AudioInput, Self) {
        let (producer, consumer) = RingBuffer::new(capacity);
        (
            AudioInput {
                producer,
                sequence: 0,
            },
            Self {
                consumer,
                expected: 0,
                generation: 0,
            },
        )
    }

    fn poll(&mut self, limit: usize, mut consume: impl FnMut(f32, u64, bool)) -> usize {
        let mut consumed = 0;
        while consumed < limit {
            let Ok(sample) = self.consumer.pop() else {
                break;
            };
            let discontinuity = sample.sequence != self.expected;
            if discontinuity {
                self.generation = self.generation.wrapping_add(1);
            }
            self.expected = sample.sequence.wrapping_add(1);
            consume(sample.value, self.generation, discontinuity);
            consumed += 1;
        }
        consumed
    }
}

/// Worker endpoint. Call `poll` from an ordinary thread; no async runtime needed.
pub struct FeatureWorker {
    queue: AudioQueue,
    stream: FeatureStream,
}
impl FeatureWorker {
    /// Resampling delay in 44.1 kHz analysis samples.
    pub fn resampler_delay(&self) -> usize {
        self.stream.resampler_delay()
    }
    /// Process at most `limit` queued samples. The generation changes after a gap;
    /// accent indices restart at zero in the new generation.
    pub fn poll(&mut self, limit: usize, mut emit: impl FnMut(u64, AccentFrame)) -> usize {
        let stream = &mut self.stream;
        self.queue.poll(limit, |sample, generation, discontinuity| {
            if discontinuity {
                stream.reset();
            }
            stream.process(&[sample], |frame| emit(generation, frame));
        })
    }
}

/// Create a bounded mono audio queue and a worker that emits spectral accents.
/// Rejects zero capacity or input rates outside 8–192 kHz. Capacity is in samples.
pub fn audio_channel(
    sample_rate: usize,
    capacity: usize,
) -> Result<(AudioInput, FeatureWorker), StreamError> {
    if capacity == 0 {
        return Err(StreamError::ZeroCapacity);
    }
    let stream = FeatureStream::new(sample_rate)?;
    let (input, queue) = AudioQueue::new(capacity);
    Ok((input, FeatureWorker { queue, stream }))
}

/// Worker endpoint. Call `poll` from an ordinary thread; no async runtime needed.
pub struct MeterWorker {
    queue: AudioQueue,
    stream: MeterStream,
}
impl MeterWorker {
    /// Resampling delay in 44.1 kHz analysis samples.
    pub fn resampler_delay(&self) -> usize {
        self.stream.resampler_delay()
    }
    /// Process at most `limit` queued samples. The generation changes after a gap;
    /// meter timestamps restart at zero in the new generation.
    pub fn poll(&mut self, limit: usize, mut emit: impl FnMut(u64, TrackerUpdate)) -> usize {
        let stream = &mut self.stream;
        self.queue.poll(limit, |sample, generation, discontinuity| {
            if discontinuity {
                stream.reset();
            }
            stream.process(&[sample], |frame| emit(generation, frame));
        })
    }
}

/// Create a bounded mono audio queue and a worker that emits meter estimates.
/// Rejects zero capacity or input rates outside 8–192 kHz. Capacity is in samples.
pub fn meter_channel(
    sample_rate: usize,
    capacity: usize,
) -> Result<(AudioInput, MeterWorker), StreamError> {
    if capacity == 0 {
        return Err(StreamError::ZeroCapacity);
    }
    let stream = MeterStream::new(sample_rate)?;
    let (input, queue) = AudioQueue::new(capacity);
    Ok((input, MeterWorker { queue, stream }))
}
