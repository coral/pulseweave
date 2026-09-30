//! Spectral accent extraction from mono 44.1 kHz audio.
use crate::dsp::{EnvelopeFilter, erb_to_hz, hz_to_erb};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

pub const SAMPLE_RATE: usize = 44_100;
pub const BANDS: usize = 36;
pub const HOP: usize = 512;
pub const ACCENT_RATE: f64 = SAMPLE_RATE as f64 / 256.0;
const N: usize = 1024;
pub(crate) const GROUP_DELAY: usize = 11;
// Fixed window coefficient keeps FFT window generation numerically stable.
#[allow(clippy::approx_constant)]
const WINDOW_TAU: f64 = 6.28318530717958;

/// A finalized spectral accent frame. One frame represents 256 audio samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccentFrame {
    pub index: u64,
    pub bands: [f64; BANDS],
}

/// Allocation-free processing after construction. Input is mono at 44.1 kHz.
/// Fixed 512-sample hops make output independent of caller block boundaries.
/// The final accent waits for the following envelope sample; it is not predicted.
pub struct OnsetAnalyzer {
    audio: [f64; N],
    filled: usize,
    window: [f64; N],
    edges: [(usize, usize, usize); BANDS],
    fft: Arc<dyn Fft<f64>>,
    spectrum: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    filters: [EnvelopeFilter; BANDS],
    previous: Option<[f64; BANDS]>,
    count: f64,
    mean: f64,
    variance: f64,
    index: u64,
}
impl Default for OnsetAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
impl OnsetAnalyzer {
    /// Allocate FFT workspaces and initialize the spectral band filters.
    pub fn new() -> Self {
        let fft = FftPlanner::new().plan_fft_forward(N);
        let scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
        let low = hz_to_erb(50.0);
        let high = hz_to_erb(SAMPLE_RATE as f64 * 0.45);
        let hz: [f64; BANDS + 2] =
            std::array::from_fn(|i| erb_to_hz(low + i as f64 * (high - low) / (BANDS + 1) as f64));
        let edges = std::array::from_fn(|i| {
            let center = (hz[i + 1] / SAMPLE_RATE as f64 * N as f64).round() as usize;
            let low = (hz[i] / SAMPLE_RATE as f64 * N as f64).round() as usize;
            let high = (hz[i + 2] / SAMPLE_RATE as f64 * N as f64).round() as usize;
            (low.min(center - 1), center, high.max(center + 1))
        });
        Self {
            audio: [0.; N],
            filled: 0,
            window: std::array::from_fn(|i| {
                0.5 - 0.5 * (((i + 1) as f64 * WINDOW_TAU) / (N + 1) as f64).cos()
            }),
            edges,
            fft,
            spectrum: vec![Complex::default(); N],
            scratch,
            filters: std::array::from_fn(|_| EnvelopeFilter::default()),
            previous: None,
            count: 0.,
            mean: 0.,
            variance: 0.,
            index: 0,
        }
    }
    /// Clear audio, normalization, and filter histories; restart frame indices at zero.
    pub fn reset(&mut self) {
        self.audio.fill(0.);
        self.filled = 0;
        for f in &mut self.filters {
            f.reset();
        }
        self.previous = None;
        self.count = 0.;
        self.mean = 0.;
        self.variance = 0.;
        self.index = 0;
    }
    /// Consume any number of mono samples and emit finalized accent frames.
    /// Nonfinite samples are treated as zero; incomplete windows remain buffered.
    pub fn process(&mut self, samples: &[f64], mut emit: impl FnMut(AccentFrame)) {
        for &sample in samples {
            self.push(sample, &mut emit);
        }
    }
    /// Buffer one sample, returning true when a complete FFT window was processed.
    pub(crate) fn push(&mut self, sample: f64, emit: &mut impl FnMut(AccentFrame)) -> bool {
        self.audio[self.filled] = if sample.is_finite() { sample } else { 0.0 };
        self.filled += 1;
        if self.filled != N {
            return false;
        }
        self.frame(emit);
        self.audio.copy_within(HOP..N, 0);
        self.filled = HOP;
        true
    }
    /// Return the number of envelope frames, including the pending accent.
    pub(crate) fn raw_frames(&self) -> usize {
        if self.previous.is_some() {
            self.index as usize + 1
        } else {
            0
        }
    }
    /// Predict the pending accent and filter tail using zero future input.
    pub(crate) fn predicted_tail(&self) -> [[f64; BANDS]; GROUP_DELAY + 1] {
        let mut filters = self.filters.clone();
        let mut envelopes = [[0.; BANDS]; GROUP_DELAY + 1];
        envelopes[0] = self.previous.unwrap_or([0.; BANDS]);
        for envelope in &mut envelopes[1..] {
            *envelope = std::array::from_fn(|b| filters[b].process(0.).max(0.));
        }
        std::array::from_fn(|i| {
            std::array::from_fn(|b| {
                let next = if i < GROUP_DELAY {
                    envelopes[i + 1][b]
                } else {
                    envelopes[i][b]
                };
                envelopes[i][b] * 0.09999999999999998
                    + (next - envelopes[i][b]).max(0.) * (0.9 * ACCENT_RATE * 0.5 / 10.0)
            })
        })
    }
    /// Normalize and transform a complete audio window, then update spectral accents.
    fn frame(&mut self, emit: &mut impl FnMut(AccentFrame)) {
        // Normalize each overlapping FFT input block using running
        // sample variance; overlapping samples are intentionally counted again.
        for &x in &self.audio {
            let n = self.count + 1.0;
            let mean = (self.mean * self.count + x) / n;
            self.variance = if self.count == 0.0 {
                0.0
            } else {
                (1.0 - 1.0 / self.count) * self.variance
                    + n * (mean - self.mean) * (mean - self.mean)
            };
            self.count = n;
            self.mean = mean;
        }
        let std = self.variance.sqrt();
        for i in 0..N {
            // Zero variance produces a silent spectrum.
            self.spectrum[i] = Complex::new(
                if std > 0. {
                    (self.audio[i] - self.mean) * self.window[i] / std
                } else {
                    0.
                },
                0.,
            );
        }
        self.fft
            .process_with_scratch(&mut self.spectrum, &mut self.scratch);
        let energy: [f64; BANDS] = std::array::from_fn(|band| {
            let (low, center, high) = self.edges[band];
            let mut sum = 0.0;
            for bin in low..=center {
                let weighted = (bin - low) as f64 / (center - low) as f64
                    * self.spectrum[bin].norm()
                    / (N as f64).sqrt();
                sum += weighted * weighted;
            }
            for bin in center + 1..=high {
                let weighted = (1.0 - (bin - center) as f64 / (high - center) as f64)
                    * self.spectrum[bin].norm()
                    / (N as f64).sqrt();
                sum += weighted * weighted;
            }
            (sum * 100.0 + 1.0).ln() * 0.21667906533553166
        });
        for sub in 0..2 {
            let envelope = std::array::from_fn(|band| {
                self.filters[band]
                    .process(if sub == 0 { energy[band] * 2.0 } else { 0.0 })
                    .max(0.0)
            });
            if let Some(previous) = self.previous {
                let bands = std::array::from_fn(|b| {
                    previous[b] * 0.09999999999999998
                        + (envelope[b] - previous[b]).max(0.0) * (0.9 * ACCENT_RATE * 0.5 / 10.0)
                });
                emit(AccentFrame {
                    index: self.index,
                    bands,
                });
                self.index += 1;
            }
            self.previous = Some(envelope);
        }
    }
}
