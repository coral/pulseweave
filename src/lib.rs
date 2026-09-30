//! Streaming bar, beat, and tatum analysis for mono audio.
//!
//! The numerical building blocks live in [`dsp`], with periodicity
//! evidence in [`period`] and joint period/phase inference in [`decoder`].
//! Use [`stream::MeterStream`] directly or [`stream::meter_channel`] for a
//! bounded audio-callback handoff to an ordinary worker thread.
#![forbid(unsafe_code)]
pub mod decoder;
pub mod dsp;
pub mod onset;
pub mod period;
pub mod pulse;
pub mod stream;
pub mod tracker;
