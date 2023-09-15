# TODOs
1.
- [x] refactor event detection
- [-] adjust plotting with PyQtGraph: PSD, value grid, spectrogram, network stuff
- [ ] extract FOOOF params

2. 
- [x] Improve peak, burst & seizure detection
- [ ] % Spikes in bursts
- [-] fix FOOOF and see if working with it helps
- [ ] fixup network stuff
- [ ] box & violin plots

3.
- [ ] PAC
- [ ] Waveforms analysis
- [ ] make runtime errors/all errors visible on UI
- [ ] support CMOS
- [ ] WorkerPools 

# How to detect noise?
- Spectrogram: steady backgorund across time and electrodes
- baseline region PSDs
- SNR
- referencing

# What's our quantities to export?
- SNR, RMS, approx entropy, MI, TE, GC 
- PSD: peaks, exponent & offset, band integral/power
- Spectrogram: peaks, noise bands deviations from stationary
- Peaks: Amp, width, slopes
- Events: IPI, freq decomposition, patterns, IBI, correlations, TE, MI, Coh, GC, SGC
- Correlation: Max (lag, amount)
- SGC: freq peaks
- coherence: Max regions (freqs, lags)
