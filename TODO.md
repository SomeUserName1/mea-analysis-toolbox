# TODOs
1.
- [ ] rewrite analysis GUI & hook up functions
- [ ] refactor burst detections
- [ ] refactor seizure detection
- [-] adjust plotting with PyQtGraph
- [ ] extract FOOOF params

2. 
- [-] Improve peak, burst & seizure detection
- [-] fix FOOOF and see if working with it helps
- [ ] fixup transfer entropy
- [ ] WorkerPools 

3.
- [ ] PSI
- [ ] PAC
- [ ] Avalanche analysis
- [ ] Waveforms analysis
- [ ] make runtime errors/all errors visible on UI
- [ ] support CMOS

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
- Bursts: Envelope, mv Var, IPI, freq decomposition, patterns, IBI, correlations, TE, MI, Coh, GC, SGC
- Seizures: as in bursts
- Correlation: Max (lag, amount)
- SGC: freq peaks
- coherence: Max regions (freqs, lags)
