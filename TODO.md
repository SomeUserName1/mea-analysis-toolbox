# Work Packages
## Programming
### 1. CMOS support & code simplification
- rearange matrix on input and follow-up
    - [ ] import & model
    - [ ] selection
    - [ ] preproc
    - [ ] analyze
    - [ ] views
- support CMOS

### 2. Batch processing & Group analysis
- co-register micrograph to retrieve electrode position from atlas
- between group alignment of slices
- aggregation to group
- statistical tests & p-values, effect sizes
- comparative visualizations (per subject plot in axis)
- Peter/Niko and adjust plotting
- parallelization

### 3. Extend analysis
#### 1. Extend preprocessing
- Detect bad channels using pyprep, SNR, impedances
- rereferencing
- artifact removal using ICA
- eproching

#### 2. Analysis
- Per electrode
    - Improve peak detection
    - improve burst detection
    - FOOOF
    - waveform sorting
    - Waveform & spike analysis
    - entropies & other complexity measures
- Between electrodes
    - CSD
    - TE/granger causality
    - coherence/functional connectivity
    - correlation
    - PAC
- spontaneous activity specific

#### 3. Viz
- per band power animation
- Use [Fastplotlib](https://github.com/kushalkolar/fastplotlib) for plotting or Qt if to be run locally
- creative part

## Lab, biochem, genetics
### Recordings
- Ask Uli/Daniela for MEA introduction
- Ask peter to try vibratome
- check TFA regularly - nur für tötung, braucht es __nicht__ für recordings
- Setup protocol & docs for MEA recordings
- Mainz nachfragen wie stimuliert
- Uli/MCS micromanipulators

### Lectures
- Get up from next week for biochem II & IV, org Chemistry
- genetics block course(s)
