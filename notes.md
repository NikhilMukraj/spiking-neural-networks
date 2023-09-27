# notes

may need to optimize spiking neural network by optimizing input equation as a grammar and associated parameters simultaneously
read eeg data by converting to csv beforehand
generated eeg data should have same dt as eeg data to replicate, downsampling isnt great so keep it to a minimum
real eeg data and generated eeg data must have same timestep value or converted to the same timestep value when performing comparison
use rustfft, ndarray_complex to calculate fouriers on eeg data
