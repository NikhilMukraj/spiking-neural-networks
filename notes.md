# notes

may need to optimize spiking neural network by optimizing input equation as a grammar and associated parameters simultaneously
read eeg data by converting to csv beforehand
generated eeg data should have same dt as eeg data to replicate, downsampling isnt great so keep it to a minimum
real eeg data and generated eeg data must have same timestep value or converted to the same timestep value when performing comparison
use rustfft, ndarray_complex to calculate fouriers on eeg data
mse doesnt work that well as an objective because it essentially says this is or is not correct, it doesnt give a good degree of correctness, the earth moving distance may be a better metric
need to implemend emd

neuromodulation of neuronal properties needs to be implemented through a neuromodulator that
can modify things like Vth, Vreset, tau_m, (a, b, and w values in adex) and neurotranmissions properties

should be inputtable by user and user should be able to decide which properties
are modified and in what ways

need to move some lif parameters into cell

correlational neuromodulation of bursting without neurotransmission
if high correlation
    minimal change to weight
if low correlation
    high change to weight

if negative weight is negative if positive weight is positive
    magnitude based on pearson's r
