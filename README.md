# Spiking Neural Networks

Generalized spiking neural network system with various intergrate and fire models as well as Hodgkin Huxley models,
EEG processing with fourier transforms, and power spectral density calculations

## Biological Neuron Models Broken Down

- (todo...)
- (explanation of integrate and fire)
- (explanation of izhikevich)
- (explaination of hodgkin huxley)
- (explanation of ion channels)
- (explanation of neurotransmission, how hodgkin hux system is adapted for izhikevich)

## Notes

- To fit Izhikevich neuron to Hodgkin Huxley model, can either:
  - Fit voltage changes in Izhikevich to voltage changes in Hodgkin Huxley
  - Fit Izhikevich curve to Hodgkin Huxley
    - Can either use a Fourier transform to compare or use mean squared error at each iteration
    - Or, compare the difference between spike times and the amplitude of the spikes (spike time difference being post minus pre, could compare individual spike differences or average spike difference)
      - Write trait for iterate function so coupled neuron firing code can be shared between Hodgkin Huxley and Izhikevich neurons
      - Cell might need a rename to IntegrateAndFireModel
      - Perform this for multiple static inputs, 0 to 100
      - Or perform this with coupled neurons (might need to account for weights)
      - Or both at the same time
- Can also implement version that either adds neurotransmitter current or adds the current to stimulus

- Eventually remove old neurotransmitter system and replace it with new one
- Eventually remove existing genetic algorithm fit for matching an EEG signal and replace it with R-STDP one or at least genetic algorithm that changes weights rather that input equation
- Eventually remove input equation system

- **Need to rename all `input_voltage` to `input_current`**
- **Need to rename all post_synaptic and all pre_synaptic to postsynaptic and presynaptic**

- Hopfield network
  - [Hopfield network pseudocode](https://www.geeksforgeeks.org/hopfield-neural-network/)
  - [Hopfield network tutorial](https://github.com/ImagineOrange/Hopfield-Network/blob/main/hopfield_MNIST.py)
  - [Hopfield network explained](https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073)
- When done with Hopfield, move to the [cue model](https://onlinelibrary.wiley.com/doi/full/10.1111/tops.12247#:~:text=Guanfacine%20increases%20(Yohimbine%20decreases)%20the,effect%20on%20nonpreferred%20direction%20neurons.)
  - Cue input is fed into working memory neurons
    - Cue is -1 or 1
  - Working memory neurons loop back into themselves with some bayesian noise
  - Cue is removed and working memory output can be decoded
    - Decoded by taking weighted sum of working memory neurons
    - If below 0, then percieved cue is -1, if above 0, percieved cue is 1
  - Firing rate of neurons increase over time signal should become more unstable over time and starts to not represent the same signal
- When done with cue models, move to [liquid state machines](https://medium.com/@noraveshfarshad/reservoir-computing-model-of-prefrontal-cortex-4cf0629a8eff#:~:text=In%20a%20reservoir%20computing%20model,as%20visual%20or%20auditory%20cues.) (also accessible [here](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006624))
- When done modeling memory, attempt classifying with liquid state machines

## Todo

### Backend

- [x] Integrate and fire models
  - [x] Basic
  - [x] Adaptive
  - [x] Adaptive Exponential
  - [x] Izhikevich
  - [x] Izhikevich Leaky Hybrid
- [x] Static input test
- [ ] STDP test
  - [x] Single coupled neurons
  - [x] Multiple coupled neurons
  - [ ] Single coupled R-STDP
  - [ ] Multiple coupled R-STDP
  - [ ] Testing with weights summing to 1
- [x] Lattice
  - [x] Graph representation of lattice
    - [x] Adjacency list
    - [x] Adjacency matrix
  - [ ] Generating GIFs from lattice
    - [x] Naive approach
    - [ ] Optimized GIF generation
  - [x] Different potentiation types
    - [x] Inhibitory
    - [x] Excitatory
  - [ ] Recording lattice over time
    - [ ] Textual
      - [x] Averaged
      - [x] Grid
      - [ ] EEG
    - [x] Binary
      - [x] Averaged
      - [x] Grid
  - [x] Lattice testing without STDP
  - [x] Lattice testing with STDP
  - [ ] Lattice with EEG evaluation
    - [x] Analysis with Fourier transforms
      - [x] Calculation of spectral analysis
      - [x] Calculation of Earth moving distance
- [ ] Hodgkin Huxley
  - [x] Basic gating
  - [ ] Neurotransmission
    - [x] Systemized method for adding ionotropic neurotransmitters
    - [x] AMPA
    - [x] NMDA
    - [ ] GABA
      - [x] GABAa
      - [ ] GABAb
        - [x] GABAb primary
        - [ ] GABAb secondary
  - [ ] Additional gating
    - [x] Systemized method for adding gates
    - [ ] L-Type Calcium
    - [ ] T-Type Calcium
    - [ ] M-current
  - [ ] More complex neurotransmission equations (with delay time constants and such)
  - [ ] Multicompartmental models
    - [ ] Systemized method for adding compartments
- [ ] TOML parsing
  - [x] Integrate and fire parsing
    - [x] Static input
    - [x] STDP testing
    - [x] Lattice
  - [ ] Hodgkin Huxley
    - [x] Static input
    - [ ] STDP testing
    - [x] Built in neurotransmitters
    - [ ] New neurotransmitter from TOML
    - [x] Built in additional gates
    - [ ] New gates from TOML
- [ ] Izhikevich neurotransmission
  - [ ] Fitting Izhikevich neuron to Hodgkin Huxley model with genetic algorithm
    - [ ] Objective function
      - [x] Finding spikes
      - [ ] Comparing spikes
      - [ ] Comparing static and coupled inputs
    - [ ] Fitting with CUDA backend (and transfering this to Python interface)
  - [ ] Using existing neurotransmitter framework with Izhikevich as either input stimulus or additional current added on
  - [ ] Remove existing neurotranmission system and replace it with new one
- [ ] Simulating modulation of other neurotransmitters on lattice
- [ ] Simulation of working memory (refer to guanfacine working memory model)
  - [ ] Discrete state neuron (for testing)
  - [ ] Discrete learning rules
  - [ ] Hopfield network
  - [ ] Liquid state machine
- [ ] Simulation of psychiatric illness
- [ ] Simulation of virtual medications
- [ ] R-STDP based classifier
  - [ ] Simple encoding of input
  - [ ] Modifying the bursting parameters to encode more information in input
    - [ ] Potentially having weights directly calculated/modified from bursting parameters

### Lixirnet

- [x] Integrate and fire models
  - [x] Basic
  - [x] Adaptive
  - [x] Adaptive Exponential
  - [x] Izhikevich
  - [x] Izhikevich Leaky Hybrid
- [x] Static input test
- [ ] STDP test
  - [x] Regular STDP
  - [ ] R-STDP
- [ ] Lattice
  - [ ] Graphs input
    - [ ] Adjacency list
    - [ ] Adjacency matrix
- [ ] Hodgkin Huxley
  - [x] Basic gating
  - [ ] Neurotransmission
  - [ ] Additional gating

### CUDA

- [ ] Parallel integrate and fire
  - [ ] Parallel voltage update
  - [ ] Parallel adaptive update
  - [ ] Parallel input calculation
- [ ] Parallel Hodgkin Huxley
- [ ] Interfacing from Python

## Docs

(see other `.md` fils)

## Results

### Lattice

![Lattice](backend/results/lattice_images.gif)

### Hodgkin Huxley

#### Neurotransmission

![AMPA](backend/results/ampa.png)

![GABAa](backend/results/gabaa.png)

![GABAb](backend/results/gabab.png)

![NMDA](backend/results/nmda.png)

#### Additional Gates

<!-- ![L-Type Calcium]() -->

## Sources

- (todo)
- izhikevich
- destexhe
- antipsychotics sim paper
- dopamine model with hodgkin huxley
- biological signal processing richard b wells
