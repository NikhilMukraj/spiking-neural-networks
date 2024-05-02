# Spiking Neural Networks

Generalized spiking neural network system with various intergrate and fire models as well as Hodgkin Huxley models,
EEG processing with fourier transforms, and power spectral density calculations

## Biological Neuron Models Broken Down

- (todo...)
- (explanation of integrate and fire)
- (explanation of izhikevich)
- (explaination of hodgkin huxley)
- (explanation of ion channels)
- (explanation of neurotransmission, how hodgkin hux system is adapted for izhikevich, explain why receptor kinetics are fixed)

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
    - Fitting bursting Izhikevich to bursting Hodgkin Huxley
      - Need way to detect bursts for Izhikevich fitting, probably something that has a burst tolerance (distance between spikes that differentiates either part of a burst group or the next set of bursts firing)
      - Then comparing the distance between burst groups and the intervals of the burst groups
- Can also implement version that either adds neurotransmitter current or adds the current to stimulus

- Eventually remove old neurotransmitter system and replace it with new one
  - Gap condutance should be retrieved from TOML
- Eventually remove existing genetic algorithm fit for matching an EEG signal and replace it with R-STDP one or at least genetic algorithm that changes weights rather that input equation

- Add neurotransmitter output to each presynaptic neuron that calculates concentration with its own membrane potential, then have postsynaptic neurons sum the concentration * weight to calculate their neurotransmitters
- Separate receptor kinetics struct, dependent on t_total

- **Split `get_dv_change_and_get_spike` into `get_basic_dv_change` and `get_basic_spike`, that way there does not need to be a split between basic and rest of integrate and fires**
  - Could have a function return the correct function for each IFType for now
    - One set of iterate and spike is used such that it directly modifies current voltage and current w, the other one is built to allow arc mutex access, it first calculates dv and dw and spike and then unlocks mutex to modify neuron
    - Need to built get dv, get dw, and get spike for each IF type
      - **Build a function to return the correct function based on IF type**
    - Inputs need to be calculated first, then after inputs calculated dv, dw, and spiking changes (and neurotransmitter concentration and respective currents) can be applied
      - For neuron, get input, add to Hashmap\<Position, Input\>, then apply input to each neuron
  - Could a function that sets a private field within the struct to the correct dv change function and the correct spiking function and then call a method that called that function instead of matching each time
  - Could implement this by integrating IFType into cell struct and setting the right function when IFType is called

- **Move non initialization parameters from IFParameters to cell struct**
  - Make function to translate IFParameters and STDPParameters to cell struct
  - Have a set of bayesian parameters for ensemble of neurons to use
    - Have separate function to get those parameters from TOML
    - Bayesian should only be used with standard deviation is not 0 (for all functions)
- **Completely remove IFParameters**
  - Repurpose get_if_params function to get IFCell parameters
  - Consider removing 0-1 scaling default
  - Make sure to use regular parameters default if IFType is not Izhikevich or Izhikevich Leaky, but if it is use the Izhikevich default
  - Update code in obsidian when refactor is done, maybe update results

- Split `main.rs` functions into a few different files for readability

- Should create a CellType enum to store IFType and Hodgkin Huxley type for later use in lattice simulation function

- Use Rayon to thread lattice calculations (remove storing dv and is_spiking in hashmap and place it in the struct)

- Lixirnet should be reworked after neurotransmission refactor, should just pull from backend
  - Update by copying over backend
  - For now Lixirnet can work with lattices by converting adjacency matrices in Numpy to Rust
  - Should have an option to convert the matrix to and adjacency list later, or implement a direct conversion from dictionary to adjacency list

- Input from cell grid functions should be refactored to work with Hodgkin Huxley cells via a trait and condensed into one function where weighting is optional

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
    - **Or perceived cue could be above or below a given baseline, cue itself can be a fast (or excitatory) spike train or a slow (or potentially inhibitory) spike train, 0 is a baseline spike train speed (spike train just being a series of spikes)**
      - Poisson neuron should be used to generate spike train
      - Might be more practical to use an excitatory and inhibitory input and check deviation from baseline over time
  - Firing rate of neurons increase over time signal should become more unstable over time and starts to not represent the same signal
  - To also model forgetting, increasing amounts of noise can be added to working memory model over time
- When done with cue models, move to [liquid state machines](https://medium.com/@noraveshfarshad/reservoir-computing-model-of-prefrontal-cortex-4cf0629a8eff#:~:text=In%20a%20reservoir%20computing%20model,as%20visual%20or%20auditory%20cues.) (also accessible [here](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006624))
  - Recurrent connections in reservoir compute act as working memory that stores information through recurrent connections that may slowly degrade over time, target is slowing the degradation in order to improve memory recall
  - Decoding unit acts as readout, decoding unit likely would need some training in the form of R-STDP
  - Can check accuracy of liquid state machine or stability of answer over time, similar to simple reccurent model
  - Model of memory using reservoir compute and R-STDP could model effects of dopamine by modulating relevant R-STDP parameters and modulating the neuron parameters as well, could also model effects of drugs by training first and the messing with modulated values
- When done modeling memory, attempt general classification tasks with liquid state machines

- [Gap junction equation and various models for different currents](https://www.maths.nottingham.ac.uk/plp/pmzsc/cnn/CNN4.pdf)

- Phase plane analysis of adaptive $w$ and voltage $v$ values

- Look into delta rule for learning
- [Implementation details of a Izhikevich R-STDP synapse](https://link.springer.com/article/10.1007/s00521-022-07220-6)

### Notes on what to modulate

- Synaptic condutance of ion channels (potentially rate/gating constants)
  - Na+, K+
  - Leak current
  - Ca++ (L-current HVA, T-current)
  - M-current
  - Rectifying channels
- Synaptic conductance of ligand gated channels (potentially maximal neurotransmitter concentration) (and forward and backward rate constants)
  - AMPA, GABA(a/b), NMDA
- Metabotropic neurotransmitters (concentration)
  - Dopamine
  - Serotonin
  - Nitric oxide
  - Acetylcholine
  - Glutamate
  - Adrenaline
- Astrocytes
- Weights
  - Weights between certain neurons or specific projections (pyramidal or chandelier for example)

(simulation total time should be around 10 min)

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
    - Note: input spike train is being inputted into input layers, depending on how strongly the output neurons are firing (and which neurons are spiking) reward is applied, this input is being inputted for specific duration *it is not instantaneous*
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
    - [ ] Option to rewrite Fourier analysis to file
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
    - [ ] [Cable theory](https://boulderschool.yale.edu/sites/default/files/files/DayanAbbott.pdf)
    - [ ] Systemized method for adding compartments
  - [ ] Hodgkin Huxley lattice simulation
    - [ ] Spike detection (have a window of past voltages and use find peaks to determine if it is spiking)
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
    - [x] Objective function
      - [x] Finding spikes
      - [x] Comparing spikes
        - [x] Amplitude of spikes, spike time differences, and number of spikes
        - [x] Scaling data properly
      - [x] Comparing static and coupled inputs
      - [x] Comparing spikes under various input conditions
    - [ ] [Spike time concidence objective function](https://www.sciencedirect.com/science/article/pii/S0893608019303065)
    - [ ] Potential objective function refactor with spike amplitude being height subtracted by minimum
    - [ ] Fitting with CUDA backend (and transfering this to Python interface)
  - [ ] Using existing neurotransmitter framework with Izhikevich as either input stimulus or additional current added on
    - [x] Remove existing neurotranmission system
    - [ ] Integrate and fire models with ligand gated channels interacting with neurotransmitters
      - [ ] Moving neurotransmitter concentration into seperate struct and moving receptor kinetics variables to seperate struct (with parameter $T_max$)
        - [ ] Presynaptic neuron calculates concentration and saves it
        - [ ] Post synaptic neuron applies weight to the concentration and sums it, then applies receptor kinetics
        - [ ] Integrate this into Hodgkin Huxley models too
      - [ ] Option to record each neurotransmitter current over time in lattice (g * r)
      - [ ] Recording g, r, and T over time
        - [ ] Coupling tests
        - [ ] STDP tests
    - [ ] Approximation of neurotransmitter in synapse over time (as well as receptor occupancy over time)
      - $\frac{dT}{dt} = \alpha T + T_{max} H(V_p - V_{th})$ where $T$ is neurotransmitter concentration, $T_{max}$ is maximum neurotransmitter concentration, $\alpha$ is clearance rate, $H(x)$ is the heaviside function, $V_p$ is the average of the presynaptic voltages, and $V_{th}$ is the spiking threshold
      - If not using average, add $w \alpha T + w T_{max} H(V_p - V_{th}) w$ for each presynaptic voltage where $w$ is a weight
      - Cap $T$ to make sure it does not go below 0
        - Apply change and then do `self.t = max(0, t_change);`
      - Receptor occupancy could be assumed to be at maximum
      - Could be implemented with a trait neurotransmitter that has apply neurotransmitter change to apply t and r changes and get r to retrieve modifier
- [ ] Poisson neuron
- [ ] Astrocytes model
  - [Coupled with Hodgkin Huxley neurons](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3843665/)
  - [Astrocytes equations](https://www.sciencedirect.com/science/article/pii/S0960077922011481)
  - [Astrocytes + Izhikevich](https://www.frontiersin.org/articles/10.3389/fncel.2021.631485/full)
    - [Code for astrocytes and neural network](https://github.com/altergot/neuro-astro-network)
  - [ ] Tripartite synapse
    - Record how weights change over time
  - [ ] Neuro-astrocytic network (hippocampal model)
    - Could be tested with or without STDP occuring
    - Record how weights change over time
- [ ] Simulating modulation of other neurotransmitters on lattice
- [ ] Simulation of working memory (refer to guanfacine working memory model)
  - [ ] Discrete state neuron (for testing)
  - [ ] Discrete learning rules
  - [ ] Hopfield network
  - [ ] Simple recurrent memory
  - [ ] Liquid state machine
    - Should have a cue and retrieval system (for now supervised, could look into unsupervised methods later)
      - Present cue for duration, remove cue, see how long retrieval signal lasts
        - Matching task, present cue, remove cue, present a new cue and determine whether the new cue is the same or different (DMS task)
      - Could add noise over time similar to simple recurrent memory to modulate forgetting if signal stability stays constant
      - **Measure signal stability after cue is removed (see guanfacine paper)**
    - Could model cognition with something similar to a traveling salesman problem
  - [ ] Liquid state machine with astrocytes
  - [ ] Neuro-astrocyte memory model
- [ ] Simulation of psychiatric illness
- [ ] Simulation of virtual medications
- [ ] R-STDP based classifier
  - Reward may need to be applied after a grace period so the model can converge on an answer first
  - [ ] Simple encoding of input
  - [ ] Modifying the bursting parameters to encode more information in input
    - [ ] Potentially having weights directly calculated/modified from bursting parameters
  - [ ] Liquid state machine with R-STDP
    - Could look into weighted graphs input, each node could a place on a prism geometry and each weight could be a node in between each node, could also be rotated in the geometry for data augmentation purposes
    - Or could input as an adjacency matrix (SMILES enumeration compatible)
  - [ ] Liquid state machine with astrocytes and R-STDP
  - [ ] Combining input with neurotransmission, encoding certain inputs with more or less neurotransmitter (ionotropic or otherwise)

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
