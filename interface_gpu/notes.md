# notes

- attempt to debug current model with print statements during execution to isolate where nans appear, check if it happens during conversion
- if still cannot debug, combine spike train and izhikevich neuron into one neuronal model and try that
- if that does not work try using LIF or AdEx (check kernel/conversion) instead of izhikevich or rent amd or intel gpu (cudo (setup ssh stuff)? or alternative)
- two compartment model with various ion channels + point attractor + dopamine
- ring attractor + point attractor + dopamine
- grid attractor (+ point attractor + dopamine) (find github reference or just increase dimensions of ring attractor)
- serotonin model or other neurotransmitters (acetylcholine, maybe nitric oxide, etc) (hallucination/learning model?)
- general gpu plasticity impl followed by nb macro impl (cpu, gpu, and pyo3)
- toleman eichenbaum machine (tem) (find reference on github as well as paper) (electrical then chemical) (+ dopamine)
- horizontal + vertical control (thousand brains theory + hierarchy + consensus voting) (+ dopamine)
