# notes

- attempt to debug current model with print statements during execution to isolate where nans appear, check if it happens during conversion
- see what happens when all weights are 0 (including dopamine)
- if still cannot debug, combine spike train and izhikevich neuron into one neuronal model and try that
- if that does not work try using LIF or AdEx (check kernel/conversion) instead of izhikevich or rent amd or intel gpu (cudo (setup ssh stuff)? or alternative)
- two compartment model with various ion channels + point attractor + dopamine
- ring attractor + point attractor + dopamine
- grid attractor (+ point attractor + dopamine) (find github reference)
- serotonin model or other neurotransmitters (acetylcholine) (hallucination/learning model?)
- general gpu plasticity impl followed by nb macro impl
- toleman eichenbaum machine (tem) (find reference on github) (electrical then chemical) (+ dopamine)
- horizontal + vertical control (thousand brains theory + hierarchy + consensus voting) (+ dopamine)
