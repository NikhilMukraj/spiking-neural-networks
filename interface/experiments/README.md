# README

## Schizophrenia Simulation Pipeline

### `[simulation_parameters]`

---

### Required Fields

| Field        | Type    | Description                                    |
|--------------|---------|------------------------------------------------|
| `filename`   | string  | Path to the file where results will be saved. |

### Optional Fields

The following fields have default values if not specified in the TOML file:

| Field                          | Type    | Default Value      | Description                                                   |
|--------------------------------|---------|--------------------|---------------------------------------------------------------|
| `iterations1`                  | integer | 3000               | Number of iterations for the first phase of the simulation   |
| `iterations2`                  | integer | 3000               | Number of iterations for the second phase of the simulation  |
| `peaks_on`                     | boolean | false              | Whether to write peaks to the output                            |
| `cue_firing_rate`              | float   | 0.01               | Baseline firing rate for the cue spike trains                    |
| `second_cue`                   | boolean | true               | Whether a second cue is present                             |
| `second_cue_is_noisy`          | boolean | false              | Whether the second cue is noisy                             |
| `first_cue_is_noisy`           | boolean | false              | Whether the first cue is noisy                              |
| `noisy_cue_noise_level`        | float   | 0.1                | Noise level for noisy cues                                  |
| `noisy_cue_firing_rate`        | float   | 0.01               | Firing rate for noisy cues                                  |
| `measure_snr`                  | boolean | false              | Whether to measure the signal-to-noise ratio                |
| `first_window`                 | integer | 1000               | Accuracy calculation window size for the first phase                        |
| `second_window`                | integer | 1000               | Accuracy calculation window size for the second phase                       |
| `trials`                       | integer | 10                 | Number of trials to run in the simulation                   |
| `num_patterns`                 | integer | 3                  | Number of patterns to simulate                              |
| `weights_scalar`               | float   | 1                  | Scaling factor for excitatory weights                                  |
| `inh_weights_scalar`           | float   | 0.25               | Scaling factor for inhibitory weights                     |
| `a`                            | float   | 1                  | Autoassociative network pattern calculation variable, $s\sum_{i}\sum_{j}(\xi^{u}_{i}-b)(\xi^{u}_{j}-a)$                          |
| `b`                            | float   | 1                  | Autoassociative network pattern calculation variable, $s\sum_{i}\sum_{j}(\xi^{u}_{i}-b)(\xi^{u}_{j}-a)$                          |
| `correlation_threshold`        | float   | 0.08               | Threshold for considering patterns as correlated too correlated (will generate new better if threshold crossed)           |
| `use_correlation_as_accuracy`  | boolean | false              | Use correlation as a measure of accuracy (true if maxiamlly correlated pattern is inputed pattern)                 |
| `get_all_accuracies`           | boolean | false              | Whether to write all accuracies to output                         |
| `skew`                         | float   | 1                  | Skew parameter for distribution of initial voltage values                           |
| `exc_n`                        | integer | 7                  | Number of excitatory neurons                                |
| `inh_n`                        | integer | 3                  | Number of inhibitory neurons.                               |
| `distortion`                   | float   | 0.15               | Amount of distortion in the patterns presented                      |
| `dt`                           | float   | 1                  | Time step of the simulation                                 |
| `c_m`                          | float   | 25                 | Membrane capacitance                                        |

---

### `[variables]`

---

### Variable Fields

The following variables will be combined in every possible way, for how ever many specified trials,
simulation parameters will hold constant while these states change

| Field                     | Type    | Default Value | Description                                                |
|---------------------------|---------|---------------|------------------------------------------------------------|
| `prob_of_exc_to_inh`      | array[float]   | `[0.5]`       | Probability of excitatory-to-inhibitory connections       |
| `exc_to_inh`              | array[float]   | `[1]`         | Strength of excitatory-to-inhibitory connections        |
| `spike_train_to_exc`      | array[float]   | `[5]`         | Spike train values for excitatory neurons                |
| `nmda_g`                  | array[float]   | `[0.6]`       | Conductance for NMDA receptors                           |
| `ampa_g`                  | array[float]   | `[1]`         | Conductance for AMPA receptors                           |
| `gabaa_g`                 | array[float]   | `[1.2]`       | Conductance for GABAa receptors                          |
| `nmda_clearance`          | array[float]   | `[0.001]`     | Clearance rate for NMDA neurotransmitters                |
| `ampa_clearance`          | array[float]   | `[0.001]`     | Clearance rate for AMPA neurotransmitters                |
| `gabaa_clearance`         | array[float]   | `[0.001]`     | Clearance rate for GABAa neurotransmitters               |

If the `glutamate_clearance` field is present, it overrides `nmda_clearance` and `ampa_clearance` with its value

| Field                 | Type    | Default Value | Description                                                |
|-----------------------|---------|---------------|------------------------------------------------------------|
| `glutamate_clearance` | array   | None          | Single value applied to both NMDA and AMPA clearance rates |

---

### Example Argument File

```toml
[simulation_parameters]
peaks_on = true
second_cue = false
second_cue_is_noisy = false
use_correlation_as_accuracy = true
measure_snr = true
weights_scalar = 1
inh_weights_scalar = 0.85
skew = 0.1
c_m = 25
a = -1
b = 0
first_window = 1000
second_window = 3000
iterations1 = 2000
iterations2 = 3000
trials = 15

filename = "grti_with_cue.json"

[variables]
spike_train_to_exc = [5]
prob_of_exc_to_inh = [1]
glutamate_clearance = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
gabaa_clearance = [0.001]
```
