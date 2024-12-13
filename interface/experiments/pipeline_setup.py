import toml


def frange(x, y, step):
  while x < y + step:
    yield x
    x += step

def parse_range_or_list(data):
    result = {}

    for key, value in data.items():
        if isinstance(value, dict):
            if 'min' in value and 'max' in value and 'step' in value:
                result[key] = list(frange(value['min'], value['max'], value['step']))
            else:
                result[key] = value
        else:
            result[key] = value

    return result

def parse_toml(f):
    toml_data = toml.load(f)

    parsed_data = {}
    for section, data in toml_data.items():
        parsed_data[section] = parse_range_or_list(data)

    return parsed_data

def generate_key_helper(key, parsed, given_key):
    if len(parsed['variables'][given_key]) != 1:
        key.append(f'{given_key}: {current_state[given_key]}')

def try_max(a):
    if len(a) == 0:
        return 0
    else:
        return max(a)

def get_weights(n, patterns, a=0, b=0, scalar=1):
    w = np.zeros([n, n])
    for pattern in patterns:
        for i in range(n):
            for j in range(n):
                w[i][j] += (pattern[i] - b) * (pattern[j] - a) 
    for diag in range(n):
        w[diag][diag] = 0

    w *= scalar
    
    return w

def weights_ie(n, scalar, patterns, num_patterns):
    w = np.zeros([n, n])
    for pattern in patterns:
        for i in range(n):
            for j in range(n):
                w[i][j] += pattern[i * n + j]
    
    return (w * scalar) / num_patterns

def check_uniqueness(patterns):
    for n1, i in enumerate(patterns):
        for n2, j in enumerate(patterns):
            if n1 != n2 and (np.array_equal(i, j) or np.array_equal(np.logical_not(i).astype(int), j)):
                return True
    
    return False

def calculate_correlation(patterns):
    num_patterns = patterns.shape[0]
    correlation_matrix = np.zeros((num_patterns, num_patterns))
    
    for i in range(num_patterns):
        for j in range(i, num_patterns):
            correlation = np.dot(patterns[i], patterns[j])
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation
            
    return correlation_matrix

def skewed_random(x, y, skew_factor=1, size=1):
    rand = np.random.beta(skew_factor, 1, size=size)
    
    return x + rand * (y - x)

def setup_neuron(neuron):
    neuron.current_voltage = skewed_random(-65, 30, 0.1)[0]
    neuron.c_m = 25

    return neuron

def reset_spike_train(neuron):
    neuron.chance_of_firing = 0

    return neuron

def get_spike_train_setup_function(pattern_index, distortion, firing_rate, stay_unflipped=False):
    def setup_spike_train(pos, neuron):
        x, y = pos
        index = x * exc_n + y
        state = patterns[pattern_index][index] == 1

        if np.random.uniform(0, 1) < distortion:
            if not stay_unflipped:
                state ^= 1
            else:
                if state != 0:
                    state = 0

        if state:
            neuron.chance_of_firing = firing_rate
        else:
            neuron.chance_of_firing = 0

        return neuron

    return setup_spike_train

def get_spike_train_same_firing_rate_setup(firing_rate):
    def setup_spike_train(neuron):
        neuron.chance_of_firing = firing_rate
    
        return neuron

    return setup_spike_train

def get_noisy_spike_train_setup_function(noise_level, firing_rate):
    def setup_spike_train(neuron):
        if np.random.uniform(0, 1) < noise_level:
            neuron.chance_of_firing = firing_rate
        else:
            neuron.chance_of_firing = 0
        
        return neuron

    return setup_spike_train

def find_peaks_above_threshold(series, threshold):
    peaks, _ = scipy.signal.find_peaks(np.array(series))
    filtered_peaks = [index for index in peaks if series[index] > threshold]
    
    return filtered_peaks

def acc(true_pattern, pred_pattern, threshold=10): 
    current_pred_pattern = pred_pattern
    current_pred_pattern[pred_pattern < threshold] = 0 
    current_pred_pattern[pred_pattern >= threshold] = 1
    return (true_pattern.reshape(exc_n, exc_n) == current_pred_pattern.reshape(exc_n, exc_n)).sum() / (num)

def signal_to_noise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)

def determine_accuracy(
    desired_pattern_index, 
    num_patterns,
    window, 
    peaks, 
    use_correlation_as_accuracy=True, 
    get_all_accuracies=False
):
    if not use_correlation_as_accuracy:
        if not get_all_accuracies:
            current_acc = try_max(
                [acc(patterns[desired_pattern_index], np.array([len([j for j in i if j >= window]) for i in peaks]), threshold=i) for i in range(0, firing_max)]
            )
            current_acc_inv = try_max(
                [acc(np.logical_not(patterns[desired_pattern_index]).astype(int), np.array([len([j for j in i if j >= second_window]) for i in peaks]), threshold=i) for i in range(0, firing_max)]
            )

            current_acc = max(current_acc, current_acc_inv)
        else:
            accs = []
            for pattern_index in range(num_patterns):
                current_acc = try_max(
                    [
                        acc(
                            patterns[pattern_index], 
                            np.array([len([j for j in i if j >= window]) for i in peaks]), 
                            threshold=i
                        ) 
                        for i in range(0, firing_max)
                    ]
                )

                current_acc_inv = try_max(
                    [
                        acc(
                            np.logical_not(patterns[pattern_index]).astype(int), 
                            np.array([len([j for j in i if j >= window]) for i in peaks]), 
                            threshold=i
                        ) 
                        for i in range(0, firing_max)
                    ]
                )

                accs.append(max(current_acc, current_acc_inv))

            current_acc = [float(i) for i in accs]
    else:
        correlation_coefficients = []
        for pattern_index in range(num_patterns):
            correlation_coefficients.append(
                np.corrcoef(patterns[pattern_index], np.array([len([j for j in i if j >= window]) for i in peaks]))[0, 1]
            )
            
        current_acc = bool(desired_pattern_index == np.argmax(correlation_coefficients))

    return current_acc
