use ndarray::{Array1, ArrayView1, Axis, concatenate, s};


fn argsort(arr: &Array1<f32>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(|&i, &j| arr[i].partial_cmp(&arr[j]).unwrap());
    indices
}

fn diff(arr: &Array1<f32>) -> Array1<f32> {
    let mut result = arr.to_owned();

    let len = result.len();
    result = Array1::from_shape_vec(
        len - 1, 
        (1..len)
            .map(|i| result[i] - result[i - 1])
            .collect()
        )
        .unwrap();

    result
}

fn searchsorted(a: &ArrayView1<f32>, v: &ArrayView1<f32>) -> Array1<usize> {
    v.map(|&x| a.iter().position(|&y| y > x).unwrap_or(a.len()))
}

fn cumsum(array: &Array1<f32>) -> Array1<f32> {
    let mut running_sum = 0.0;

    let result = array.map(|&x| {
        running_sum += x;
        running_sum
    });

    result
}

fn get_cdf(weights: &Array1<f32>, sorter: &Vec<usize>, indices: &Array1<usize>) -> Vec<f32> {
    let sorted_cum_weights = concatenate![
        Axis(0), 
        Array1::from_vec(vec![0.]),
        cumsum(&weights.select(Axis(0), sorter))
    ];

    sorted_cum_weights.select(Axis(0), indices.to_vec().as_slice())
        .map(|x| x / sorted_cum_weights.last().expect("Cannot get last weight"))
        .iter()
        .map(|x| *x)
        .collect::<Vec<f32>>()
}

// https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_stats_py.py#L9896
// made in reference to scipy implementation
pub fn earth_moving_distance(
    u_values: Array1<f32>,
    v_values: Array1<f32>,
    u_weights: Array1<f32>,
    v_weights: Array1<f32>,
) -> f32 {
    // u_sorter = np.argsort(u_values)
    // v_sorter = np.argsort(v_values)

    let u_sorter = argsort(&u_values);
    let v_sorter = argsort(&v_values);

    // all_values = np.concatenate((u_values, v_values))
    // all_values.sort(kind='mergesort')

    let mut all_values = concatenate![Axis(0), u_values, v_values];
    let data_slice = all_values.as_slice_mut().unwrap();
    data_slice.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let all_values = Array1::from_vec(data_slice.to_vec());

    // # Compute the differences between pairs of successive values of u and v.
    // deltas = np.diff(all_values)

    let deltas = diff(&all_values);

    // # Get the respective positions of the values of u and v among the values of
    // # both distributions.
    // u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    // v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    let all_values_sliced = &all_values.slice(s![0 as isize .. -1 as isize]);
    let u_cdf_indices = searchsorted(&u_values.select(Axis(0), &u_sorter).view(), all_values_sliced.into());
    let v_cdf_indices = searchsorted(&v_values.select(Axis(0), &v_sorter).view(), all_values_sliced.into());

    // # Calculate the CDFs of u and v using their weights, if specified.
    // if u_weights is None:
    //     u_cdf = u_cdf_indices / u_values.size
    // else:
    //     u_sorted_cumweights = np.concatenate(([0],
    //                                           np.cumsum(u_weights[u_sorter])))
    //     u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    let u_cdf = get_cdf(&u_weights, &u_sorter, &u_cdf_indices);

    // if v_weights is None:
    //     v_cdf = v_cdf_indices / v_values.size
    // else:
    //     v_sorted_cumweights = np.concatenate(([0],
    //                                           np.cumsum(v_weights[v_sorter])))
    //     v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    let v_cdf = get_cdf(&v_weights, &v_sorter, &v_cdf_indices);

    // np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))

    u_cdf.iter()
        .zip(&v_cdf)
        .zip(deltas.iter())
        .map(|((uc, vc), delta)| (uc - vc).abs() * delta)
        .sum()
}
