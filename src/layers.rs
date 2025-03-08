use autodiff::F; // Add autodiff import
use ndarray::{Array1, Array2};
use rand::Rng;

pub trait Layer {
    fn forward(&self, input: &Array1<F>) -> Array1<F>;
}

pub struct DenseLayer {
    weights: Array2<F>,
    bias: Array1<F>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize weights with Xavier/Glorot initialization
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            F::var(rng.gen::<f64>() * 2.0 * scale - scale)
        });

        // Initialize biases to zero
        let bias = Array1::from_vec(vec![F::var(0.0); output_size]);

        DenseLayer { weights, bias }
    }
}

impl Layer for DenseLayer {
    fn forward(&self, input: &Array1<F>) -> Array1<F> {
        self.weights.dot(input) + &self.bias
    }
}
