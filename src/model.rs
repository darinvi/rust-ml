use crate::layers::Layer;

pub struct Model {
    input_shape: usize,
    layers: Vec<Layer>,
}

impl Model {
    pub fn new(input_shape: usize) -> Self {
        Model {
            input_shape,
            layers: vec![],
        }
    }

    pub fn add(&mut self, layer_type: &str) {
        self.layers.push(layer);
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut x = input;
        for layer in self.layers.iter() {
            x = layer.forward(x);
        }
        x
    }
}
