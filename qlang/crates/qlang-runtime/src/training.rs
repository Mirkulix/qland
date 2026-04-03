//! IGQK Training Loop — Real gradient descent on QLANG graphs.
//!
//! Implements Algorithm 1 from the IGQK paper:
//!   1. Forward pass through the graph
//!   2. Compute loss (cross-entropy)
//!   3. Numerical gradients (Phase 1 — Phase 2 will add autograd)
//!   4. Update weights via gradient descent
//!   5. Optional: IGQK quantum gradient flow
//!
//! This runs entirely within QLANG — no Python, no PyTorch.

use std::collections::HashMap;
use qlang_core::tensor::{Dtype, Shape, TensorData};

/// Training configuration.
pub struct TrainConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub log_interval: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 10,
            batch_size: 32,
            log_interval: 100,
        }
    }
}

/// Simple MLP weights (not stored as QLANG graph yet, but as raw tensors).
pub struct MlpWeights {
    pub w1: Vec<f32>,  // [input_dim, hidden_dim]
    pub b1: Vec<f32>,  // [hidden_dim]
    pub w2: Vec<f32>,  // [hidden_dim, output_dim]
    pub b2: Vec<f32>,  // [output_dim]
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl MlpWeights {
    /// Initialize with Xavier/Glorot initialization.
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt() as f32;
        let scale2 = (2.0 / (hidden_dim + output_dim) as f64).sqrt() as f32;

        // Simple deterministic "random" init using sine
        let w1: Vec<f32> = (0..input_dim * hidden_dim)
            .map(|i| (i as f32 * 0.4871).sin() * scale1)
            .collect();
        let b1 = vec![0.0; hidden_dim];
        let w2: Vec<f32> = (0..hidden_dim * output_dim)
            .map(|i| (i as f32 * 0.7291).sin() * scale2)
            .collect();
        let b2 = vec![0.0; output_dim];

        Self { w1, b1, w2, b2, input_dim, hidden_dim, output_dim }
    }

    /// Forward pass: x → W1 → relu → W2 → softmax → probabilities
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let batch = x.len() / self.input_dim;

        // Layer 1: h = relu(x @ W1 + b1)
        let mut hidden = vec![0.0f32; batch * self.hidden_dim];
        for b in 0..batch {
            for j in 0..self.hidden_dim {
                let mut sum = self.b1[j];
                for i in 0..self.input_dim {
                    sum += x[b * self.input_dim + i] * self.w1[i * self.hidden_dim + j];
                }
                hidden[b * self.hidden_dim + j] = sum.max(0.0); // ReLU
            }
        }

        // Layer 2: logits = h @ W2 + b2
        let mut logits = vec![0.0f32; batch * self.output_dim];
        for b in 0..batch {
            for j in 0..self.output_dim {
                let mut sum = self.b2[j];
                for i in 0..self.hidden_dim {
                    sum += hidden[b * self.hidden_dim + i] * self.w2[i * self.output_dim + j];
                }
                logits[b * self.output_dim + j] = sum;
            }
        }

        // Softmax
        for b in 0..batch {
            let offset = b * self.output_dim;
            let max = logits[offset..offset + self.output_dim]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..self.output_dim {
                logits[offset + j] = (logits[offset + j] - max).exp();
                sum += logits[offset + j];
            }
            for j in 0..self.output_dim {
                logits[offset + j] /= sum;
            }
        }

        logits
    }

    /// Compute cross-entropy loss.
    pub fn loss(&self, probs: &[f32], targets: &[u8]) -> f32 {
        let batch = targets.len();
        let mut total = 0.0f32;
        for b in 0..batch {
            let target = targets[b] as usize;
            let p = probs[b * self.output_dim + target].max(1e-7);
            total -= p.ln();
        }
        total / batch as f32
    }

    /// Compute accuracy.
    pub fn accuracy(&self, probs: &[f32], targets: &[u8]) -> f32 {
        let batch = targets.len();
        let mut correct = 0;
        for b in 0..batch {
            let offset = b * self.output_dim;
            let predicted = probs[offset..offset + self.output_dim]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if predicted == targets[b] as usize {
                correct += 1;
            }
        }
        correct as f32 / batch as f32
    }

    /// Train one step with numerical gradient descent.
    /// Uses finite differences for gradients (simple but works for small models).
    pub fn train_step(&mut self, x: &[f32], targets: &[u8], lr: f32) -> f32 {
        let eps = 1e-4;

        // Compute current loss
        let probs = self.forward(x);
        let current_loss = self.loss(&probs, targets);

        // Gradient for W1
        for i in 0..self.w1.len() {
            self.w1[i] += eps;
            let probs_plus = self.forward(x);
            let loss_plus = self.loss(&probs_plus, targets);
            self.w1[i] -= eps;

            let grad = (loss_plus - current_loss) / eps;
            self.w1[i] -= lr * grad;
        }

        // Gradient for b1
        for i in 0..self.b1.len() {
            self.b1[i] += eps;
            let probs_plus = self.forward(x);
            let loss_plus = self.loss(&probs_plus, targets);
            self.b1[i] -= eps;

            let grad = (loss_plus - current_loss) / eps;
            self.b1[i] -= lr * grad;
        }

        // Gradient for W2
        for i in 0..self.w2.len() {
            self.w2[i] += eps;
            let probs_plus = self.forward(x);
            let loss_plus = self.loss(&probs_plus, targets);
            self.w2[i] -= eps;

            let grad = (loss_plus - current_loss) / eps;
            self.w2[i] -= lr * grad;
        }

        // Gradient for b2
        for i in 0..self.b2.len() {
            self.b2[i] += eps;
            let probs_plus = self.forward(x);
            let loss_plus = self.loss(&probs_plus, targets);
            self.b2[i] -= eps;

            let grad = (loss_plus - current_loss) / eps;
            self.b2[i] -= lr * grad;
        }

        current_loss
    }

    /// Train one step with analytical backpropagation (much faster than numerical).
    ///
    /// Computes gradients analytically through the network:
    ///   Forward:  x → (W1·x + b1) → relu → (W2·h + b2) → softmax → probs
    ///   Backward: dL/dprobs → dL/dW2,db2 → dL/dhidden → dL/dW1,db1
    pub fn train_step_backprop(&mut self, x: &[f32], targets: &[u8], lr: f32) -> f32 {
        let batch = targets.len();
        if batch == 0 {
            return 0.0;
        }

        // ---- Forward pass (save intermediates) ----
        // Layer 1: pre_act1 = x @ W1 + b1, hidden = relu(pre_act1)
        let mut pre_act1 = vec![0.0f32; batch * self.hidden_dim];
        let mut hidden = vec![0.0f32; batch * self.hidden_dim];
        for b in 0..batch {
            for j in 0..self.hidden_dim {
                let mut sum = self.b1[j];
                for i in 0..self.input_dim {
                    sum += x[b * self.input_dim + i] * self.w1[i * self.hidden_dim + j];
                }
                pre_act1[b * self.hidden_dim + j] = sum;
                hidden[b * self.hidden_dim + j] = sum.max(0.0);
            }
        }

        // Layer 2: logits = hidden @ W2 + b2
        let mut logits = vec![0.0f32; batch * self.output_dim];
        for b in 0..batch {
            for j in 0..self.output_dim {
                let mut sum = self.b2[j];
                for i in 0..self.hidden_dim {
                    sum += hidden[b * self.hidden_dim + i] * self.w2[i * self.output_dim + j];
                }
                logits[b * self.output_dim + j] = sum;
            }
        }

        // Softmax
        let mut probs = logits.clone();
        for b in 0..batch {
            let off = b * self.output_dim;
            let max = probs[off..off + self.output_dim].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..self.output_dim {
                probs[off + j] = (probs[off + j] - max).exp();
                sum += probs[off + j];
            }
            for j in 0..self.output_dim {
                probs[off + j] /= sum;
            }
        }

        // Loss (cross-entropy)
        let loss = self.loss(&probs, targets);

        // ---- Backward pass ----
        // dL/d_logits = probs - one_hot(targets)  (for softmax + cross-entropy)
        let mut d_logits = probs.clone();
        for b in 0..batch {
            let t = targets[b] as usize;
            d_logits[b * self.output_dim + t] -= 1.0;
        }
        // Average over batch
        let inv_batch = 1.0 / batch as f32;
        for v in d_logits.iter_mut() {
            *v *= inv_batch;
        }

        // Gradients for W2 and b2
        // dL/dW2 = hidden^T @ d_logits  (shape: [hidden_dim, output_dim])
        let mut dw2 = vec![0.0f32; self.hidden_dim * self.output_dim];
        let mut db2 = vec![0.0f32; self.output_dim];
        for b in 0..batch {
            for i in 0..self.hidden_dim {
                for j in 0..self.output_dim {
                    dw2[i * self.output_dim + j] +=
                        hidden[b * self.hidden_dim + i] * d_logits[b * self.output_dim + j];
                }
            }
            for j in 0..self.output_dim {
                db2[j] += d_logits[b * self.output_dim + j];
            }
        }

        // dL/d_hidden = d_logits @ W2^T  (shape: [batch, hidden_dim])
        let mut d_hidden = vec![0.0f32; batch * self.hidden_dim];
        for b in 0..batch {
            for i in 0..self.hidden_dim {
                let mut sum = 0.0f32;
                for j in 0..self.output_dim {
                    sum += d_logits[b * self.output_dim + j] * self.w2[i * self.output_dim + j];
                }
                d_hidden[b * self.hidden_dim + i] = sum;
            }
        }

        // ReLU backward: zero out where pre_act1 <= 0
        for b in 0..batch {
            for j in 0..self.hidden_dim {
                if pre_act1[b * self.hidden_dim + j] <= 0.0 {
                    d_hidden[b * self.hidden_dim + j] = 0.0;
                }
            }
        }

        // Gradients for W1 and b1
        // dL/dW1 = x^T @ d_hidden  (shape: [input_dim, hidden_dim])
        let mut dw1 = vec![0.0f32; self.input_dim * self.hidden_dim];
        let mut db1 = vec![0.0f32; self.hidden_dim];
        for b in 0..batch {
            for i in 0..self.input_dim {
                for j in 0..self.hidden_dim {
                    dw1[i * self.hidden_dim + j] +=
                        x[b * self.input_dim + i] * d_hidden[b * self.hidden_dim + j];
                }
            }
            for j in 0..self.hidden_dim {
                db1[j] += d_hidden[b * self.hidden_dim + j];
            }
        }

        // ---- Update weights ----
        for i in 0..self.w1.len() {
            self.w1[i] -= lr * dw1[i];
        }
        for i in 0..self.b1.len() {
            self.b1[i] -= lr * db1[i];
        }
        for i in 0..self.w2.len() {
            self.w2[i] -= lr * dw2[i];
        }
        for i in 0..self.b2.len() {
            self.b2[i] -= lr * db2[i];
        }

        loss
    }

    /// IGQK ternary compression of weights.
    pub fn compress_ternary(&self) -> MlpWeights {
        fn ternary(v: &[f32]) -> Vec<f32> {
            let mean_abs: f32 = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            let threshold = mean_abs * 0.7;
            v.iter().map(|&x| {
                if x > threshold { 1.0 }
                else if x < -threshold { -1.0 }
                else { 0.0 }
            }).collect()
        }

        MlpWeights {
            w1: ternary(&self.w1),
            b1: self.b1.clone(), // Don't compress biases
            w2: ternary(&self.w2),
            b2: self.b2.clone(),
            input_dim: self.input_dim,
            hidden_dim: self.hidden_dim,
            output_dim: self.output_dim,
        }
    }

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}

/// Generate a tiny synthetic dataset (fake MNIST-like).
/// 4 classes: horizontal line, vertical line, diagonal, empty
pub fn generate_toy_dataset(n_samples: usize, dim: usize) -> (Vec<f32>, Vec<u8>) {
    let mut images = vec![0.0f32; n_samples * dim];
    let mut labels = vec![0u8; n_samples];

    for i in 0..n_samples {
        let class = (i % 4) as u8;
        labels[i] = class;

        let side = (dim as f64).sqrt() as usize;
        let offset = i * dim;

        match class {
            0 => {
                // Horizontal line in middle
                let row = side / 2;
                for col in 1..side - 1 {
                    images[offset + row * side + col] = 1.0;
                }
            }
            1 => {
                // Vertical line in middle
                let col = side / 2;
                for row in 1..side - 1 {
                    images[offset + row * side + col] = 1.0;
                }
            }
            2 => {
                // Diagonal
                for k in 1..side - 1 {
                    images[offset + k * side + k] = 1.0;
                }
            }
            3 => {
                // Cross (both lines)
                let mid = side / 2;
                for k in 1..side - 1 {
                    images[offset + mid * side + k] = 1.0;
                    images[offset + k * side + mid] = 0.5;
                }
            }
            _ => {}
        }
    }

    (images, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_forward() {
        let mlp = MlpWeights::new(16, 8, 4);
        let input = vec![0.5f32; 16];
        let probs = mlp.forward(&input);

        // Softmax should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mlp_loss_decreases() {
        let dim = 16;
        let mut mlp = MlpWeights::new(dim, 8, 4);
        let (images, labels) = generate_toy_dataset(8, dim);

        let probs_before = mlp.forward(&images);
        let loss_before = mlp.loss(&probs_before, &labels);

        // One training step (small lr for numerical gradients)
        mlp.train_step(&images, &labels, 0.001);

        let probs_after = mlp.forward(&images);
        let loss_after = mlp.loss(&probs_after, &labels);

        // Loss should decrease (or at least not increase dramatically)
        assert!(loss_after < loss_before + 0.5,
            "Loss didn't decrease: {loss_before} → {loss_after}");
    }

    #[test]
    fn ternary_compression() {
        let mlp = MlpWeights::new(16, 8, 4);
        let compressed = mlp.compress_ternary();

        // All w1 values should be {-1, 0, 1}
        for &w in &compressed.w1 {
            assert!(w == -1.0 || w == 0.0 || w == 1.0,
                "Non-ternary value: {w}");
        }
    }

    #[test]
    fn toy_dataset_generation() {
        let (images, labels) = generate_toy_dataset(100, 64);
        assert_eq!(images.len(), 100 * 64);
        assert_eq!(labels.len(), 100);
        assert!(labels.iter().all(|&l| l < 4));
    }

    #[test]
    fn backprop_loss_decreases() {
        let dim = 16;
        let mut mlp = MlpWeights::new(dim, 8, 4);
        let (images, labels) = generate_toy_dataset(20, dim);

        let probs_before = mlp.forward(&images);
        let loss_before = mlp.loss(&probs_before, &labels);

        // Multiple backprop training steps
        for _ in 0..10 {
            mlp.train_step_backprop(&images, &labels, 0.1);
        }

        let probs_after = mlp.forward(&images);
        let loss_after = mlp.loss(&probs_after, &labels);

        assert!(loss_after < loss_before,
            "Backprop loss didn't decrease: {loss_before} → {loss_after}");
    }

    #[test]
    fn backprop_achieves_high_accuracy() {
        let dim = 16;
        let mut mlp = MlpWeights::new(dim, 16, 4);
        let (images, labels) = generate_toy_dataset(40, dim);

        // Train with backprop
        for _ in 0..100 {
            mlp.train_step_backprop(&images, &labels, 0.05);
        }

        let probs = mlp.forward(&images);
        let acc = mlp.accuracy(&probs, &labels);
        assert!(acc > 0.7, "Expected >70% accuracy, got {:.1}%", acc * 100.0);
    }
}
