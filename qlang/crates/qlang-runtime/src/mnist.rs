//! MNIST Data Loader — Load the real MNIST dataset.
//!
//! Downloads and parses the IDX binary format:
//! - train-images-idx3-ubyte (60000 images, 28×28)
//! - train-labels-idx1-ubyte (60000 labels, 0-9)
//! - t10k-images-idx3-ubyte  (10000 test images)
//! - t10k-labels-idx1-ubyte  (10000 test labels)
//!
//! If files don't exist, generates synthetic MNIST-like data.

use std::fs;
use std::path::Path;

/// MNIST dataset.
pub struct MnistData {
    pub train_images: Vec<f32>,  // [n_train, 784], normalized to [0, 1]
    pub train_labels: Vec<u8>,   // [n_train], values 0-9
    pub test_images: Vec<f32>,   // [n_test, 784]
    pub test_labels: Vec<u8>,    // [n_test]
    pub n_train: usize,
    pub n_test: usize,
    pub image_size: usize,       // 784 = 28×28
    pub n_classes: usize,        // 10
}

impl MnistData {
    /// Try to load from IDX files, or generate synthetic data.
    pub fn load(data_dir: &str) -> Self {
        let train_images_path = format!("{}/train-images-idx3-ubyte", data_dir);
        let train_labels_path = format!("{}/train-labels-idx1-ubyte", data_dir);
        let test_images_path = format!("{}/t10k-images-idx3-ubyte", data_dir);
        let test_labels_path = format!("{}/t10k-labels-idx1-ubyte", data_dir);

        if Path::new(&train_images_path).exists() {
            // Load real MNIST
            let train_images = parse_idx_images(&train_images_path);
            let train_labels = parse_idx_labels(&train_labels_path);
            let test_images = parse_idx_images(&test_images_path);
            let test_labels = parse_idx_labels(&test_labels_path);

            let n_train = train_labels.len();
            let n_test = test_labels.len();

            Self {
                train_images, train_labels,
                test_images, test_labels,
                n_train, n_test,
                image_size: 784,
                n_classes: 10,
            }
        } else {
            // Generate synthetic MNIST-like data
            Self::synthetic(1000, 200)
        }
    }

    /// Generate synthetic MNIST-like data.
    /// Each digit is a simple pattern (not real handwriting).
    pub fn synthetic(n_train: usize, n_test: usize) -> Self {
        let image_size = 784; // 28×28

        let mut train_images = vec![0.0f32; n_train * image_size];
        let mut train_labels = vec![0u8; n_train];
        let mut test_images = vec![0.0f32; n_test * image_size];
        let mut test_labels = vec![0u8; n_test];

        for i in 0..n_train {
            let label = (i % 10) as u8;
            train_labels[i] = label;
            draw_digit(&mut train_images[i * image_size..(i + 1) * image_size], label, i);
        }

        for i in 0..n_test {
            let label = (i % 10) as u8;
            test_labels[i] = label;
            draw_digit(&mut test_images[i * image_size..(i + 1) * image_size], label, i + n_train);
        }

        Self {
            train_images, train_labels,
            test_images, test_labels,
            n_train, n_test,
            image_size,
            n_classes: 10,
        }
    }

    /// Get a batch of training data.
    pub fn train_batch(&self, offset: usize, batch_size: usize) -> (&[f32], &[u8]) {
        let start = offset % self.n_train;
        let end = (start + batch_size).min(self.n_train);
        (
            &self.train_images[start * self.image_size..end * self.image_size],
            &self.train_labels[start..end],
        )
    }
}

/// Draw a synthetic digit pattern on a 28×28 image.
fn draw_digit(image: &mut [f32], digit: u8, seed: usize) {
    let w = 28usize;
    // Add slight variation based on seed
    let offset_x = ((seed * 7) % 3) as i32; // 0..2 only, no negatives for safety
    let offset_y = ((seed * 13) % 3) as i32;

    // Safe pixel setter
    let mut set = |x: i32, y: i32, val: f32| {
        if x >= 0 && x < w as i32 && y >= 0 && y < w as i32 {
            image[y as usize * w + x as usize] = val;
        }
    };

    let ox = offset_x;
    let oy = offset_y;

    match digit {
        0 => {
            for angle in 0..60 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 60.0;
                set(14 + ox + (a.cos() * 8.0) as i32, 14 + oy + (a.sin() * 10.0) as i32, 1.0);
            }
        }
        1 => {
            for y in 4..24 { set(14 + ox, y + oy, 1.0); }
        }
        2 => {
            for x in 8..20 { set(x + ox, 6 + oy, 1.0); }
            for i in 0..14 { set(19 - i + ox, 6 + i + oy, 1.0); }
            for x in 8..20 { set(x + ox, 20 + oy, 1.0); }
        }
        3 => {
            for x in 10..20 { set(x, 6 + oy, 1.0); set(x, 13 + oy, 1.0); set(x, 20 + oy, 1.0); }
            for y in 6..21 { set(19 + ox, y + oy, 1.0); }
        }
        4 => {
            for y in 4..14 { set(10 + ox, y + oy, 1.0); }
            for x in 10..20 { set(x + ox, 13 + oy, 1.0); }
            for y in 4..24 { set(18 + ox, y + oy, 1.0); }
        }
        5 => {
            for x in 8..20 { set(x + ox, 6 + oy, 1.0); set(x + ox, 13 + oy, 1.0); set(x + ox, 20 + oy, 1.0); }
            for y in 6..14 { set(8 + ox, y + oy, 1.0); }
            for y in 13..21 { set(19 + ox, y + oy, 1.0); }
        }
        6 => {
            for y in 4..24 { set(10 + ox, y + oy, 1.0); }
            for angle in 0..40 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 40.0;
                set(15 + (a.cos() * 5.0) as i32, 18 + (a.sin() * 5.0) as i32, 1.0);
            }
        }
        7 => {
            for x in 8..22 { set(x + ox, 5 + oy, 1.0); }
            for i in 0..18 { set(21 - i / 2 + ox, 5 + i + oy, 1.0); }
        }
        8 => {
            for angle in 0..40 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 40.0;
                set(14 + (a.cos() * 5.0) as i32, 9 + (a.sin() * 4.0) as i32, 1.0);
                set(14 + (a.cos() * 5.0) as i32, 19 + (a.sin() * 4.0) as i32, 1.0);
            }
        }
        9 => {
            for angle in 0..40 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 40.0;
                set(15 + (a.cos() * 5.0) as i32, 10 + (a.sin() * 5.0) as i32, 1.0);
            }
            for y in 10..24 { set(20 + ox, y + oy, 1.0); }
        }
        _ => {}
    }
}

/// Parse IDX image file format.
fn parse_idx_images(path: &str) -> Vec<f32> {
    let data = fs::read(path).unwrap();
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    assert_eq!(magic, 2051, "Invalid image magic number");

    let n_images = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let n_rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let n_cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;

    let pixels = &data[16..];
    pixels.iter()
        .map(|&b| b as f32 / 255.0)
        .collect()
}

/// Parse IDX label file format.
fn parse_idx_labels(path: &str) -> Vec<u8> {
    let data = fs::read(path).unwrap();
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    assert_eq!(magic, 2049, "Invalid label magic number");

    data[8..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_mnist() {
        let data = MnistData::synthetic(100, 20);
        assert_eq!(data.n_train, 100);
        assert_eq!(data.n_test, 20);
        assert_eq!(data.train_images.len(), 100 * 784);
        assert_eq!(data.train_labels.len(), 100);
        assert!(data.train_labels.iter().all(|&l| l < 10));
    }

    #[test]
    fn batch_loading() {
        let data = MnistData::synthetic(100, 20);
        let (images, labels) = data.train_batch(0, 10);
        assert_eq!(images.len(), 10 * 784);
        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn digit_patterns_distinct() {
        // Each digit should produce a different pattern
        let mut patterns: Vec<Vec<u8>> = Vec::new();
        for d in 0..10u8 {
            let mut img = vec![0.0f32; 784];
            draw_digit(&mut img, d, 0);
            let binary: Vec<u8> = img.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).collect();
            patterns.push(binary);
        }

        // All patterns should be different
        for i in 0..10 {
            for j in (i + 1)..10 {
                assert_ne!(patterns[i], patterns[j],
                    "Digit {i} and {j} have identical patterns");
            }
        }
    }
}
