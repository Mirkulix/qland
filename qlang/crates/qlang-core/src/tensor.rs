use serde::{Deserialize, Serialize};
use std::fmt;

/// Data type of tensor elements — what the CPU registers actually hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dtype {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    Bool,
    /// Ternary: {-1, 0, +1} — core to IGQK compression
    Ternary,
}

impl Dtype {
    /// Size in bytes of a single element.
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::Bool => 1,
            Dtype::I8 => 1,
            Dtype::Ternary => 1, // stored as i8, uses only {-1, 0, 1}
            Dtype::F16 | Dtype::I16 => 2,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F64 | Dtype::I64 => 8,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dtype::F16 => write!(f, "f16"),
            Dtype::F32 => write!(f, "f32"),
            Dtype::F64 => write!(f, "f64"),
            Dtype::I8 => write!(f, "i8"),
            Dtype::I16 => write!(f, "i16"),
            Dtype::I32 => write!(f, "i32"),
            Dtype::I64 => write!(f, "i64"),
            Dtype::Bool => write!(f, "bool"),
            Dtype::Ternary => write!(f, "ternary"),
        }
    }
}

/// Shape of a tensor: list of dimension sizes.
/// Empty = scalar. [n] = vector. [m, n] = matrix.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape(pub Vec<Dim>);

/// A single dimension — either a known size or dynamic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    Fixed(usize),
    Dynamic, // unknown at compile time, resolved at runtime
}

impl Shape {
    pub fn scalar() -> Self {
        Shape(vec![])
    }

    pub fn vector(n: usize) -> Self {
        Shape(vec![Dim::Fixed(n)])
    }

    pub fn matrix(m: usize, n: usize) -> Self {
        Shape(vec![Dim::Fixed(m), Dim::Fixed(n)])
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements, if all dimensions are known.
    pub fn numel(&self) -> Option<usize> {
        let mut total = 1usize;
        for d in &self.0 {
            match d {
                Dim::Fixed(n) => total = total.checked_mul(*n)?,
                Dim::Dynamic => return None,
            }
        }
        Some(total)
    }

    /// Check if this shape is compatible with another (for broadcasting or connection).
    pub fn is_compatible_with(&self, other: &Shape) -> bool {
        if self.rank() != other.rank() {
            return false;
        }
        self.0.iter().zip(other.0.iter()).all(|(a, b)| match (a, b) {
            (Dim::Fixed(x), Dim::Fixed(y)) => x == y,
            _ => true, // dynamic dims are compatible with anything
        })
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match d {
                Dim::Fixed(n) => write!(f, "{n}")?,
                Dim::Dynamic => write!(f, "?")?,
            }
        }
        write!(f, "]")
    }
}

/// Complete tensor type: dtype + shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorType {
    pub dtype: Dtype,
    pub shape: Shape,
}

impl TensorType {
    pub fn new(dtype: Dtype, shape: Shape) -> Self {
        Self { dtype, shape }
    }

    pub fn f32_scalar() -> Self {
        Self::new(Dtype::F32, Shape::scalar())
    }

    pub fn f32_vector(n: usize) -> Self {
        Self::new(Dtype::F32, Shape::vector(n))
    }

    pub fn f32_matrix(m: usize, n: usize) -> Self {
        Self::new(Dtype::F32, Shape::matrix(m, n))
    }

    pub fn ternary_matrix(m: usize, n: usize) -> Self {
        Self::new(Dtype::Ternary, Shape::matrix(m, n))
    }

    /// Total memory in bytes.
    pub fn size_bytes(&self) -> Option<usize> {
        self.shape.numel().map(|n| n * self.dtype.size_bytes())
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}>{}", self.dtype, self.shape)
    }
}

/// Concrete tensor data — the actual values in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub dtype: Dtype,
    pub shape: Shape,
    pub data: Vec<u8>, // raw bytes, interpreted according to dtype
}

impl TensorData {
    /// Create a tensor filled with zeros.
    pub fn zeros(tensor_type: &TensorType) -> Option<Self> {
        let n_bytes = tensor_type.size_bytes()?;
        Some(Self {
            dtype: tensor_type.dtype,
            shape: tensor_type.shape.clone(),
            data: vec![0u8; n_bytes],
        })
    }

    /// Create a f32 tensor from a slice.
    pub fn from_f32(shape: Shape, values: &[f32]) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            dtype: Dtype::F32,
            shape,
            data,
        }
    }

    /// Read f32 values from this tensor.
    pub fn as_f32_slice(&self) -> Option<Vec<f32>> {
        if self.dtype != Dtype::F32 {
            return None;
        }
        Some(
            self.data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        )
    }

    /// Get the tensor type.
    pub fn tensor_type(&self) -> TensorType {
        TensorType::new(self.dtype, self.shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_has_one_element() {
        let s = Shape::scalar();
        assert_eq!(s.numel(), Some(1));
        assert_eq!(s.rank(), 0);
    }

    #[test]
    fn matrix_shape() {
        let s = Shape::matrix(28, 28);
        assert_eq!(s.numel(), Some(784));
        assert_eq!(s.rank(), 2);
    }

    #[test]
    fn tensor_type_display() {
        let t = TensorType::f32_matrix(28, 28);
        assert_eq!(t.to_string(), "Tensor<f32>[28, 28]");
    }

    #[test]
    fn ternary_size() {
        let t = TensorType::ternary_matrix(1024, 1024);
        assert_eq!(t.size_bytes(), Some(1024 * 1024)); // 1 byte per ternary
    }

    #[test]
    fn f32_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = TensorData::from_f32(Shape::vector(4), &values);
        assert_eq!(tensor.as_f32_slice().unwrap(), values);
    }

    #[test]
    fn shape_compatibility() {
        let a = Shape::matrix(28, 28);
        let b = Shape::matrix(28, 28);
        assert!(a.is_compatible_with(&b));

        let c = Shape::matrix(28, 14);
        assert!(!a.is_compatible_with(&c));
    }
}
