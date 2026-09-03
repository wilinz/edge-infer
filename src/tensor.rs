//! 后端无关的张量表示。
//!
//! 输入用借用视图 [`TensorView`] 避免拷贝；输出用拥有所有权的 [`Tensor`]。

/// 元素类型。识别流水线只用到这三种。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    I32,
    I64,
}

impl DType {
    pub const fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}

/// 借用的输入张量。
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub shape: &'a [i64],
    pub dtype: DType,
    /// 原始字节，长度须等于 `shape` 元素积 × `dtype.size()`。
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn f32(shape: &'a [i64], data: &'a [f32]) -> Self {
        Self { shape, dtype: DType::F32, data: bytes_of(data) }
    }

    pub fn i32(shape: &'a [i64], data: &'a [i32]) -> Self {
        Self { shape, dtype: DType::I32, data: bytes_of(data) }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product::<i64>().max(0) as usize
    }
}

/// 拥有所有权的输出张量。
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn zeros(shape: Vec<i64>, dtype: DType) -> Self {
        let n = shape.iter().product::<i64>().max(0) as usize;
        Self { data: vec![0u8; n * dtype.size()], shape, dtype }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product::<i64>().max(0) as usize
    }

    /// 按 f32 解释数据。dtype 不是 F32 时返回 None。
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        Some(cast_slice(&self.data))
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView { shape: &self.shape, dtype: self.dtype, data: &self.data }
    }
}

#[inline]
fn bytes_of<T>(s: &[T]) -> &[u8] {
    // SAFETY: 只读地把 POD 切片按字节看待，长度按 size_of 换算，不越界。
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[inline]
fn cast_slice(b: &[u8]) -> &[f32] {
    // SAFETY: 调用方已确认 dtype 为 F32；Vec<u8> 起始地址满足 4 字节对齐
    // （分配器保证 ≥ 8 字节对齐）。
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}
