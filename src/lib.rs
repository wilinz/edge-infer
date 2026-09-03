// Copyright 2026 wilinz.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! 推理引擎抽象层。
//!
//! 上层（解码循环）只依赖 [`Engine`] trait，不关心底下是 LiteRT 还是 Core ML。
//!
//! 哪些后端被编进来由 feature 决定：
//!   - Android    → `litert`（识别的多签名 tflite；手部检测另走 GPU 加速器）
//!   - iOS/macOS  → `coreml`（模型是 .mlpackage，构建期编成 .mlmodelc）
//!
//! 曾经三端统一走 ExecuTorch（iOS 靠它的 Core ML delegate）。后来两边各自
//! 换掉了：Android 换 LiteRT 是因为手部检测必须离开 CPU——同在 ExecuTorch 上时
//! 两者共用那个**进程级单例**线程池，识别一跑手部检测就掉帧（28 → 22fps），而
//! GPU 加速器只在 LiteRT 2.x 上有；iOS 换裸 Core ML 是因为那边 ExecuTorch 本来
//! 也只是 Core ML 的一层壳，撤掉后延迟与准确率不变（500 条 ExpRate 74.60%
//! 逐位相同），却省掉约 10MB 静态库与整套运行时构件。
//!
//! 两个后端的模型都按同一组方法名组织：
//!   `prefix_enc` / `prefill` / `step`
//! 单签名模型（每个方法一个文件）与多签名模型（一个文件多方法）都由后端内部消化，
//! 对上层表现一致。

pub mod tensor;

pub mod backends;

pub use tensor::{DType, Tensor, TensorView};

use std::fmt;

/// 模型里的一个可调用方法。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    /// 笔画（+ 可选图像）→ prefix embeddings
    PrefixEnc,
    /// prefix embeddings → 初始 KV cache
    Prefill,
    /// (token, pos, kv) → (logits, kv)
    Step,
}

impl Method {
    /// 多签名模型内部的方法名。
    pub fn signature(self) -> &'static str {
        match self {
            Method::PrefixEnc => "forward",
            Method::Prefill => "prefill",
            Method::Step => "step",
        }
    }
}

#[derive(Debug)]
pub enum EngineError {
    /// 模型文件缺失或格式不对。
    Load(String),
    /// 方法不存在（例如单签名模型里找 `prefill`）。
    MethodNotFound(Method),
    /// 输入形状与模型不符。
    Shape(String),
    /// 后端执行失败，带原始错误码。
    Execute { method: Method, code: i32, msg: String },
    /// 该后端未编入本次构建。
    Unavailable(&'static str),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Load(m) => write!(f, "模型加载失败: {m}"),
            EngineError::MethodNotFound(m) => write!(f, "方法不存在: {m:?}"),
            EngineError::Shape(m) => write!(f, "形状不匹配: {m}"),
            EngineError::Execute { method, code, msg } => {
                write!(f, "{method:?} 执行失败 (code={code}): {msg}")
            }
            EngineError::Unavailable(b) => write!(f, "后端未编入本次构建: {b}"),
        }
    }
}

impl std::error::Error for EngineError {}

pub type Result<T> = std::result::Result<T, EngineError>;

/// 后端无关的推理引擎。
///
/// 实现者需保证 `&mut self` 期间独占底层解释器；跨线程共享由上层的注册表加锁完成。
pub trait Engine: Send {
    /// 后端名，用于日志与基准报告。
    fn backend_name(&self) -> &'static str;

    /// 执行一个方法。输入按模型签名顺序给出，返回全部输出张量。
    fn run(&mut self, method: Method, inputs: &[TensorView<'_>]) -> Result<Vec<Tensor>>;

    /// 查询某方法的输出形状（部分后端在动态形状下返回 None）。
    fn output_shape(&mut self, method: Method, index: usize) -> Option<Vec<i64>>;

    /// 是否支持该方法。
    fn has_method(&self, method: Method) -> bool;

    /// 声明某个方法的第 `out_index` 个输出，下一次调用时直接作为第 `in_index`
    /// 个输入使用，中间不经过调用方。
    ///
    /// 用于自回归解码的 KV cache：每步把整块 cache 送进去再整块取回，是纯粹
    /// 的搬运，而每步真正变化的只有一个槽位。后端若支持就返回 true，之后调用
    /// 方给这个位置的张量会被忽略，对应的输出也不再回传。
    ///
    /// 默认不支持。
    fn feed_output_back(&mut self, _method: Method, _out_index: usize, _in_index: usize) -> bool {
        false
    }

    /// 声明 `from` 方法的第 `out_index` 个输出，直接作为 `to` 方法第 `in_index`
    /// 个输入使用。
    ///
    /// 用于 prefill 算出的 KV 交给 decode 的第一步：两者若在同一个模型里，
    /// 缓冲可以直接对接，省掉一趟往返。声明成功后调用方给这两个位置的张量
    /// 都会被忽略。默认不支持。
    fn pipe_output(
        &mut self,
        _from: Method,
        _out_index: usize,
        _to: Method,
        _in_index: usize,
    ) -> bool {
        false
    }
}

/// 引擎构造参数。
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// 模型目录，各后端按自己的命名约定在其中找文件。
    pub model_dir: String,
    /// CPU 线程数；0 表示后端默认。
    /// LiteRT 走 XNNPACK 时建议显式设 4，避开大小核架构的调度拖累。
    pub num_threads: i32,
    /// 想要哪个后端。编进来的后端里挑不到时退回可用的那个。
    pub prefer: Prefer,
}

/// 后端偏好。
///
/// 不是「用哪个」的硬性指定，而是偏好：只编了一个后端的构建（iOS/macOS）
/// 上，这个字段没有意义，退回唯一可用的那个即可。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Prefer {
    /// 按编译期顺序挑。识别走这条。
    #[default]
    Auto,
    /// LiteRT，并尽量启用 GPU delegate。手部检测走这条：它的模型小、
    /// 层数浅，正好是 GPU delegate 划算的那一类，而且能把它从 CPU 上挪开。
    LiteRtGpu,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { model_dir: String::new(), num_threads: 0, prefer: Prefer::Auto }
    }
}


/// 从内存字节打开一个单方法模型。
///
/// 手部检测的两个模型是独立的单方法 PTE（只有 `forward`），与识别那套
/// 多方法模型不同，所以用 [`Method::PrefixEnc`] 作为它们的统一入口——
/// 那个枚举值在后端里映射到方法名 "forward"。
pub fn open_from_bytes(bytes: &[u8], cfg: &EngineConfig) -> Result<Box<dyn Engine>> {
    // 先看偏好。两个后端都编进来时（Android）由它决定；只编了一个时
    // 下面的兜底分支接管。
    #[cfg(feature = "litert")]
    if cfg.prefer == Prefer::LiteRtGpu {
        // 走 LiteRT 2.x 的 CompiledModel：GPU 加速器只在那条路上，经典的
        // TfLiteGpuDelegateV2 在 2.x 运行时里已经没有了。
        return backends::litert::LiteRtCompiledEngine::open_bytes(bytes, cfg)
            .map(|e| Box::new(e) as Box<dyn Engine>);
    }
    #[cfg(feature = "litert")]
    {
        return backends::litert::LiteRtCompiledEngine::open_bytes(bytes, cfg)
            .map(|e| Box::new(e) as Box<dyn Engine>);
    }
    #[cfg(not(feature = "litert"))]
    {
        let _ = (bytes, cfg);
        Err(EngineError::Unavailable("未编入任何推理后端"))
    }
}

/// 从文件路径打开一个单方法模型。
///
/// 与 [`open_from_bytes`] 并列，多这一条是因为 Core ML 只能从文件加载——它
/// 没有从内存字节建模型的入口，`.mlpackage` 还得先编译成 `.mlmodelc`。其它
/// LiteRT 本来就能收字节，这里读进来转交即可，调用方
/// 不必按平台分叉。
///
/// 按扩展名分派：`.mlpackage` / `.mlmodelc` 走 Core ML，其余走字节那条。
pub fn open_path(path: &str, cfg: &EngineConfig) -> Result<Box<dyn Engine>> {
    #[cfg(feature = "coreml")]
    if path.ends_with(".mlpackage") || path.ends_with(".mlmodelc") {
        return backends::coreml::CoreMlEngine::open_path(path, cfg)
            .map(|e| Box::new(e) as Box<dyn Engine>);
    }
    let bytes = std::fs::read(path)
        .map_err(|e| EngineError::Load(format!("读模型失败 {path}: {e}")))?;
    open_from_bytes(&bytes, cfg)
}

/// 把一行诊断写到平台日志。
///
/// Android 上应用进程的 stderr 是丢弃的，`eprintln!` 看不见，所以走 liblog
/// （build.rs 里本来就链了 `-llog`）。其他平台仍用 stderr。各 crate 共用这一
/// 个出口，省得每处都抄一遍 `__android_log_write` 的声明。
pub fn log_line(msg: &str) {
    #[cfg(target_os = "android")]
    {
        unsafe extern "C" {
            fn __android_log_write(
                prio: i32,
                tag: *const std::ffi::c_char,
                text: *const std::ffi::c_char,
            ) -> i32;
        }
        // 4 = ANDROID_LOG_INFO
        if let Ok(text) = std::ffi::CString::new(msg) {
            unsafe { __android_log_write(4, c"mwh".as_ptr(), text.as_ptr()) };
        }
    }
    #[cfg(not(target_os = "android"))]
    eprintln!("[mwh] {msg}");
}
