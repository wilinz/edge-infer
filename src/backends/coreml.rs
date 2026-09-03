//! 裸 Core ML 后端（iOS / macOS）。
//!
//! 与 [`super::executorch`] 的区别只在于中间少了一层：那条路是
//! PyTorch → ExecuTorch（CoreMLPartitioner）→ `.pte`，Core ML 只是 ExecuTorch
//! 里的一个 delegate，图先被切成分区、吃不下的留在 portable kernel 上，运行时
//! 还要驮着整套 ExecuTorch；这条路整张图直接交给 Core ML，模型是系统原生的
//! `.mlpackage`，链一个系统框架即可，没有第三方运行时。
//!
//! 两处与其它后端不同的约束：
//!
//! * **只能从文件加载**。Core ML 没有从内存字节建模型的入口，所以
//!   [`open_path`] 收路径而不是字节。手部检测原先走 `open_from_bytes`，
//!   接到这个后端上时要改成传路径（见 `hand_track::Model::from_path`）。
//!
//! * **`.mlpackage` 是源格式**，运行前要编译成 `.mlmodelc`。编译在 shim 里做，
//!   按源文件修改时间缓存到 caches 目录，只有首次（或换了模型）才付这个钱。
//!
//! 方法名到模型的映射沿用其它后端的约定：单方法模型统一用
//! [`Method::PrefixEnc`] 作入口，它在这里就是模型本身。

use std::ffi::{c_char, c_int, c_void, CString};

use crate::{DType, Engine, EngineConfig, EngineError, Method, Result, Tensor, TensorView};

#[repr(C)]
#[derive(Clone, Copy)]
struct CmlTensor {
    data: *mut c_void,
    shape: *const i64,
    ndim: usize,
    dtype: i32,
}

unsafe extern "C" {
    fn cml_model_load(
        path: *const c_char,
        function_name: *const c_char,
        num_threads: i32,
        err: *mut c_int,
    ) -> *mut c_void;
    fn cml_model_output_count(handle: *mut c_void) -> usize;
    fn cml_model_free(handle: *mut c_void);
    fn cml_model_run(
        handle: *mut c_void,
        inputs: *const CmlTensor,
        n_inputs: usize,
        input_names: *const *const c_char,
        outputs: *mut *mut CmlTensor,
        n_out: *mut usize,
        err: *mut c_int,
    ) -> c_int;
    fn cml_tensors_free(tensors: *mut CmlTensor, n: usize);
}

pub struct CoreMlEngine {
    handle: *mut c_void,
    /// 各输入对应的模型特征名，按位置。
    ///
    /// Core ML 是按名字喂输入的，而 Engine trait 的接口是按位置给张量，
    /// 中间这层映射必须显式钉住：模型描述那边是个字典，取出来无序，按字典序
    /// 排会把 (token_id, pos, past_kv) 排成 (past_kv, pos, token_id)。
    /// 单输入模型（手部检测）没有歧义，留空即可。
    input_names: Vec<CString>,
    /// 最近一次输出的形状，供 [`Engine::output_shape`] 回答。
    ///
    /// Core ML 的 modelDescription 在动态形状下只给出约束而非具体尺寸，
    /// 与其解析那套描述，不如记住上一次真实跑出来的——调用方（解码循环）
    /// 也只在跑过之后才问。
    last_shapes: Vec<Vec<i64>>,
}

// SAFETY: handle 只在 &mut self 下使用，跨线程共享由上层注册表加锁。
unsafe impl Send for CoreMlEngine {}

impl CoreMlEngine {
    /// 从 `.mlpackage`（或已编译的 `.mlmodelc`）路径打开。
    pub fn open_path(path: &str, cfg: &EngineConfig) -> Result<Self> {
        Self::open_function(path, None, cfg)
    }

    /// 打开多函数模型里的某个函数。`function` 为 None 时是普通单函数模型。
    pub fn open_function(path: &str, function: Option<&str>, cfg: &EngineConfig) -> Result<Self> {
        let c = CString::new(path).map_err(|_| EngineError::Load("路径含 NUL".into()))?;
        let f = match function {
            Some(n) => Some(CString::new(n).map_err(|_| EngineError::Load("函数名含 NUL".into()))?),
            None => None,
        };
        let f_ptr = f.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let mut err: c_int = 0;
        // SAFETY: c/f 在调用期间存活；handle 的所有权转给我们，Drop 里释放。
        let handle = unsafe { cml_model_load(c.as_ptr(), f_ptr, cfg.num_threads, &mut err) };
        if handle.is_null() {
            return Err(EngineError::Load(format!("Core ML 加载失败 (code={err}): {path}")));
        }
        Ok(Self { handle, input_names: Vec::new(), last_shapes: Vec::new() })
    }

    /// 模型目录下的单方法模型，按 `<name>.mlpackage` 命名。
    pub fn open_in_dir(dir: &str, name: &str, cfg: &EngineConfig) -> Result<Self> {
        Self::open_path(&format!("{dir}/{name}.mlpackage"), cfg)
    }

    /// 钉住输入名与位置的对应关系。多输入模型必须调一次。
    pub fn with_input_names(mut self, names: &[&str]) -> Result<Self> {
        self.input_names = names
            .iter()
            .map(|n| CString::new(*n).map_err(|_| EngineError::Load("输入名含 NUL".into())))
            .collect::<Result<Vec<_>>>()?;
        Ok(self)
    }
}

impl Drop for CoreMlEngine {
    fn drop(&mut self) {
        // SAFETY: handle 由 cml_model_load 分配，只在此处释放一次。
        unsafe { cml_model_free(self.handle) };
    }
}

impl Engine for CoreMlEngine {
    fn backend_name(&self) -> &'static str {
        "coreml"
    }

    fn run(&mut self, method: Method, inputs: &[TensorView<'_>]) -> Result<Vec<Tensor>> {
        // 单方法模型：一个 CoreMlEngine 就是一个模型，方法名不参与分派。
        // 多方法（识别那套 prefix_enc/prefill/step）要按方法各开一个模型，
        // 由上层组合，这里不做隐式路由。
        let mut boxed: Vec<CmlTensor> = Vec::with_capacity(inputs.len());
        for t in inputs {
            if t.dtype != DType::F32 {
                return Err(EngineError::Shape(
                    "Core ML 后端目前只收 f32 输入（模型 IO 都是 fp32）".into(),
                ));
            }
            boxed.push(CmlTensor {
                data: t.data.as_ptr() as *mut c_void,
                shape: t.shape.as_ptr(),
                ndim: t.shape.len(),
                dtype: 0,
            });
        }

        // 名字指针数组要活到调用返回，所以先落在局部变量上。
        let name_ptrs: Vec<*const c_char> =
            self.input_names.iter().map(|c| c.as_ptr()).collect();
        let names_arg = if name_ptrs.len() == inputs.len() {
            name_ptrs.as_ptr()
        } else {
            std::ptr::null()
        };

        let mut out_ptr: *mut CmlTensor = std::ptr::null_mut();
        let mut n_out: usize = 0;
        let mut err: c_int = 0;
        // SAFETY: boxed 与它引用的输入切片在调用期间都存活；输出的所有权
        // 转到这里，下面拷完立刻还给 cml_tensors_free。
        let rc = unsafe {
            cml_model_run(
                self.handle,
                boxed.as_ptr(),
                boxed.len(),
                names_arg,
                &mut out_ptr,
                &mut n_out,
                &mut err,
            )
        };
        if rc != 0 || out_ptr.is_null() {
            return Err(EngineError::Execute {
                method,
                code: err,
                msg: format!("rc={rc}"),
            });
        }

        // SAFETY: shim 保证 out_ptr 指向 n_out 个已初始化的 CmlTensor，
        // 每个的 data/shape 都是它 malloc 的、长度与 ndim/numel 相符。
        let slice = unsafe { std::slice::from_raw_parts(out_ptr, n_out) };
        let mut outs = Vec::with_capacity(n_out);
        self.last_shapes.clear();
        for t in slice {
            let shape: Vec<i64> = unsafe { std::slice::from_raw_parts(t.shape, t.ndim) }.to_vec();
            let numel: usize = shape.iter().product::<i64>().max(0) as usize;
            let data = unsafe { std::slice::from_raw_parts(t.data as *const u8, numel * 4) }.to_vec();
            self.last_shapes.push(shape.clone());
            outs.push(Tensor { shape, dtype: DType::F32, data });
        }
        unsafe { cml_tensors_free(out_ptr, n_out) };
        Ok(outs)
    }

    fn output_shape(&mut self, _method: Method, index: usize) -> Option<Vec<i64>> {
        self.last_shapes.get(index).cloned()
    }

    fn has_method(&self, method: Method) -> bool {
        // 单方法模型：只认统一入口。
        method == Method::PrefixEnc
    }
}

/// 模型声明了几个输出。加载后即可问，不必先跑一次。
pub fn output_count(engine: &CoreMlEngine) -> usize {
    // SAFETY: handle 有效。
    unsafe { cml_model_output_count(engine.handle) }
}

/// 识别那套三方法模型：一个方法一个 Core ML 模型，按 [`Method`] 路由。
///
/// ExecuTorch 那边 prefill 与 step 可以合成一个多方法 `.pte` 共享权重；
/// Core ML 的多函数模型要 iOS 18+，且 coremltools 侧的拼装另有一套流程，
/// 所以这里先按「一个方法一个文件」组织——代价是 decoder 的权重存了两份
/// （95.7 MB 对合并版 .pte 的 54.5 MB），换来的是不依赖任何第三方运行时。
///
/// 文件名约定（`model_dir` 下）：
///   `prefix_enc.mlmodelc` / `decoder_prefill.mlmodelc` / `decoder_step.mlmodelc`
/// 找不到 `.mlmodelc` 时退回同名的 `.mlpackage`（源格式，首次加载要现编）。
pub struct CoreMlPipeline {
    prefix_enc: CoreMlEngine,
    prefill: CoreMlEngine,
    step: CoreMlEngine,
}

impl CoreMlPipeline {
    pub fn open(cfg: &EngineConfig) -> Result<Self> {
        let dir = cfg.model_dir.clone();
        // 预编译的优先：.mlmodelc 是构建期用 coremlcompiler 编好的，设备上
        // 直接加载；.mlpackage 是源格式，首次加载要现编（几秒）。
        let path_of = |name: &str| -> String {
            let compiled = format!("{dir}/{name}.mlmodelc");
            if std::path::Path::new(&compiled).exists() {
                compiled
            } else {
                format!("{dir}/{name}.mlpackage")
            }
        };
        let pick = |name: &str| -> Result<CoreMlEngine> {
            CoreMlEngine::open_path(&path_of(name), cfg)
        };
        // decoder 优先用合并的多函数模型：prefill 与 step 是同一份权重的两种
        // 用法，合并后磁盘上只存一份（52MB 对拆开的 95.7MB）。
        let merged = path_of("decoder");
        let has_merged = std::path::Path::new(&merged).exists();
        // 名字与 export_coreml_ios.py 里 ct.TensorType(name=...) 钉的一致，
        // 顺序与 mwh-decode 传张量的顺序一致。
        let (prefill, step) = if has_merged {
            (
                CoreMlEngine::open_function(&merged, Some("prefill"), cfg)?,
                CoreMlEngine::open_function(&merged, Some("step"), cfg)?,
            )
        } else {
            (pick("decoder_prefill")?, pick("decoder_step")?)
        };
        Ok(Self {
            prefix_enc: pick("prefix_enc")?
                .with_input_names(&["img", "stroke", "stroke_real_len"])?,
            prefill: prefill.with_input_names(&["prefix"])?,
            step: step.with_input_names(&["token_id", "pos", "past_kv"])?,
        })
    }

    fn engine(&mut self, method: Method) -> &mut CoreMlEngine {
        match method {
            Method::PrefixEnc => &mut self.prefix_enc,
            Method::Prefill => &mut self.prefill,
            Method::Step => &mut self.step,
        }
    }
}

impl Engine for CoreMlPipeline {
    fn backend_name(&self) -> &'static str {
        "coreml"
    }

    fn run(&mut self, method: Method, inputs: &[TensorView<'_>]) -> Result<Vec<Tensor>> {
        // 各自是单方法模型，路由到对应那个后按它自己的入口跑。
        self.engine(method).run(Method::PrefixEnc, inputs)
    }

    fn output_shape(&mut self, method: Method, index: usize) -> Option<Vec<i64>> {
        self.engine(method).output_shape(Method::PrefixEnc, index)
    }

    fn has_method(&self, _method: Method) -> bool {
        true
    }
}
