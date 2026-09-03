//! LiteRT 2.x 的 CompiledModel API 绑定。
//!
//! 与隔壁 `tflite_sys` 的区别不只是新旧：LiteRT 2.x 把 GPU 从「delegate」改成
//! 了「accelerator」，`TfLiteGpuDelegateV2Create` 这条路在 2.x 的运行时里已经
//! 不存在，只能走这套 Environment → Options → CompiledModel → TensorBuffer。
//!
//! 为什么值得为它单写一套绑定：手部检测要用 GPU。这台机器上 TFLite 的
//! OpenCL 后端起不来（应用进程 dlopen 不到 vendor 的 SPHAL 命名空间），只有
//! OpenGL 那条能用；而官方 AAR 里的 `libLiteRtClGlAccelerator.so` 两个后端都
//! 带着，正是 MediaPipe 当年跑到 30fps 用的同一类实现。
//!
//! 符号全部来自 `libLiteRt.so`（com.google.ai.edge.litert:litert 的 AAR）。
//! 加速器那个 `.so` 由运行时按名字自行加载，不参与链接，只要跟着进
//! `lib/<abi>/` 就行。

#![allow(non_camel_case_types, non_upper_case_globals)]

use std::ffi::{c_char, c_void, CStr};
use std::ptr;

// ── C 类型 ───────────────────────────────────────────────────────────────

pub type LiteRtStatus = i32;
pub const kLiteRtStatusOk: LiteRtStatus = 0;

/// 不透明句柄，全部是 `struct X*`。
type Handle = *mut c_void;

pub type LiteRtEnvironment = Handle;
pub type LiteRtModel = Handle;
pub type LiteRtOptions = Handle;
pub type LiteRtCompiledModel = Handle;
pub type LiteRtTensorBuffer = Handle;
pub type LiteRtSignature = Handle;
pub type LiteRtTensor = Handle;
pub type LiteRtTensorBufferRequirements = Handle;
pub type LiteRtOpaqueOptions = Handle;
pub type LiteRtProfiler = Handle;

/// `LiteRtHwAccelerators` 的位掩码。
pub const kLiteRtHwAcceleratorCpu: i32 = 1 << 0;
pub const kLiteRtHwAcceleratorGpu: i32 = 1 << 1;
/// NPU 位。libLiteRt.so 里确有 `LiteRtRegisterNpuAccelerator`，但它走 Dispatch
/// API，要另外提供厂商的 dispatch 库（库里的提示：「You should provide the
/// `DispatchLibraryDir` option to use NPU.」）。我们没打包任何 dispatch 库，
/// 这台骁龙 870 上也只有 SNPE/DSP 那套、没有 QNN HTP 运行时。实测把这个位
/// 加进掩码，LiteRT 只会打一行
/// `[compiled_model.cc:986] You should provide the DispatchLibraryDir option to
/// use NPU.`，委托结果和不加时完全一样（prefill 455/455、decode 484/484 全在
/// LITERT_CL）。所以常量留着备查，掩码里不带。
pub const kLiteRtHwAcceleratorNpu: i32 = 1 << 2;

pub const kLiteRtElementTypeFloat32: i32 = 1;
pub const kLiteRtElementTypeInt32: i32 = 2;


pub const kLiteRtTensorBufferLockModeRead: i32 = 0;
pub const kLiteRtTensorBufferLockModeWrite: i32 = 1;

/// 与 `litert/c/litert_layout.h` 的 `LiteRtLayout` 布局一致。
///
/// `rank` 与 `has_strides` 是同一个 `unsigned int` 存储单元里的位域（头文件
/// 里特意注明两者用同一底层类型，好让 MSVC 与 Clang 的布局一致），所以这里
/// 用一个 `u32` 表示，低 7 位是 rank，第 8 位是 has_strides。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LiteRtLayout {
    pub packed: u32,
    pub dimensions: [i32; 8],
    pub strides: [u32; 8],
}

impl LiteRtLayout {
    pub fn rank(&self) -> usize {
        (self.packed & 0x7f) as usize
    }

    pub fn dims(&self) -> Vec<i32> {
        self.dimensions[..self.rank().min(8)].to_vec()
    }

    pub fn zeroed() -> Self {
        Self { packed: 0, dimensions: [0; 8], strides: [0; 8] }
    }

    pub fn from_dims(dims: &[i32]) -> Self {
        let mut dimensions = [0i32; 8];
        let rank = dims.len().min(8);
        dimensions[..rank].copy_from_slice(&dims[..rank]);
        Self {
            // has_strides = 0：我们只用紧凑布局。
            packed: rank as u32 & 0x7f,
            dimensions,
            strides: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LiteRtRankedTensorType {
    pub element_type: i32,
    pub layout: LiteRtLayout,
}

#[link(name = "LiteRt")]
unsafe extern "C" {
    fn LiteRtCreateEnvironment(
        num_options: i32,
        options: *const c_void,
        environment: *mut LiteRtEnvironment,
    ) -> LiteRtStatus;
    fn LiteRtDestroyEnvironment(environment: LiteRtEnvironment);

    fn LiteRtCreateModelFromBuffer(
        environment: LiteRtEnvironment,
        buffer_addr: *const c_void,
        buffer_size: usize,
        model: *mut LiteRtModel,
    ) -> LiteRtStatus;
    fn LiteRtDestroyModel(model: LiteRtModel);

    fn LiteRtCreateOptions(options: *mut LiteRtOptions) -> LiteRtStatus;
    fn LiteRtSetOptionsHardwareAccelerators(
        options: LiteRtOptions,
        accelerators: i32,
    ) -> LiteRtStatus;
    fn LiteRtDestroyOptions(options: LiteRtOptions);

    fn LiteRtCreateOpaqueOptions(
        payload_identifier: *const c_char,
        payload_data: *mut c_void,
        payload_destructor: Option<unsafe extern "C" fn(*mut c_void)>,
        options: *mut LiteRtOpaqueOptions,
    ) -> LiteRtStatus;
    fn LiteRtAddOpaqueOptions(
        options: LiteRtOptions,
        opaque_options: LiteRtOpaqueOptions,
    ) -> LiteRtStatus;

    fn LiteRtCreateCompiledModel(
        environment: LiteRtEnvironment,
        model: LiteRtModel,
        compilation_options: LiteRtOptions,
        compiled_model: *mut LiteRtCompiledModel,
    ) -> LiteRtStatus;
    fn LiteRtRunCompiledModel(
        compiled_model: LiteRtCompiledModel,
        signature_index: usize,
        num_input_buffers: usize,
        input_buffers: *mut LiteRtTensorBuffer,
        num_output_buffers: usize,
        output_buffers: *mut LiteRtTensorBuffer,
    ) -> LiteRtStatus;
    fn LiteRtDestroyCompiledModel(compiled_model: LiteRtCompiledModel);

    fn LiteRtCreateManagedTensorBuffer(
        env: LiteRtEnvironment,
        buffer_type: i32,
        tensor_type: *const LiteRtRankedTensorType,
        buffer_size: usize,
        buffer: *mut LiteRtTensorBuffer,
    ) -> LiteRtStatus;
    fn LiteRtLockTensorBuffer(
        tensor_buffer: LiteRtTensorBuffer,
        host_mem_addr: *mut *mut c_void,
        lock_mode: i32,
    ) -> LiteRtStatus;
    fn LiteRtUnlockTensorBuffer(buffer: LiteRtTensorBuffer) -> LiteRtStatus;
    fn LiteRtDestroyTensorBuffer(buffer: LiteRtTensorBuffer);

    fn LiteRtGetCompiledModelOutputTensorLayouts(
        compiled_model: LiteRtCompiledModel,
        signature_index: usize,
        num_layouts: usize,
        layouts: *mut LiteRtLayout,
        update_allocation: bool,
    ) -> LiteRtStatus;
    fn LiteRtGetCompiledModelInputTensorLayout(
        compiled_model: LiteRtCompiledModel,
        signature_index: usize,
        input_index: usize,
        layout: *mut LiteRtLayout,
    ) -> LiteRtStatus;

    fn LiteRtGetNumModelSignatures(model: LiteRtModel, num: *mut usize) -> LiteRtStatus;
    fn LiteRtGetModelSignature(
        model: LiteRtModel,
        signature_index: usize,
        signature: *mut LiteRtSignature,
    ) -> LiteRtStatus;
    fn LiteRtGetSignatureKey(
        signature: LiteRtSignature,
        signature_key: *mut *const c_char,
    ) -> LiteRtStatus;
    fn LiteRtGetNumSignatureInputs(signature: LiteRtSignature, num: *mut usize) -> LiteRtStatus;
    fn LiteRtGetNumSignatureOutputs(signature: LiteRtSignature, num: *mut usize) -> LiteRtStatus;
    fn LiteRtGetSignatureInputName(
        signature: LiteRtSignature,
        input_idx: usize,
        input_name: *mut *const c_char,
    ) -> LiteRtStatus;
    fn LiteRtGetSignatureOutputName(
        signature: LiteRtSignature,
        output_idx: usize,
        output_name: *mut *const c_char,
    ) -> LiteRtStatus;
    fn LiteRtGetSignatureInputTensorByIndex(
        signature: LiteRtSignature,
        input_idx: usize,
        tensor: *mut LiteRtTensor,
    ) -> LiteRtStatus;
    fn LiteRtGetSignatureOutputTensorByIndex(
        signature: LiteRtSignature,
        output_idx: usize,
        tensor: *mut LiteRtTensor,
    ) -> LiteRtStatus;
    fn LiteRtGetRankedTensorType(
        tensor: LiteRtTensor,
        ranked_tensor_type: *mut LiteRtRankedTensorType,
    ) -> LiteRtStatus;

    fn LiteRtGetCompiledModelInputBufferRequirements(
        compiled_model: LiteRtCompiledModel,
        signature_index: usize,
        input_index: usize,
        buffer_requirements: *mut LiteRtTensorBufferRequirements,
    ) -> LiteRtStatus;
    fn LiteRtGetCompiledModelOutputBufferRequirements(
        compiled_model: LiteRtCompiledModel,
        signature_index: usize,
        output_index: usize,
        buffer_requirements: *mut LiteRtTensorBufferRequirements,
    ) -> LiteRtStatus;
    fn LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
        requirements: LiteRtTensorBufferRequirements,
        num_types: *mut i32,
    ) -> LiteRtStatus;
    fn LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
        requirements: LiteRtTensorBufferRequirements,
        type_index: i32,
        ty: *mut i32,
    ) -> LiteRtStatus;
    fn LiteRtGetTensorBufferRequirementsBufferSize(
        requirements: LiteRtTensorBufferRequirements,
        buffer_size: *mut usize,
    ) -> LiteRtStatus;
    fn LiteRtGetTensorBufferRequirementsStrides(
        requirements: LiteRtTensorBufferRequirements,
        num_strides: *mut i32,
        strides: *mut *const u32,
    ) -> LiteRtStatus;
    fn LiteRtCompiledModelGetProfiler(
        compiled_model: LiteRtCompiledModel,
        profiler: *mut LiteRtProfiler,
    ) -> LiteRtStatus;
    fn LiteRtStartProfiler(profiler: LiteRtProfiler) -> LiteRtStatus;
    fn LiteRtStopProfiler(profiler: LiteRtProfiler) -> LiteRtStatus;
    fn LiteRtResetProfiler(profiler: LiteRtProfiler) -> LiteRtStatus;
    fn LiteRtGetProfileSummary(
        profiler: LiteRtProfiler,
        compiled_model: LiteRtCompiledModel,
        summary: *mut *const c_char,
    ) -> LiteRtStatus;

    fn LiteRtCreateManagedTensorBufferFromRequirements(
        env: LiteRtEnvironment,
        tensor_type: *const LiteRtRankedTensorType,
        requirements: LiteRtTensorBufferRequirements,
        buffer: *mut LiteRtTensorBuffer,
    ) -> LiteRtStatus;
}

pub const kLiteRtTensorBufferTypeHostMemory: i32 = 1;

// ── 安全封装 ─────────────────────────────────────────────────────────────

/// 进程级的 LiteRT 环境。
///
/// GPU 加速器在环境里初始化一次（建 EGL 上下文、探测设备），每个模型再建
/// 一次是纯浪费，所以做成单例。
fn environment() -> Result<LiteRtEnvironment, String> {
    use std::sync::OnceLock;
    static ENV: OnceLock<usize> = OnceLock::new();
    let raw = *ENV.get_or_init(|| {
        let mut env: LiteRtEnvironment = ptr::null_mut();
        let st = unsafe { LiteRtCreateEnvironment(0, ptr::null(), &mut env) };
        if st != kLiteRtStatusOk {
            return 0;
        }
        env as usize
    });
    if raw == 0 {
        return Err("LiteRtCreateEnvironment 失败".into());
    }
    Ok(raw as LiteRtEnvironment)
}

/// 模型里的一个签名，连同它自己的输入输出缓冲。
///
/// 识别的 `decoder.tflite` 里有 `prefill` 与 `decode` 两个签名共享一份权重；
/// 手部检测那两个模型各只有一个。两种情况用同一套结构，区别只是签名个数。
struct Signature {
    key: String,
    inputs: Vec<LiteRtTensorBuffer>,
    outputs: Vec<LiteRtTensorBuffer>,
    input_shapes: Vec<Vec<i32>>,
    output_shapes: Vec<Vec<i32>>,
    /// 每个输入输出的 `LiteRtElementType`。识别的 prefix_enc 第三个输入是
    /// int32（笔画真实长度），一律按 f32 分配会让缓冲小一半、类型也不匹配。
    input_types: Vec<i32>,
    output_types: Vec<i32>,
}

/// 一个已编译好的模型。
pub struct CompiledModel {
    /// 模型字节。必须由这里持有到模型销毁为止——`LiteRtCreateModelFromBuffer`
    /// 不复制，权重自始至终指向这块内存（头文件原话："The caller must ensure
    /// that the buffer remains valid for the lifetime of the model."）。
    /// 让调用方的临时 Vec 出作用域，权重就成了悬垂指针：前几层还能算出看着
    /// 正常的数，越往后越是读到被复用的内存，最终 logits 全是 NaN。
    _bytes: std::sync::Arc<Vec<u8>>,
    model: LiteRtModel,
    compiled: LiteRtCompiledModel,
    signatures: Vec<Signature>,
    /// 实际拿到的是不是 GPU。日志用。
    pub on_gpu: bool,
}

// SAFETY: 句柄本身不带线程亲和性，外层用 &mut 独占。
unsafe impl Send for CompiledModel {}

impl CompiledModel {
    /// 从内存字节编译。
    ///
    /// 形状不用调用方给：编译完成后 `LiteRtGetCompiledModel*TensorLayout`
    /// 就能逐个签名查出来。这跟经典 API 那侧要先 `declare_input_shape` 不同
    /// ——那边是解释器建好前形状还没定，这边编译时已经定死了。
    ///
    /// GPU 要不到就退回 CPU：加速器 `.so` 可能没打进包，设备也可能既没有可用
    /// 的 OpenCL 也没有 OpenGL ES 3。退回后仍然可用，只是慢。
    pub fn new(model_bytes: &[u8], want_gpu: bool, num_threads: i32) -> Result<Self, String> {
        Self::from_shared(std::sync::Arc::new(model_bytes.to_vec()), want_gpu, num_threads)
    }

    /// 同 [`from_shared`]，接管调用方已经读好的字节。
    pub fn from_owned(bytes: Vec<u8>, want_gpu: bool, num_threads: i32) -> Result<Self, String> {
        Self::from_shared(std::sync::Arc::new(bytes), want_gpu, num_threads)
    }

    /// 用一份共享的模型字节编译。
    ///
    /// 同一个文件可以编译多次（比如 prefill 交给 GPU、decode 留在 CPU），
    /// 而 `LiteRtCreateModelFromBuffer` 不复制字节，所以几十 MB 的权重只需
    /// 在内存里存一份，由 `Arc` 保证活到所有模型销毁。
    pub fn from_shared(
        bytes: std::sync::Arc<Vec<u8>>,
        want_gpu: bool,
        num_threads: i32,
    ) -> Result<Self, String> {
        let env = environment()?;

        let mut model: LiteRtModel = ptr::null_mut();
        let st = unsafe {
            LiteRtCreateModelFromBuffer(
                env,
                bytes.as_ptr() as *const c_void,
                bytes.len(),
                &mut model,
            )
        };
        if st != kLiteRtStatusOk {
            return Err(format!("LiteRtCreateModelFromBuffer status={st}"));
        }

        // 签名的名字和输入输出个数从模型对象上问，编译前就能拿到。
        let sig_meta = match unsafe { signature_metadata(model) } {
            Ok(v) => v,
            Err(e) => {
                unsafe { LiteRtDestroyModel(model) };
                return Err(e);
            }
        };

        // 编译与建缓冲要一起成败：GPU 编得出来、缓冲却建不出来的情况是常态
        // （GL buffer 要调用线程上有 EGL 上下文，而推理跑在常驻 worker 线程
        // 上），只回退编译那半步等于没回退。
        let (compiled, signatures, on_gpu) = match Self::build(env, model, &sig_meta, want_gpu, num_threads) {
            Ok(v) => (v.0, v.1, want_gpu),
            Err(e) if want_gpu => match Self::build(env, model, &sig_meta, false, num_threads) {
                Ok(v) => (v.0, v.1, false),
                Err(e2) => {
                    unsafe { LiteRtDestroyModel(model) };
                    return Err(format!("GPU 与 CPU 都失败: {e} / {e2}"));
                }
            },
            Err(e) => {
                unsafe { LiteRtDestroyModel(model) };
                return Err(e);
            }
        };

        Ok(Self { _bytes: bytes, model, compiled, signatures, on_gpu })
    }

    /// 编译一次，并把每个签名的缓冲建好。任何一步失败都把已建的东西收干净，
    /// 好让调用方原样换个硬件档再试。
    fn build(
        env: LiteRtEnvironment,
        model: LiteRtModel,
        sig_meta: &[SignatureMeta],
        gpu: bool,
        num_threads: i32,
    ) -> Result<(LiteRtCompiledModel, Vec<Signature>), String> {
        let compiled = Self::compile(env, model, gpu, num_threads)?;

        let mut signatures = Vec::with_capacity(sig_meta.len());
        for (idx, meta) in sig_meta.iter().enumerate() {
            let built = (|| -> Result<Signature, String> {
                // 形状以编译结果为准（编译时可能被解析/重排），类型以模型里
                // 声明的为准——编译不会改数据类型。
                let input_types = meta.input_types.clone();
                let output_types = meta.output_types.clone();
                let input_shapes = input_layouts(compiled, idx, input_types.len())?;
                let output_shapes = output_layouts(compiled, idx, output_types.len())?;
                let inputs =
                    Self::alloc_buffers(env, compiled, idx, true, &input_shapes, &input_types)?;
                let outputs =
                    Self::alloc_buffers(env, compiled, idx, false, &output_shapes, &output_types)?;
                // 按调用方序号重排。运行时给的顺序是参数名的字典序
                // （args_10 排在 args_2 前面），而上层是按位置传参的。
                // 在这里排好，外面所有按下标的访问就都对了。
                fn pick<T: Clone>(v: &[T], order: &[usize]) -> Vec<T> {
                    order.iter().map(|&i| v[i].clone()).collect()
                }
                Ok(Signature {
                    key: meta.key.clone(),
                    inputs: meta.input_order.iter().map(|&i| inputs[i]).collect(),
                    outputs: meta.output_order.iter().map(|&i| outputs[i]).collect(),
                    input_shapes: pick(&input_shapes, &meta.input_order),
                    output_shapes: pick(&output_shapes, &meta.output_order),
                    input_types: pick(&input_types, &meta.input_order),
                    output_types: pick(&output_types, &meta.output_order),
                })
            })();
            match built {
                Ok(sig) => signatures.push(sig),
                Err(e) => {
                    unsafe {
                        for s in &signatures {
                            for &b in s.inputs.iter().chain(s.outputs.iter()) {
                                LiteRtDestroyTensorBuffer(b);
                            }
                        }
                        LiteRtDestroyCompiledModel(compiled);
                    }
                    return Err(e);
                }
            }
        }
        Ok((compiled, signatures))
    }

    fn compile(
        env: LiteRtEnvironment,
        model: LiteRtModel,
        gpu: bool,
        num_threads: i32,
    ) -> Result<LiteRtCompiledModel, String> {
        let mut options: LiteRtOptions = ptr::null_mut();
        let st = unsafe { LiteRtCreateOptions(&mut options) };
        if st != kLiteRtStatusOk {
            return Err(format!("LiteRtCreateOptions status={st}"));
        }
        // GPU 那档也把 CPU 一起给上：算子不被加速器接受时由 CPU 兜底，
        // 而不是整个编译失败。
        // 至少要声明一个加速器：一个都不给时 LiteRtCreateCompiledModel 直接
        // 报 status=1。GPU 那档把 CPU 一起给上，算子不被加速器接受时由 CPU
        // 兜底，而不是整个编译失败。
        let hw = if gpu {
            kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu
        } else {
            kLiteRtHwAcceleratorCpu
        };
        unsafe { LiteRtSetOptionsHardwareAccelerators(options, hw) };

        // 线程数两档都要设。GPU 是 partial offload：encoder 只有一部分算子
        // 上 GPU，其余仍由 XNNPACK 在 CPU 上跑，漏设就是让剩下那些算子单线程
        // 执行——曾据此错误地得出「encoder 上 GPU 更慢」的结论。
        set_cpu_num_threads(options, num_threads);

        let mut compiled: LiteRtCompiledModel = ptr::null_mut();
        let st = unsafe { LiteRtCreateCompiledModel(env, model, options, &mut compiled) };
        unsafe { LiteRtDestroyOptions(options) };
        if st != kLiteRtStatusOk {
            return Err(format!("LiteRtCreateCompiledModel status={st}"));
        }
        Ok(compiled)
    }

    fn alloc_buffers(
        env: LiteRtEnvironment,
        compiled: LiteRtCompiledModel,
        sig: usize,
        is_input: bool,
        shapes: &[Vec<i32>],
        types: &[i32],
    ) -> Result<Vec<LiteRtTensorBuffer>, String> {
        let mut out = Vec::with_capacity(shapes.len());
        for (i, (shape, &ty)) in shapes.iter().zip(types).enumerate() {
            let tensor_type = LiteRtRankedTensorType {
                element_type: ty,
                layout: LiteRtLayout::from_dims(shape),
            };

            // 缓冲怎么建，交给运行时按加速器的 requirements 决定，别自己挑
            // 类型再算大小：GPU 那条路上算子跑在 OpenGL 上，它要的是 GL
            // buffer，还带着自己的对齐与 stride 要求。手工用
            // LiteRtCreateManagedTensorBuffer 挑一个类型塞进去，编译能过、
            // 每帧 invoke 会失败（"Node number N (LITERT_OPENGL) failed to
            // invoke"）——这正是 C++ 封装里 CreateInputOutputBuffers 走
            // FromRequirements 的原因。
            let mut req: LiteRtTensorBufferRequirements = ptr::null_mut();
            let st = unsafe {
                if is_input {
                    LiteRtGetCompiledModelInputBufferRequirements(compiled, sig, i, &mut req)
                } else {
                    LiteRtGetCompiledModelOutputBufferRequirements(compiled, sig, i, &mut req)
                }
            };
            if st != kLiteRtStatusOk || req.is_null() {
                return Err(format!("取第 {i} 个张量的缓冲需求 status={st}"));
            }

            let mut buf: LiteRtTensorBuffer = ptr::null_mut();
            let st = unsafe {
                LiteRtCreateManagedTensorBufferFromRequirements(
                    env,
                    &tensor_type,
                    req,
                    &mut buf,
                )
            };
            if st != kLiteRtStatusOk || buf.is_null() {
                return Err(format!(
                    "LiteRtCreateManagedTensorBufferFromRequirements(第 {i} 个) status={st}"
                ));
            }
            // 新建的缓冲是 malloc 出来的，内容是脏的。识别的 KV cache 只有前
            // n_prefix 个位置被 prefill 写过，其余位置虽然会被 attention 的
            // mask 屏蔽，但屏蔽是「乘以 0」——脏内存里若有 NaN/Inf，0×NaN 仍
            // 是 NaN，logits 整片烂掉。清一次零，之后每帧都是覆盖写，不再有
            // 这个开销。
            zero_buffer(buf, size_of_buffer(req));
            out.push(buf);
        }
        Ok(out)
    }

    /// 按名字找签名下标。单签名模型（手部检测）不看名字，直接给 0——它的
    /// 键是转换器起的 `serving_default` 之类，与我们的 Method 对不上，
    /// 而且也没有第二个可选。
    pub fn signature_index(&self, key: Option<&str>) -> Option<usize> {
        match key {
            _ if self.signatures.len() == 1 => Some(0),
            Some(k) => self.signatures.iter().position(|s| s.key == k),
            None => None,
        }
    }

    /// 一行描述某个签名的输入输出，排错用：顺序、形状、元素类型都在里面。
    pub fn describe(&self, sig: usize) -> String {
        let Some(s) = self.signatures.get(sig) else {
            return format!("签名 {sig} 不存在");
        };
        format!(
            "key={} in={:?}/{:?} out={:?}/{:?}",
            s.key, s.input_shapes, s.input_types, s.output_shapes, s.output_types
        )
    }

    pub fn signature_keys(&self) -> Vec<&str> {
        self.signatures.iter().map(|s| s.key.as_str()).collect()
    }

    pub fn input_shape(&self, sig: usize, index: usize) -> Vec<i32> {
        self.signatures
            .get(sig)
            .and_then(|s| s.input_shapes.get(index))
            .cloned()
            .unwrap_or_default()
    }

    pub fn output_shape(&self, sig: usize, index: usize) -> Vec<i32> {
        self.signatures
            .get(sig)
            .and_then(|s| s.output_shapes.get(index))
            .cloned()
            .unwrap_or_default()
    }

    pub fn output_count(&self, sig: usize) -> usize {
        self.signatures.get(sig).map_or(0, |s| s.outputs.len())
    }

    /// 一个输入张量的字节数。
    pub fn input_bytes(&self, sig: usize, index: usize) -> usize {
        self.signatures.get(sig).map_or(0, |s| {
            let (Some(shape), Some(&ty)) = (s.input_shapes.get(index), s.input_types.get(index))
            else {
                return 0;
            };
            numel_of(shape) * element_size(ty)
        })
    }

    /// 一个输出张量的字节数。
    pub fn output_bytes(&self, sig: usize, index: usize) -> usize {
        self.signatures.get(sig).map_or(0, |s| {
            let (Some(shape), Some(&ty)) = (s.output_shapes.get(index), s.output_types.get(index))
            else {
                return 0;
            };
            numel_of(shape) * element_size(ty)
        })
    }

    /// 输出的元素类型（`LiteRtElementType`）。
    pub fn output_type(&self, sig: usize, index: usize) -> i32 {
        self.signatures
            .get(sig)
            .and_then(|s| s.output_types.get(index).copied())
            .unwrap_or(kLiteRtElementTypeFloat32)
    }

    /// 按原始字节写一个输入。不足按零补，多余截断。
    ///
    /// 走字节而不是 `&[f32]`：prefix_enc 的第三个输入是 int32，多一层
    /// f32 转换既没必要也会把它写坏。调用方的 `TensorView` 本来就是字节。
    pub fn set_input_bytes(&mut self, sig: usize, index: usize, data: &[u8]) {
        let cap = self.input_bytes(sig, index);
        let Some(s) = self.signatures.get(sig) else { return };
        let Some(&buf) = s.inputs.get(index) else { return };
        if cap == 0 {
            return;
        }
        let mut addr: *mut c_void = ptr::null_mut();
        if unsafe { LiteRtLockTensorBuffer(buf, &mut addr, kLiteRtTensorBufferLockModeWrite) }
            != kLiteRtStatusOk
            || addr.is_null()
        {
            return;
        }
        // SAFETY: 缓冲按 cap 字节分配，写入长度取二者较小值。
        unsafe {
            let dst = std::slice::from_raw_parts_mut(addr as *mut u8, cap);
            let n = data.len().min(cap);
            dst[..n].copy_from_slice(&data[..n]);
            if n < cap {
                dst[n..].fill(0);
            }
            LiteRtUnlockTensorBuffer(buf);
        }
    }

    /// 把某个签名的输出缓冲与输入缓冲对调。
    ///
    /// 自回归解码每步都要把整块 KV cache 送进去、再整块取回，而每步真正变化
    /// 的只有一个槽位。对调之后，上一步的输出缓冲直接成为下一步的输入缓冲，
    /// 数据不再经过 CPU——GPU 上尤其值钱，省掉一次显存回传和一次上传。
    ///
    /// 两个缓冲的形状与类型必须一致（KV 的进出正好如此），否则不做处理。
    pub fn swap_io_buffers(&mut self, sig: usize, in_idx: usize, out_idx: usize) -> bool {
        let Some(s) = self.signatures.get_mut(sig) else { return false };
        let (Some(is), Some(os)) = (s.input_shapes.get(in_idx), s.output_shapes.get(out_idx))
        else {
            return false;
        };
        if is != os {
            return false;
        }
        if in_idx >= s.inputs.len() || out_idx >= s.outputs.len() {
            return false;
        }
        std::mem::swap(&mut s.inputs[in_idx], &mut s.outputs[out_idx]);
        true
    }

    /// 取运行时的逐算子耗时汇总。需要建模型时开了 `enable_profiling`
    /// （环境变量 MWH_PROFILE，见 compile）。
    pub fn profile_summary(&mut self) -> Option<String> {
        let prof = self.profiler()?;
        let mut txt: *const c_char = ptr::null();
        if unsafe { LiteRtGetProfileSummary(prof, self.compiled, &mut txt) } != kLiteRtStatusOk
            || txt.is_null()
        {
            return None;
        }
        Some(unsafe { CStr::from_ptr(txt) }.to_string_lossy().into_owned())
    }

    /// 清零并开始采集。
    pub fn profile_start(&mut self) {
        if let Some(p) = self.profiler() {
            unsafe {
                LiteRtResetProfiler(p);
                LiteRtStartProfiler(p);
            }
        }
    }

    /// 停止采集。
    pub fn profile_stop(&mut self) {
        if let Some(p) = self.profiler() {
            unsafe { LiteRtStopProfiler(p) };
        }
    }

    fn profiler(&mut self) -> Option<LiteRtProfiler> {
        let mut p: LiteRtProfiler = ptr::null_mut();
        if unsafe { LiteRtCompiledModelGetProfiler(self.compiled, &mut p) } == kLiteRtStatusOk
            && !p.is_null()
        {
            Some(p)
        } else {
            None
        }
    }

    /// 把一个签名的输出缓冲与另一个签名的输入缓冲对调。
    ///
    /// prefill 与 decode 在同一个模型里，prefill 算完的 KV 正是 decode 第一步
    /// 要吃的东西。不对调就得先从 GPU 拷回 CPU、再拷回 GPU，实测这一趟占了
    /// prefill 全部耗时的八成（读出 13ms，跑图只有 2.5ms）。
    pub fn swap_across(
        &mut self,
        from_sig: usize,
        out_idx: usize,
        to_sig: usize,
        in_idx: usize,
    ) -> bool {
        if from_sig == to_sig {
            return self.swap_io_buffers(from_sig, in_idx, out_idx);
        }
        let shapes_match = matches!(
            (
                self.signatures.get(from_sig).and_then(|s| s.output_shapes.get(out_idx)),
                self.signatures.get(to_sig).and_then(|s| s.input_shapes.get(in_idx)),
            ),
            (Some(o), Some(i)) if o == i
        );
        if !shapes_match {
            return false;
        }
        let (a, b) = if from_sig < to_sig {
            let (l, r) = self.signatures.split_at_mut(to_sig);
            (&mut l[from_sig], &mut r[0])
        } else {
            let (l, r) = self.signatures.split_at_mut(from_sig);
            (&mut r[0], &mut l[to_sig])
        };
        if out_idx >= a.outputs.len() || in_idx >= b.inputs.len() {
            return false;
        }
        std::mem::swap(&mut a.outputs[out_idx], &mut b.inputs[in_idx]);
        true
    }

    /// 取出某个签名的输出缓冲句柄，并换上给定的那个。
    ///
    /// 用于跨模型对接：encoder 与 decoder 是两个独立的 `.tflite`，但缓冲都来自
    /// 同一个 LiteRT 环境，句柄可以互换。调用方负责保证形状与类型一致。
    pub fn take_output_buffer(
        &mut self,
        sig: usize,
        idx: usize,
        replacement: LiteRtTensorBuffer,
    ) -> Option<LiteRtTensorBuffer> {
        let s = self.signatures.get_mut(sig)?;
        let slot = s.outputs.get_mut(idx)?;
        Some(std::mem::replace(slot, replacement))
    }

    /// 同上，换的是输入缓冲。
    pub fn take_input_buffer(
        &mut self,
        sig: usize,
        idx: usize,
        replacement: LiteRtTensorBuffer,
    ) -> Option<LiteRtTensorBuffer> {
        let s = self.signatures.get_mut(sig)?;
        let slot = s.inputs.get_mut(idx)?;
        Some(std::mem::replace(slot, replacement))
    }

    /// 某个输出缓冲的句柄（不取走）。
    pub fn output_buffer(&self, sig: usize, idx: usize) -> Option<LiteRtTensorBuffer> {
        self.signatures.get(sig)?.outputs.get(idx).copied()
    }

    /// 某个输入缓冲的句柄（不取走）。
    pub fn input_buffer(&self, sig: usize, idx: usize) -> Option<LiteRtTensorBuffer> {
        self.signatures.get(sig)?.inputs.get(idx).copied()
    }

    /// 某个输入的形状与元素类型，用于跨模型对接前的核对。
    pub fn input_spec(&self, sig: usize, idx: usize) -> Option<(&[i32], i32)> {
        let s = self.signatures.get(sig)?;
        Some((s.input_shapes.get(idx)?.as_slice(), *s.input_types.get(idx)?))
    }

    /// 某个输出的形状与元素类型。
    pub fn output_spec(&self, sig: usize, idx: usize) -> Option<(&[i32], i32)> {
        let s = self.signatures.get(sig)?;
        Some((s.output_shapes.get(idx)?.as_slice(), *s.output_types.get(idx)?))
    }

    pub fn invoke(&mut self, sig: usize) -> Result<(), String> {
        let s = self
            .signatures
            .get_mut(sig)
            .ok_or_else(|| format!("签名下标越界: {sig}"))?;
        let st = unsafe {
            LiteRtRunCompiledModel(
                self.compiled,
                sig,
                s.inputs.len(),
                s.inputs.as_mut_ptr(),
                s.outputs.len(),
                s.outputs.as_mut_ptr(),
            )
        };
        if st != kLiteRtStatusOk {
            return Err(format!("LiteRtRunCompiledModel status={st}"));
        }
        Ok(())
    }

    /// 只加锁再解锁一个输出缓冲，不拷数据，返回耗时毫秒。
    ///
    /// 用来把 GPU 队列排干。整图下沉后 `invoke` 只负责下发，某一段的真实计算
    /// 时间会一直挂到下游第一次加锁才显形；诊断时在这里插一次同步，就能把
    /// 各段的耗时各归各位。
    pub fn sync_output(&self, sig: usize, index: usize) -> f64 {
        let Some(s) = self.signatures.get(sig) else { return 0.0 };
        let Some(&buf) = s.outputs.get(index) else { return 0.0 };
        let t = std::time::Instant::now();
        let mut addr: *mut c_void = ptr::null_mut();
        if unsafe { LiteRtLockTensorBuffer(buf, &mut addr, kLiteRtTensorBufferLockModeRead) }
            == kLiteRtStatusOk
        {
            unsafe { LiteRtUnlockTensorBuffer(buf) };
        }
        t.elapsed().as_secs_f64() * 1000.0
    }

    /// 按原始字节读一个输出。
    ///
    /// 返回 (加锁毫秒数, 拷贝毫秒数)。这两段必须分开计时：GPU 缓冲的
    /// `LiteRtLockTensorBuffer` 会阻塞到产出它的 GPU 队列跑完，而 `invoke`
    /// 在整图下沉后只是下发。合在一起量的话，"等 GPU 算完"和"真的在搬字节"
    /// 会被算成同一笔，看不出还有没有搬运可省。
    pub fn get_output_bytes(&self, sig: usize, index: usize, dst: &mut [u8]) -> (f64, f64) {
        let cap = self.output_bytes(sig, index);
        let Some(s) = self.signatures.get(sig) else { return (0.0, 0.0) };
        let Some(&buf) = s.outputs.get(index) else { return (0.0, 0.0) };
        if cap == 0 {
            return (0.0, 0.0);
        }
        let mut addr: *mut c_void = ptr::null_mut();
        let t_lock = std::time::Instant::now();
        let st = unsafe { LiteRtLockTensorBuffer(buf, &mut addr, kLiteRtTensorBufferLockModeRead) };
        let lock_ms = t_lock.elapsed().as_secs_f64() * 1000.0;
        if st != kLiteRtStatusOk || addr.is_null() {
            return (lock_ms, 0.0);
        }
        let t_copy = std::time::Instant::now();
        // SAFETY: 同上，按分配时的字节数读回。
        unsafe {
            let src = std::slice::from_raw_parts(addr as *const u8, cap);
            let n = dst.len().min(cap);
            dst[..n].copy_from_slice(&src[..n]);
            LiteRtUnlockTensorBuffer(buf);
        }
        (lock_ms, t_copy.elapsed().as_secs_f64() * 1000.0)
    }
}

/// `LiteRtElementType` 的字节宽度。只列我们的模型会出现的那几种；其余按
/// 4 字节保守处理（宁可多分配，也不要越界写）。
fn element_size(ty: i32) -> usize {
    match ty {
        kLiteRtElementTypeFloat32 | kLiteRtElementTypeInt32 => 4,
        _ => 4,
    }
}

/// 需求里写明的缓冲字节数。
fn size_of_buffer(req: LiteRtTensorBufferRequirements) -> usize {
    let mut size: usize = 0;
    if unsafe { LiteRtGetTensorBufferRequirementsBufferSize(req, &mut size) } != kLiteRtStatusOk {
        return 0;
    }
    size
}

/// 把一块刚建好的缓冲清零。
fn zero_buffer(buf: LiteRtTensorBuffer, size: usize) {
    if size == 0 {
        return;
    }
    let mut addr: *mut c_void = ptr::null_mut();
    if unsafe { LiteRtLockTensorBuffer(buf, &mut addr, kLiteRtTensorBufferLockModeWrite) }
        != kLiteRtStatusOk
        || addr.is_null()
    {
        return;
    }
    // SAFETY: size 来自这块缓冲自己的需求。
    unsafe {
        std::ptr::write_bytes(addr as *mut u8, 0, size);
        LiteRtUnlockTensorBuffer(buf);
    }
}

/// 告诉 XNNPACK 用几个线程。
///
/// LiteRT 不导出 `Lrt*CpuOptions`——SDK 只给了 `litert_cpu_options.cc` 让你自己
/// 编进去，而那份源码依赖 absl。但它做的事很简单：把设置拼成一段 TOML，用
/// identifier "xnnpack" 包成 opaque options 挂到 LiteRtOptions 上。这里直接拼
/// 那段 TOML，省掉一整条 C++ 依赖链。
///
/// 不设的话由运行时自己决定，实测识别的 encoder 要 263ms；五月那版显式设 4 线程
/// 的纯 XNNPACK 基线是 150ms。大小核架构上线程数给多了也会被小核拖累，所以由
/// 调用方按平台指定，而不是让它用满所有核。
fn set_cpu_num_threads(options: LiteRtOptions, num_threads: i32) {
    if num_threads <= 0 {
        return;
    }
    set_opaque_toml(options, c"xnnpack", &format!("num_threads = {num_threads}\n"));
}

/// 把一段 TOML 作为 opaque options 挂到 `LiteRtOptions` 上。
///
/// LiteRT 的各类子选项（xnnpack / gpu_options / runtime_options_string）都是
/// 同一个形状：一个 identifier 加一段 TOML。SDK 里那几个 `Lrt*Options` 构造
/// 函数没有导出，源码又依赖 absl，直接拼 TOML 省掉整条 C++ 依赖链。
fn set_opaque_toml(options: LiteRtOptions, identifier: &std::ffi::CStr, toml: &str) {
    let Ok(payload) = std::ffi::CString::new(toml) else {
        return;
    };
    // 载荷所有权交给运行时，由这个 destructor 释放——与 SDK 里
    // MakeCStringPayload 的做法一致。
    unsafe extern "C" fn free_payload(p: *mut c_void) {
        if !p.is_null() {
            drop(unsafe { std::ffi::CString::from_raw(p as *mut c_char) });
        }
    }

    let mut opaque: LiteRtOpaqueOptions = ptr::null_mut();
    let st = unsafe {
        LiteRtCreateOpaqueOptions(
            identifier.as_ptr(),
            payload.into_raw() as *mut c_void,
            Some(free_payload),
            &mut opaque,
        )
    };
    if st != kLiteRtStatusOk {
        crate::log_line(&format!("LiteRtCreateOpaqueOptions({identifier:?}) status={st}"));
        return;
    }
    let st = unsafe { LiteRtAddOpaqueOptions(options, opaque) };
    if st != kLiteRtStatusOk {
        crate::log_line(&format!("LiteRtAddOpaqueOptions({identifier:?}) status={st}"));
    }
}

fn numel_of(shape: &[i32]) -> usize {
    shape.iter().map(|&d| d.max(0) as usize).product()
}

/// 一个签名的元信息：键，以及每个输入输出的元素类型。
struct SignatureMeta {
    key: String,
    input_types: Vec<i32>,
    output_types: Vec<i32>,
    /// 调用方序号 → 运行时下标。见 [`argument_order`]。
    input_order: Vec<usize>,
    output_order: Vec<usize>,
}

/// 按参数名里的序号排出「调用方序号 → 运行时下标」。
///
/// LiteRT 的签名参数是按名字字典序排的：`args_0, args_1, args_10, args_11,
/// ..., args_17, args_2, args_3, ...`。参数少的时候看不出来，一旦超过十个就
/// 会错位——KV cache 逐层分开传之后有 18 个输入，全错位了，识别结果变成一
/// 串与输入无关的固定 token。
///
/// 导出脚本给的名字形如 `args_<n>` / `output_<n>`，按其中的数字排即可。取不到
/// 名字或名字里没有数字时退回原顺序。
fn argument_order(names: &[String]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..names.len()).collect();
    let num = |s: &str| -> Option<u64> {
        let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    };
    if names.iter().any(|n| num(n).is_none()) {
        return idx;
    }
    idx.sort_by_key(|&i| num(&names[i]).unwrap_or(0));
    idx
}

/// 枚举模型里的签名。
///
/// 形状不在这里取——编译前拿到的可能还没定死，编译后 `LiteRtGetCompiledModel*
/// TensorLayout` 给的才是运行时实际用的。类型则不会被编译改动。
///
/// # Safety
/// `model` 必须是活的 `LiteRtModel`。
unsafe fn signature_metadata(model: LiteRtModel) -> Result<Vec<SignatureMeta>, String> {
    let mut n: usize = 0;
    let st = unsafe { LiteRtGetNumModelSignatures(model, &mut n) };
    if st != kLiteRtStatusOk {
        return Err(format!("LiteRtGetNumModelSignatures status={st}"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut sig: LiteRtSignature = ptr::null_mut();
        let st = unsafe { LiteRtGetModelSignature(model, i, &mut sig) };
        if st != kLiteRtStatusOk {
            return Err(format!("LiteRtGetModelSignature({i}) status={st}"));
        }
        let mut key: *const c_char = ptr::null();
        let st = unsafe { LiteRtGetSignatureKey(sig, &mut key) };
        if st != kLiteRtStatusOk || key.is_null() {
            return Err(format!("LiteRtGetSignatureKey({i}) status={st}"));
        }
        let key = unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned();

        let (mut n_in, mut n_out) = (0usize, 0usize);
        let st_in = unsafe { LiteRtGetNumSignatureInputs(sig, &mut n_in) };
        let st_out = unsafe { LiteRtGetNumSignatureOutputs(sig, &mut n_out) };
        if st_in != kLiteRtStatusOk || st_out != kLiteRtStatusOk {
            return Err(format!("签名 {key} 的输入/输出个数取不到: {st_in}/{st_out}"));
        }

        let input_types = unsafe { tensor_types(sig, n_in, true) }
            .map_err(|e| format!("签名 {key} 的输入类型: {e}"))?;
        let output_types = unsafe { tensor_types(sig, n_out, false) }
            .map_err(|e| format!("签名 {key} 的输出类型: {e}"))?;
        let in_names = unsafe { tensor_names(sig, n_in, true) };
        let out_names = unsafe { tensor_names(sig, n_out, false) };
        let input_order = argument_order(&in_names);
        let output_order = argument_order(&out_names);
        if n_in > 3 {
            crate::log_line(&format!(
                "[签名 {key}] 输入名={in_names:?} 顺序={input_order:?}"
            ));
        }

        out.push(SignatureMeta {
            key,
            input_types,
            output_types,
            input_order,
            output_order,
        });
    }
    if out.is_empty() {
        return Err("模型里没有签名".into());
    }
    Ok(out)
}

/// 取一个签名的输入（或输出）的元素类型。
///
/// 取一个签名的输入（或输出）参数名。取不到时给空串。
///
/// # Safety
/// `sig` 必须是活的 `LiteRtSignature`。
unsafe fn tensor_names(sig: LiteRtSignature, n: usize, inputs: bool) -> Vec<String> {
    (0..n)
        .map(|i| {
            let mut p: *const c_char = ptr::null();
            let st = unsafe {
                if inputs {
                    LiteRtGetSignatureInputName(sig, i, &mut p)
                } else {
                    LiteRtGetSignatureOutputName(sig, i, &mut p)
                }
            };
            if st == kLiteRtStatusOk && !p.is_null() {
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            } else {
                String::new()
            }
        })
        .collect()
}

/// # Safety
/// `sig` 必须是活的 `LiteRtSignature`。
unsafe fn tensor_types(sig: LiteRtSignature, n: usize, inputs: bool) -> Result<Vec<i32>, String> {
    let mut types = Vec::with_capacity(n);
    for i in 0..n {
        let mut tensor: LiteRtTensor = ptr::null_mut();
        let st = unsafe {
            if inputs {
                LiteRtGetSignatureInputTensorByIndex(sig, i, &mut tensor)
            } else {
                LiteRtGetSignatureOutputTensorByIndex(sig, i, &mut tensor)
            }
        };
        if st != kLiteRtStatusOk {
            return Err(format!("取第 {i} 个张量 status={st}"));
        }
        let mut rt = LiteRtRankedTensorType {
            element_type: kLiteRtElementTypeFloat32,
            layout: LiteRtLayout::zeroed(),
        };
        let st = unsafe { LiteRtGetRankedTensorType(tensor, &mut rt) };
        if st != kLiteRtStatusOk {
            return Err(format!("第 {i} 个张量的 RankedTensorType status={st}"));
        }
        types.push(rt.element_type);
    }
    Ok(types)
}

fn input_layouts(
    compiled: LiteRtCompiledModel,
    sig: usize,
    n: usize,
) -> Result<Vec<Vec<i32>>, String> {
    let mut shapes = Vec::with_capacity(n);
    for i in 0..n {
        let mut layout = LiteRtLayout::zeroed();
        let st = unsafe { LiteRtGetCompiledModelInputTensorLayout(compiled, sig, i, &mut layout) };
        if st != kLiteRtStatusOk {
            return Err(format!("LiteRtGetCompiledModelInputTensorLayout({sig},{i}) status={st}"));
        }
        shapes.push(layout.dims());
    }
    Ok(shapes)
}

fn output_layouts(
    compiled: LiteRtCompiledModel,
    sig: usize,
    n: usize,
) -> Result<Vec<Vec<i32>>, String> {
    // 输出形状从编译好的模型里查，不让调用方硬编码——手部检测那两个模型的
    // 输出各有四五个张量，写死一份在 Rust 里迟早跟模型对不上。
    //
    // update_allocation 只在形状里真有动态维度时才给 true，与 C++ 封装一致：
    // 它会让运行时重新分配，静态模型上是白做一遍。先按 false 问一次，发现
    // 有非正的维度（动态）再按 true 问一次。
    let mut layouts = vec![LiteRtLayout::zeroed(); n];
    let query = |layouts: &mut Vec<LiteRtLayout>, update: bool| -> LiteRtStatus {
        unsafe {
            LiteRtGetCompiledModelOutputTensorLayouts(
                compiled,
                sig,
                n,
                layouts.as_mut_ptr(),
                update,
            )
        }
    };

    // 注：这里给 true 而不是照 C++ 封装那样只在动态形状时给。识别的 decode
    // 签名（KV cache 回灌那步）在 false 下 invoke 会报 status=3，给 true 才
    // 正常——那步的输出缓冲要运行时自己重新登记一次。
    let st = query(&mut layouts, true);
    if st != kLiteRtStatusOk {
        return Err(format!("LiteRtGetCompiledModelOutputTensorLayouts({sig}) status={st}"));
    }
    Ok(layouts.iter().map(|l| l.dims()).collect())
}

impl Drop for CompiledModel {
    fn drop(&mut self) {
        unsafe {
            for s in &self.signatures {
                for &b in s.inputs.iter().chain(s.outputs.iter()) {
                    LiteRtDestroyTensorBuffer(b);
                }
            }
            LiteRtDestroyCompiledModel(self.compiled);
            LiteRtDestroyModel(self.model);
        }
        // 环境是单例，不在这里销毁。
    }
}
