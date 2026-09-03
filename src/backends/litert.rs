//! LiteRT 后端。
//!
//! 走 LiteRT 2.x 的 CompiledModel API，而不是经典的 TfLite C API：
//!   1. 多签名：一个 `decoder.tflite` 里同时含 `prefill` 与 `decode`，共享
//!      权重。经典 C API 只跑子图 0，识别的解码循环没法在上面表达。
//!   2. GPU：`TfLiteGpuDelegateV2Create` 在 2.x 运行时里已经不存在，加速器
//!      改由 `LiteRtSetOptionsHardwareAccelerators` 声明。手部检测要这个。
//!
//! 模型文件约定（`model_dir` 下）：
//!   `prefix_enc.tflite` — 单签名
//!   `decoder.tflite`    — 双签名（prefill + decode）

use std::collections::HashMap;
use std::path::Path;

use super::litert_sys as sys;
use crate::{Engine, EngineConfig, EngineError, Method, Result, Tensor, TensorView};

/// 走 LiteRT 2.x CompiledModel 的引擎。
///
/// 与上面那个经典 API 的引擎并列而不是替换它：经典那套没有签名运行器绑定，
/// 也没有 GPU——`TfLiteGpuDelegateV2Create` 在 2.x 运行时里已经不存在。手部
/// 检测要 GPU，识别要多签名，两件事在这条路上一起解决。
///
/// 模型的组织与 ExecuTorch 那侧一致：识别是 `prefix_enc` 一个文件 +
/// `decoder` 一个文件（内含 prefill/decode 两个签名共享权重），手部检测是
/// 单文件单签名。
pub struct LiteRtCompiledEngine {
    models: Vec<sys::CompiledModel>,
    /// 每个方法指向 (哪个模型, 该模型里的第几个签名)。prefill 与 step 指向
    /// 同一个 `decoder.tflite` 的两个签名——它们本就共享一份权重，各编一次
    /// 是把 97MB 复制一遍。
    routes: HashMap<Method, (usize, usize)>,
    /// 已登记回灌的方法 → 若干 (输出下标, 输入下标)。KV cache 逐层分开传，
    /// 所以一个方法上会有多对。见 [`Engine::feed_output_back`]。
    feedback: HashMap<Method, Vec<(usize, usize)>>,
    /// 跨方法的缓冲对接：(来源方法, 输出下标) → (目标方法, 输入下标)。
    /// 见 [`Engine::pipe_output`]。
    pipes: Vec<(Method, usize, Method, usize)>,
    /// 已经跑过至少一次的方法。回灌要从第二次调用起才生效——第一次的输入还
    /// 得由调用方送进来（解码的第一步吃的是 prefill 算出的 KV）。
    ran: std::collections::HashSet<Method>,
    on_gpu: bool,
}

impl LiteRtCompiledEngine {
    /// 从内存字节打开单方法模型（手部检测的两个模型走这条路）。
    ///
    /// 与 [`LiteRtEngine::open_bytes`] 对齐，用 [`Method::PrefixEnc`] 作为
    /// 单方法模型的统一入口。GPU 在这里开：手部检测挪到 GPU 上，才不会和跑
    /// 在 CPU 上的识别互相排队。
    pub fn open_bytes(bytes: &[u8], cfg: &EngineConfig) -> Result<Self> {
        let want_gpu = cfg.prefer == crate::Prefer::LiteRtGpu;
        let m = sys::CompiledModel::new(bytes, want_gpu, cfg.num_threads).map_err(EngineError::Load)?;
        let on_gpu = m.on_gpu;
        let mut routes = HashMap::new();
        routes.insert(Method::PrefixEnc, (0usize, 0usize));
        Ok(Self { models: vec![m], routes, feedback: HashMap::new(), pipes: Vec::new(), ran: Default::default(), on_gpu })
    }

    /// 从目录打开识别那套模型。
    ///
    /// encoder 与 decoder 都请求 GPU 加速器（Android 上即 OpenCL），接不了的
    /// 算子由 CPU 兜底。手部检测那条流水线单独走 [`Self::open_bytes`]。
    pub fn open(cfg: &EngineConfig) -> Result<Self> {
        let dir = Path::new(&cfg.model_dir);
        let mut models = Vec::with_capacity(2);
        let mut routes = HashMap::new();

        let enc_path = dir.join("prefix_enc.tflite");
        let enc_bytes = std::fs::read(&enc_path)
            .map_err(|e| EngineError::Load(format!("{}: {e}", enc_path.display())))?;
        // 识别的三个子图现在都整图下沉到 GPU（各 1 个分区，零算子留在
        // CPU）。走到这一步是导出侧改出来的：KV cache 拆成逐层独立张量消掉
        // 了 48 个 GPU 接不了的 SLICE，encoder 的 BOOL 路径也绕掉了。
        // 具体算子数随导出的图和运行时版本变，别在注释里写死。
        models.push(
            sys::CompiledModel::from_owned(enc_bytes, true, cfg.num_threads)
                .map_err(EngineError::Load)?,
        );
        routes.insert(Method::PrefixEnc, (0usize, 0usize));

        // decoder.tflite 是双签名：prefill 与 decode 共享一份权重，只编一次。
        // 名字由导出脚本（export_tf_android_fp32.py）定，与 ExecuTorch 那侧的
        // 方法名 prefill/step 不完全一致，所以在这里对齐而不是改 Method。
        let dec_path = dir.join("decoder.tflite");
        let dec_bytes = std::fs::read(&dec_path)
            .map_err(|e| EngineError::Load(format!("{}: {e}", dec_path.display())))?;
        // prefill 与 decode 用同一个 CompiledModel：两者都在 GPU 上，共用一个
        // 模型才能让 prefill 的 KV 输出缓冲直接对接 decode 的输入缓冲，省掉
        // 一趟 GPU→CPU→GPU 的往返（实测那一趟占 prefill 耗时的八成）。
        let dec = sys::CompiledModel::from_owned(dec_bytes, true, cfg.num_threads)
            .map_err(EngineError::Load)?;

        let find = |m: &sys::CompiledModel, key: &str| {
            m.signature_index(Some(key)).ok_or_else(|| {
                EngineError::Load(format!(
                    "{} 里没有签名 {key}（有的是 {:?}）",
                    dec_path.display(),
                    m.signature_keys()
                ))
            })
        };
        routes.insert(Method::Prefill, (1usize, find(&dec, "prefill")?));
        routes.insert(Method::Step, (1usize, find(&dec, "decode")?));
        // 如实上报，别写死。这里曾经硬编码 false，于是 backend_name() 一律
        // 说自己是 litert-xnnpack，基准页把这个名字写进报告文件名，导致
        // 「识别跑在 CPU 上」的结论以讹传讹。
        let on_gpu = models[0].on_gpu || dec.on_gpu;
        models.push(dec);

        Ok(Self { models, routes, feedback: HashMap::new(), pipes: Vec::new(), ran: Default::default(), on_gpu })
    }
}

impl Engine for LiteRtCompiledEngine {
    fn backend_name(&self) -> &'static str {
        if self.on_gpu { "litert-gpu" } else { "litert-xnnpack" }
    }

    fn run(&mut self, method: Method, inputs: &[TensorView<'_>]) -> Result<Vec<Tensor>> {
        let &(mi, sig) = self
            .routes
            .get(&method)
            .ok_or(EngineError::MethodNotFound(method))?;

        // 哪些输入不用从调用方拷：
        //   - 回灌：上一次调用的输出已经在这个缓冲里（从第二次调用起）
        //   - 对接：上游方法的输出已经在这个缓冲里（上游跑过之后）
        let ran_before = self.ran.contains(&method);
        let fed: Vec<(usize, usize)> = if ran_before {
            self.feedback.get(&method).cloned().unwrap_or_default()
        } else {
            Vec::new()
        };
        let fed_after: Vec<(usize, usize)> =
            self.feedback.get(&method).cloned().unwrap_or_default();
        let mut skip: Vec<usize> = fed.iter().map(|&(_, i)| i).collect();
        for &(from, _, to, in_idx) in &self.pipes {
            if to == method && self.ran.contains(&from) {
                skip.push(in_idx);
            }
        }
        self.ran.insert(method);

        let m = &mut self.models[mi];
        let t_in = std::time::Instant::now();
        for (i, t) in inputs.iter().enumerate() {
            if skip.contains(&i) {
                continue;
            }
            m.set_input_bytes(sig, i, t.data);
        }
        let in_ms = t_in.elapsed().as_secs_f64() * 1000.0;
        let t_run = std::time::Instant::now();
        m.invoke(sig).map_err(|msg| EngineError::Execute {
            method,
            code: -1,
            msg,
        })?;
        let run_ms = t_run.elapsed().as_secs_f64() * 1000.0;
        let t_out = std::time::Instant::now();

        let n_out = m.output_count(sig);
        let mut outs = Vec::with_capacity(n_out);
        // 读出拆成两笔：加锁（等 GPU 收工）与拷贝（真的搬字节）。
        let mut lock_ms = 0.0f64;
        let mut copy_ms = 0.0f64;
        for i in 0..n_out {
            // 回灌的输出留在缓冲里给下一步用，不回传。仍要占位，保持下标对齐。
            let piped = self
                .pipes
                .iter()
                .any(|&(from, out_idx, _, _)| from == method && out_idx == i);
            if piped || fed_after.iter().any(|&(out_idx, _)| out_idx == i) {
                outs.push(Tensor::zeros(vec![0], crate::DType::F32));
                continue;
            }
            let shape: Vec<i64> = m.output_shape(sig, i).into_iter().map(|d| d as i64).collect();
            let dtype = match m.output_type(sig, i) {
                sys::kLiteRtElementTypeInt32 => crate::DType::I32,
                _ => crate::DType::F32,
            };
            let mut t = Tensor::zeros(shape, dtype);
            let (l_ms, c_ms) = m.get_output_bytes(sig, i, &mut t.data);
            lock_ms += l_ms;
            copy_ms += c_ms;
            outs.push(t);
        }
        for &(out_idx, in_idx) in &fed_after {
            m.swap_io_buffers(sig, in_idx, out_idx);
        }
        for &(from, out_idx, to, in_idx) in &self.pipes.clone() {
            if from != method {
                continue;
            }
            if let (Some(&(fmi, fsig)), Some(&(tmi, tsig))) =
                (self.routes.get(&from), self.routes.get(&to))
            {
                if fmi == tmi {
                    self.models[fmi].swap_across(fsig, out_idx, tsig, in_idx);
                } else if let (Some(ob), Some(ib)) = (
                    self.models[fmi].output_buffer(fsig, out_idx),
                    self.models[tmi].input_buffer(tsig, in_idx),
                ) {
                    // 跨模型：上游刚写好的那块直接挂到下游的输入位，下游原来
                    // 那块回填给上游做下一轮的输出，两边都不额外分配。
                    self.models[fmi].take_output_buffer(fsig, out_idx, ib);
                    self.models[tmi].take_input_buffer(tsig, in_idx, ob);
                }
            }
        }

        // 诊断开关：enc / prefill 的输出都走对接不回读，它们的真实 GPU 时间会
        // 一直挂到 step 第一步写输入时那把锁才显形（实测 12.7ms）。打开这个
        // 开关后每段跑完各排一次队，耗时就能各归各位。会打断流水，只在量数
        // 的时候开。
        const FORCE_SYNC: bool = false;
        let mut sync_ms = 0.0f64;
        if FORCE_SYNC && method != Method::Step && n_out > 0 {
            // 重新按下标取，别用上面那个 `m`：它的可变借用会被拉长到
            // 缓冲对接那几行之后，跟那里对 self.models 的借用打架。
            sync_ms = self.models[mi].sync_output(sig, 0);
        }

        // Step 每步都打会把日志淹掉（一次识别十几行，500 样本上万行），平时
        // 关掉。要定位单步耗时构成时把这个常量翻成 true。
        const LOG_STEP: bool = false;
        if method != Method::Step || LOG_STEP {
            crate::log_line(&format!(
                "[{method:?}] 写入={in_ms:.1}ms 跑图={run_ms:.1}ms 排队={sync_ms:.1}ms \
                 读出={:.1}ms(加锁={lock_ms:.1} 拷贝={copy_ms:.2}) 输出={n_out}个",
                t_out.elapsed().as_secs_f64() * 1000.0
            ));
        }
        Ok(outs)
    }

    fn output_shape(&mut self, method: Method, index: usize) -> Option<Vec<i64>> {
        let &(mi, sig) = self.routes.get(&method)?;
        Some(
            self.models[mi]
                .output_shape(sig, index)
                .into_iter()
                .map(|d| d as i64)
                .collect(),
        )
    }

    fn has_method(&self, method: Method) -> bool {
        self.routes.contains_key(&method)
    }

    fn pipe_output(
        &mut self,
        from: Method,
        out_index: usize,
        to: Method,
        in_index: usize,
    ) -> bool {
        let (Some(&(fmi, fsig)), Some(&(tmi, tsig))) =
            (self.routes.get(&from), self.routes.get(&to))
        else {
            return false;
        };
        if fmi == tmi {
            if !self.models[fmi].swap_across(fsig, out_index, tsig, in_index) {
                return false;
            }
            // 试探用的那次要还原，正式对调发生在 from 跑完之后。
            self.models[fmi].swap_across(fsig, out_index, tsig, in_index);
        } else {
            // 跨模型：缓冲句柄都来自同一个 LiteRT 环境，可以互换，但要自己
            // 核对形状与元素类型。
            let ok = match (
                self.models[fmi].output_spec(fsig, out_index),
                self.models[tmi].input_spec(tsig, in_index),
            ) {
                (Some((os, ot)), Some((is, it))) => os == is && ot == it,
                _ => false,
            };
            if !ok {
                return false;
            }
        }
        let entry = (from, out_index, to, in_index);
        if !self.pipes.contains(&entry) {
            self.pipes.push(entry);
        }
        self.ran.remove(&from);
        self.ran.remove(&to);
        true
    }

    fn feed_output_back(&mut self, method: Method, out_index: usize, in_index: usize) -> bool {
        let Some(&(mi, sig)) = self.routes.get(&method) else { return false };
        // 先试一次对调，形状对不上就不登记。
        if !self.models[mi].swap_io_buffers(sig, in_index, out_index) {
            return false;
        }
        // 试探用的那次对调要还原，正式的对调发生在每次 run 之后。
        self.models[mi].swap_io_buffers(sig, in_index, out_index);
        let pairs = self.feedback.entry(method).or_default();
        if !pairs.contains(&(out_index, in_index)) {
            pairs.push((out_index, in_index));
        }
        // 每次识别都会重新登记一遍。清掉「已跑过」的标记，让新一轮的第一步
        // 仍由调用方写入——那一步吃的是这次 prefill 算出的 KV。
        self.ran.remove(&method);
        true
    }
}

