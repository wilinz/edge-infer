//! 裸 Core ML 后端的冒烟测试：加载 .mlpackage，跑一次，打印各输出的形状与
//! 校验和。
//!
//! 校验和用来跟 Python 侧（coremltools 直接跑同一个 .mlpackage、以及原版
//! TFLite）对齐——三者一致才说明 shim 的输入摆放、输出顺序与拷贝都对。
//! 输入用固定伪随机序列，Python 那边照同一个公式生成。
//!
//! 用法：
//!   cargo run -p edge-infer --features coreml --example coreml_smoke -- \
//!     <模型.mlpackage> <N> <H> <W> <C>

#[cfg(feature = "coreml")]
fn main() {
    use edge_infer::backends::coreml::CoreMlEngine;
    use edge_infer::{Engine, EngineConfig, Method, TensorView};

    let mut args = std::env::args().skip(1);
    let path = args.next().expect("需要传 .mlpackage 路径");
    let dims: Vec<i64> = args.map(|s| s.parse().expect("维度要是整数")).collect();
    let dims = if dims.is_empty() { vec![1, 192, 192, 3] } else { dims };
    let numel: usize = dims.iter().product::<i64>() as usize;

    // 固定伪随机：x[i] = ((i * 1103515245 + 12345) % 1000) / 1000
    // 与 Python 侧同一个公式，免得为了对数值再传一个几 MB 的输入文件。
    let input: Vec<f32> = (0..numel)
        .map(|i| ((i as u64).wrapping_mul(1103515245).wrapping_add(12345) % 1000) as f32 / 1000.0)
        .collect();

    let cfg = EngineConfig { num_threads: 0, ..Default::default() };
    let mut eng = CoreMlEngine::open_path(&path, &cfg).expect("加载模型");
    println!("后端: {}", eng.backend_name());

    let t0 = std::time::Instant::now();
    let out = eng
        .run(Method::PrefixEnc, &[TensorView::f32(&dims, &input)])
        .expect("推理失败");
    println!("首次（含编译/预热）: {:?}", t0.elapsed());

    for (i, t) in out.iter().enumerate() {
        let v = t.as_f32().expect("输出应为 f32");
        let sum: f64 = v.iter().map(|&x| x as f64).sum();
        let absmax = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
        println!(
            "  out{i}: shape={:?} numel={} sum={:.6} absmax={:.6}",
            t.shape,
            v.len(),
            sum,
            absmax
        );
    }

    // 稳态延迟：首次含编译与图规划，单独测后面几次。
    let mut ms = Vec::new();
    for _ in 0..20 {
        let t = std::time::Instant::now();
        let _ = eng.run(Method::PrefixEnc, &[TensorView::f32(&dims, &input)]).unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("稳态: 中位 {:.2}ms  最快 {:.2}ms", ms[ms.len() / 2], ms[0]);
}

#[cfg(not(feature = "coreml"))]
fn main() {
    eprintln!("需要 --features coreml");
}
