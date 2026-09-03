# edge-infer

端侧推理抽象层。上层只依赖 `Engine` trait，不关心底下是 LiteRT 还是 Core ML。

## 结构

```
src/lib.rs          Engine trait、EngineConfig、错误类型、按路径/字节打开模型
src/tensor.rs       后端无关的张量：输入借用（TensorView），输出持有（Tensor）
src/backends/
  litert.rs         LiteRT 2.x 的 CompiledModel API（Android）
  litert_sys.rs     它的 C 绑定
  coreml.rs         裸 Core ML（iOS / macOS）
native/
  coreml_shim.m     Core ML 的 Objective-C API 收成 C
```

## 后端

编译期 feature 决定，避免链接目标平台没有的原生库：

```bash
cargo build --release --features litert     # Android
cargo build --release --features coreml     # iOS / macOS
```

| | Android | iOS / macOS |
|---|---|---|
| 后端 | LiteRT 2.x | 裸 Core ML |
| 模型 | `.tflite`（可多签名） | `.mlpackage` → 构建期编成 `.mlmodelc` |
| 外部构件 | `libLiteRt.so` + GPU 加速器 | 无，`CoreML.framework` 是系统框架 |

`Engine` trait 把两者抹平：方法统一为 `prefix_enc` / `prefill` / `step`，
单签名（每方法一文件）与多签名（一文件多方法）都由后端内部消化。手部检测那类
单方法模型统一用 `Method::PrefixEnc` 作入口。

### Android 的运行时构件

```bash
scripts/fetch_litert.sh android-arm64
```

两个 `.so` 取自 Google 发的官方 AAR（`com.google.ai.edge.litert:litert`）——
自行用 CMake 编的 `libtensorflowlite_c.so` GPU delegate 初始化会失败却留在执行
计划里，每次推理 168~1736 ms，官方这份稳定在 31~38 ms。产物落在
`third_party/build/<target>/install/lib`，不进仓库。

Apple 侧不需要这一步。

## Core ML 后端的几个坑

* **只能从文件加载**。没有从内存字节建模型的入口，所以 `open_path` 与
  `open_from_bytes` 并列；`.mlpackage` 还是个目录，字节那条路根本走不通。
* **`.mlpackage` 是源格式**，运行前要编译成 `.mlmodelc`。交给设备做的话首次启动
  要付几秒，所以构建期用 `xcrun coremlcompiler` 编好；shim 里保留了运行时编译
  加缓存的退路。
* **输入按名字喂**，而 `Engine` 的接口按位置给张量。模型描述那边是个字典，取出来
  无序，按字典序排会把 `(token_id, pos, past_kv)` 排成 `(past_kv, pos, token_id)`，
  所以映射由调用方显式钉住。
* **`MLMultiArray` 不保证密集排布**：Core ML 会按对齐给某些维度补 padding，这时
  `strides` 不等于「后续维度之积」。末维是 18 或 1 的输出正落在会被补齐的那一类，
  按密集假设读会全错。
* **fp16 输出**别走 `objectAtIndexedSubscript`——每个元素造一个 NSNumber，68 万
  元素的张量一次调用就是 80 ms 级别的开销。用 `__fp16` 逐元素赋值，arm64 上是
  单条指令，编译器会自动向量化。
* ObjC 里**不要用 `@available`**：它会生成对 clang compiler-rt 里
  `___isPlatformVersionAtLeast` 的调用，而这份 shim 最终由 rustc 链进 cdylib，
  那条链接行没有 `libclang_rt`。改用 `respondsToSelector:`。

## 依赖方向

```
hand-track ──┐
             ├──> edge-infer
识别核心   ──┘
```

单向：它不依赖任何上层。

## 上游

| 仓 | 关系 |
|---|---|
| [hand-track](https://github.com/wilinz/hand-track) | 手部检测流水线 |
| [air_calculator-rs](https://github.com/wilinz/air_calculator-rs) | 手写识别核心与 C ABI |

path 依赖指向兄弟目录，几个仓需要并排 checkout。

## License

Apache License 2.0 — see `LICENSE` and `NOTICE`.

The code is free to use, modify, redistribute and commercialize, including
publishing derivative applications on the App Store, Google Play or anywhere
else. Per section 6 of the Apache License 2.0, no trademark or product name
rights are granted: **Air Calculator**, **AirCalculator**, `air_calculator` as
a product name, and the application icons and logos in this repository are
reserved, and may not be used to publish or promote a derivative work without
prior written permission. Fork it, but ship it under your own name. Factual
references such as "based on Air Calculator" are fine, as long as they do not
suggest endorsement.
