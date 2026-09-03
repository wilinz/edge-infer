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

//! 按 feature 编各后端要的原生代码并链接。
//!
//! Android 走 LiteRT（动态库，运行时构件由 mwh 的 build hook 取）；
//! iOS/macOS 走裸 Core ML，只编一个 ObjC shim，CoreML.framework 是系统框架，
//! 不需要任何第三方构件。

fn main() {
    println!("cargo:rerun-if-changed=native/coreml_shim.m");
    println!("cargo:rerun-if-env-changed=TFLITE_LIB_DIR");

    #[cfg(feature = "litert")]
    link_litert();

    #[cfg(feature = "coreml")]
    build_coreml_shim();
}

/// Core ML 的 shim。没有任何第三方依赖——CoreML.framework 是系统框架，
/// 所以这个后端不需要 third_party 那套运行时构件。
#[cfg(feature = "coreml")]
fn build_coreml_shim() {
    println!("cargo:rerun-if-changed=native/coreml_shim.m");
    cc::Build::new()
        .file("native/coreml_shim.m")
        // ARC：shim 里用 __bridge_retained / __bridge_transfer 显式交接所有权。
        .flag("-fobjc-arc")
        .warnings(false)
        .compile("coreml_shim");
    for fw in ["CoreML", "Foundation"] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
}

#[cfg(feature = "litert")]
fn link_litert() {
    if let Ok(dir) = std::env::var("TFLITE_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    }
    // 链接的库名写在 tflite_sys.rs 的 #[link] 属性上（按平台分），这里只给
    // 搜索路径。两处都写会让链接器收到重复的 -l，且容易改漏一处。
}
