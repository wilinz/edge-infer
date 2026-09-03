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

//! 各推理后端的具体实现。
//!
//! 用 feature 隔离，只编目标平台需要的那个——否则会链接到不存在的原生库。

/// LiteRT 2.x 的 CompiledModel API。多签名与 GPU 加速器都只在这条路上，
/// 经典的 TfLite C API 两样都给不了，所以那套绑定已经整个撤掉。
#[cfg(feature = "litert")]
mod litert_sys;

#[cfg(feature = "litert")]
pub mod litert;

/// 裸 Core ML：整张图直接交给系统框架，不经 ExecuTorch。
/// 只有苹果平台有；模型是 .mlpackage，只能从文件加载。
#[cfg(feature = "coreml")]
pub mod coreml;
