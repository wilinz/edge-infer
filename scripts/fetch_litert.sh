#!/usr/bin/env bash
# Copyright 2026 wilinz.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 取 Android 端的 LiteRT 运行时，放进 mwh 的 build hook 认的 install 目录。
#
# 这两个 .so 是 Google 发的官方二进制，不是我们编的——自己用 CMake 编的那份
# libtensorflowlite_c.so 每次推理要 168~1736ms（GPU delegate 初始化失败却留在
# 执行计划里），官方这份稳定在 31~38ms。
#
#   libLiteRt.so                  运行时，兼容经典 TFLite C API
#   libLiteRtClGlAccelerator.so   GPU 加速器，OpenCL 与 OpenGL 两个后端
#
# 加速器由运行时按名字 dlopen，不参与链接，但要跟着进包。
#
# 只有 Android 需要。iOS/macOS 走裸 Core ML——CoreML.framework 是系统框架，
# 没有构件可取（这个脚本的前身 build_executorch.sh 就是给那条路编运行时的，
# ExecuTorch 整体移除后只剩这里这一段还有用）。
#
# 用法：
#   scripts/fetch_litert.sh android-arm64
#   scripts/fetch_litert.sh android-armv7

set -euo pipefail

TARGET="${1:-}"
case "$TARGET" in
  android-arm64) LITERT_ABI=arm64-v8a ;;
  android-armv7) LITERT_ABI=armeabi-v7a ;;
  *)
    echo "用法: $0 {android-arm64|android-armv7}" >&2
    exit 2
    ;;
esac

LITERT_VER=2.2.0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/third_party/build/$TARGET"
LIB="$OUT/install/lib"
mkdir -p "$LIB"

if [ -f "$LIB/libLiteRt.so" ]; then
  echo "已存在：$LIB/libLiteRt.so"
  exit 0
fi

AAR="$OUT/litert-$LITERT_VER.aar"
echo "拉取 LiteRT $LITERT_VER 运行时（${LITERT_ABI}）..."
curl -fsSL -o "$AAR" \
  "https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/$LITERT_VER/litert-$LITERT_VER.aar"
unzip -o -j "$AAR" "jni/$LITERT_ABI/libLiteRt.so" \
  "jni/$LITERT_ABI/libLiteRtClGlAccelerator.so" -d "$LIB" >/dev/null
rm -f "$AAR"

echo "完成：$LIB"
ls -lh "$LIB"/libLiteRt*.so
