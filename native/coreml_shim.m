// Core ML 的 C 接口，供 Rust 侧的 coreml 后端调用。
//
// 与 et_shim.cpp 平级：那边把 ExecuTorch 的 C++ API 收成 C，这边把 Core ML
// 的 Objective-C API 收成 C。Rust 只认这一层，不引 objc2 之类的绑定——
// 21 个关键点的模型每帧一次调用，桥接开销不是瓶颈，可读性更值钱。
//
// 两点值得记：
//
// 1. Core ML 只能从**文件**加载，没有从内存字节加载的入口。所以手部检测
//    那条原本 open_from_bytes 的路，在这个后端上必须改成传路径。
//
// 2. .mlpackage 是源格式，运行前要编译成 .mlmodelc。编译耗时可观（首次
//    几百毫秒到几秒），所以编完缓存到 caches 目录，按源文件的修改时间判断
//    是否复用——app 升级换了模型，时间戳变，自然重编。

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <stdint.h>
#include <string.h>

/// 与 Rust 侧 CmlTensor 逐字段对齐。dtype 目前只用到 f32（0）。
typedef struct {
  void *data;
  const int64_t *shape;
  size_t ndim;
  int32_t dtype;
} CmlTensor;

enum {
  CML_OK = 0,
  CML_ERR_LOAD = 1,
  CML_ERR_COMPILE = 2,
  CML_ERR_INPUT = 3,
  CML_ERR_RUN = 4,
  CML_ERR_OUTPUT = 5,
};

typedef struct {
  void *model;             // MLModel*，__bridge_retained
  char **input_names;      // 按模型声明顺序
  size_t n_inputs;
  char **output_names;
  size_t n_outputs;
} CmlModel;

static char *dup_cstr(NSString *s) {
  const char *u = [s UTF8String];
  size_t n = strlen(u) + 1;
  char *out = (char *)malloc(n);
  memcpy(out, u, n);
  return out;
}

/// .mlpackage → .mlmodelc，带缓存。已经是 .mlmodelc 的直接用。
static NSURL *compiled_url(NSURL *src, int *err) {
  if ([[src pathExtension] isEqualToString:@"mlmodelc"]) {
    return src;
  }

  NSFileManager *fm = [NSFileManager defaultManager];
  NSDictionary *attrs = [fm attributesOfItemAtPath:[src path] error:nil];
  NSDate *mtime = attrs[NSFileModificationDate];
  // 缓存名带上源文件名与修改时间：模型换了就自然落到另一个名字上，
  // 不需要额外的失效逻辑。
  NSString *tag = [NSString stringWithFormat:@"%@-%.0f.mlmodelc",
                                             [[src lastPathComponent] stringByDeletingPathExtension],
                                             [mtime timeIntervalSince1970]];
  NSURL *caches = [[fm URLsForDirectory:NSCachesDirectory inDomains:NSUserDomainMask] firstObject];
  NSURL *cached = [caches URLByAppendingPathComponent:tag];
  if ([fm fileExistsAtPath:[cached path]]) {
    return cached;
  }

  NSError *e = nil;
  NSURL *tmp = [MLModel compileModelAtURL:src error:&e];
  if (!tmp || e) {
    NSLog(@"[mwh/coreml] 编译失败: %@", e);
    if (err) *err = CML_ERR_COMPILE;
    return nil;
  }
  // 编译产物落在临时目录，进程退出后会被清掉；搬到 caches 才谈得上复用。
  [fm removeItemAtURL:cached error:nil];
  if (![fm moveItemAtURL:tmp toURL:cached error:&e]) {
    NSLog(@"[mwh/coreml] 缓存编译产物失败（本次直接用临时目录）: %@", e);
    return tmp;
  }
  return cached;
}

/// [function_name] 选多函数模型里的哪个函数；单函数模型传 NULL。
///
/// 识别的 decoder 把 prefill 与 step 合并成了一个多函数模型（两者是同一份
/// decoder 权重的两种用法，拆开存要多 43.7MB）。加载时各按函数名开一个
/// MLModel，磁盘上共用同一份权重。多函数模型要 iOS 18 / macOS 15 起。
void *cml_model_load(const char *path, const char *function_name,
                     int32_t num_threads, int *err) {
  (void)num_threads;  // Core ML 自己调度 ANE/GPU/CPU，没有线程数可设
  @autoreleasepool {
    if (err) *err = CML_OK;
    NSURL *src = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    NSURL *url = compiled_url(src, err);
    if (!url) return NULL;

    MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsAll;
    if (function_name) {
      // 用 respondsToSelector 而不是 @available：后者会生成对 clang
      // compiler-rt 里 ___isPlatformVersionAtLeast 的调用，而这份 shim 最终是
      // 被 rustc 链进 cdylib 的，那条链接行里没有 libclang_rt，直接报
      // "symbol(s) not found"。functionName 是 iOS 18 / macOS 15 才有的属性。
      if ([cfg respondsToSelector:@selector(setFunctionName:)]) {
        [cfg setValue:[NSString stringWithUTF8String:function_name] forKey:@"functionName"];
      } else {
        NSLog(@"[mwh/coreml] 多函数模型需要 iOS 18 / macOS 15 起");
        if (err) *err = CML_ERR_LOAD;
        return NULL;
      }
    }

    NSError *e = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&e];
    if (!model || e) {
      NSLog(@"[mwh/coreml] 加载失败: %@", e);
      if (err) *err = CML_ERR_LOAD;
      return NULL;
    }

    CmlModel *m = (CmlModel *)calloc(1, sizeof(CmlModel));
    m->model = (__bridge_retained void *)model;

    MLModelDescription *d = model.modelDescription;
    NSArray<NSString *> *ins = [d.inputDescriptionsByName allKeys];
    NSArray<NSString *> *outs = [d.outputDescriptionsByName allKeys];
    // 字典无序，而调用方按位置传参/取值。
    //
    // 输出按导出时钉的 out0/out1… 排序，正好与原 TFLite 的输出顺序一致。
    //
    // 输入这边排序只是兜底：单输入模型（手部检测）无歧义，多输入的必须由
    // 调用方在 cml_model_run 里显式给名字——按字典序排会把
    // (token_id, pos, past_kv) 排成 (past_kv, pos, token_id)，喂反了。
    ins = [ins sortedArrayUsingSelector:@selector(compare:)];
    outs = [outs sortedArrayUsingSelector:@selector(localizedStandardCompare:)];

    m->n_inputs = ins.count;
    m->input_names = (char **)calloc(ins.count, sizeof(char *));
    for (NSUInteger i = 0; i < ins.count; i++) m->input_names[i] = dup_cstr(ins[i]);

    m->n_outputs = outs.count;
    m->output_names = (char **)calloc(outs.count, sizeof(char *));
    for (NSUInteger i = 0; i < outs.count; i++) m->output_names[i] = dup_cstr(outs[i]);

    return m;
  }
}

size_t cml_model_output_count(void *handle) {
  CmlModel *m = (CmlModel *)handle;
  return m ? m->n_outputs : 0;
}

void cml_model_free(void *handle) {
  CmlModel *m = (CmlModel *)handle;
  if (!m) return;
  @autoreleasepool {
    MLModel *model = (__bridge_transfer MLModel *)m->model;
    (void)model;
  }
  for (size_t i = 0; i < m->n_inputs; i++) free(m->input_names[i]);
  for (size_t i = 0; i < m->n_outputs; i++) free(m->output_names[i]);
  free(m->input_names);
  free(m->output_names);
  free(m);
}

/// 跑一次。输入按位置对应模型声明的输入名；输出按 out0/out1… 顺序回填。
///
/// 输出的 data / shape 由这里 malloc，调用方用完必须交给 cml_tensors_free。
/// [input_names] 给出每个输入对应的模型特征名，按位置一一对应；传 NULL 时
/// 退回模型自己那份（字典序）——只对单输入模型安全。
int cml_model_run(void *handle, const CmlTensor *inputs, size_t n_inputs,
                  const char *const *input_names,
                  CmlTensor **out_tensors, size_t *n_out, int *err) {
  @autoreleasepool {
    if (err) *err = CML_OK;
    CmlModel *m = (CmlModel *)handle;
    if (!m || n_inputs != m->n_inputs) {
      if (err) *err = CML_ERR_INPUT;
      return -1;
    }
    MLModel *model = (__bridge MLModel *)m->model;

    NSMutableDictionary *feed = [NSMutableDictionary dictionaryWithCapacity:n_inputs];
    for (size_t i = 0; i < n_inputs; i++) {
      const CmlTensor *t = &inputs[i];
      NSMutableArray *shape = [NSMutableArray arrayWithCapacity:t->ndim];
      NSMutableArray *strides = [NSMutableArray arrayWithCapacity:t->ndim];
      // 行优先的 stride：末维为 1，往前逐维乘。MLMultiArray 要求显式给出。
      int64_t acc = 1;
      for (size_t d = 0; d < t->ndim; d++) [shape addObject:@(t->shape[d])];
      for (size_t d = 0; d < t->ndim; d++) [strides addObject:@0];
      for (ssize_t d = (ssize_t)t->ndim - 1; d >= 0; d--) {
        strides[d] = @(acc);
        acc *= t->shape[d];
      }

      NSError *e = nil;
      // 不拷贝：直接借调用方那块内存，deallocator 留空（生命周期由 Rust 侧
      // 保证覆盖这次调用）。每帧 192×192×3 的图省一次拷贝。
      MLMultiArray *arr = [[MLMultiArray alloc] initWithDataPointer:t->data
                                                              shape:shape
                                                           dataType:MLMultiArrayDataTypeFloat32
                                                            strides:strides
                                                        deallocator:nil
                                                              error:&e];
      if (!arr || e) {
        NSLog(@"[mwh/coreml] 构造输入失败: %@", e);
        if (err) *err = CML_ERR_INPUT;
        return -1;
      }
      const char *nm = input_names ? input_names[i] : m->input_names[i];
      feed[[NSString stringWithUTF8String:nm]] =
          [MLFeatureValue featureValueWithMultiArray:arr];
    }

    NSError *e = nil;
    MLDictionaryFeatureProvider *provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed error:&e];
    if (!provider || e) {
      NSLog(@"[mwh/coreml] 组装输入失败: %@", e);
      if (err) *err = CML_ERR_INPUT;
      return -1;
    }

    id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&e];
    if (!result || e) {
      NSLog(@"[mwh/coreml] 推理失败: %@", e);
      if (err) *err = CML_ERR_RUN;
      return -1;
    }

    CmlTensor *outs = (CmlTensor *)calloc(m->n_outputs, sizeof(CmlTensor));
    for (size_t i = 0; i < m->n_outputs; i++) {
      NSString *name = [NSString stringWithUTF8String:m->output_names[i]];
      MLFeatureValue *v = [result featureValueForName:name];
      MLMultiArray *arr = v.multiArrayValue;
      if (!arr) {
        NSLog(@"[mwh/coreml] 输出 %@ 不是 multiarray", name);
        free(outs);
        if (err) *err = CML_ERR_OUTPUT;
        return -1;
      }
      size_t ndim = arr.shape.count;
      int64_t *shape = (int64_t *)malloc(sizeof(int64_t) * (ndim ? ndim : 1));
      size_t numel = 1;
      for (size_t d = 0; d < ndim; d++) {
        shape[d] = [arr.shape[d] longLongValue];
        numel *= (size_t)shape[d];
      }
      float *data = (float *)malloc(sizeof(float) * (numel ? numel : 1));
      // getBytesWithHandler 给的是模型内部缓冲，出了 block 就不保证有效，
      // 所以这里拷一份交给调用方。fp16 输出由 Core ML 自己转成 fp32。
      // MLMultiArray **不保证密集排布**：Core ML 会按对齐要求给某些维度补
      // padding，这时 strides 不等于「后续维度之积」。手掌检测的输出末维是
      // 18 与 1，正好落在会被补齐的那一类；识别那几个末维 512/230 本来对齐，
      // 所以按密集拷也没露馅——照密集假设写会读到错位的数据，检测直接归零。
      //
      // 所以先判断是否密集：密集走直通循环（arm64 上 __fp16 转换是单条指令，
      // 编译器会向量化）；不密集才按 strides 逐元素索引。两条都不用
      // objectAtIndexedSubscript——那条每个元素造一个 NSNumber，decoder 的
      // new_kv 有 68 万个元素，一次调用就是 80ms 级别的开销。
      NSArray<NSNumber *> *strideArr = arr.strides;
      // 堆上分配而不是栈数组：ObjC 的 block 不能捕获 C 数组类型。
      int64_t *strides = (int64_t *)calloc(ndim ? ndim : 1, sizeof(int64_t));
      int dense = 1;
      {
        int64_t acc = 1;
        for (ssize_t d = (ssize_t)ndim - 1; d >= 0; d--) {
          strides[d] = [strideArr[d] longLongValue];
          if (strides[d] != acc) dense = 0;
          acc *= shape[d];
        }
      }
      const MLMultiArrayDataType dtype = arr.dataType;
      [arr getBytesWithHandler:^(const void *bytes, NSInteger size) {
        (void)size;
        if (dense) {
          if (dtype == MLMultiArrayDataTypeFloat32) {
            memcpy(data, bytes, sizeof(float) * numel);
          } else if (dtype == MLMultiArrayDataTypeFloat16) {
            const __fp16 *src = (const __fp16 *)bytes;
            for (size_t k = 0; k < numel; k++) data[k] = (float)src[k];
          } else {
            const double *src = (const double *)bytes;
            for (size_t k = 0; k < numel; k++) data[k] = (float)src[k];
          }
          return;
        }
        // 非密集：把线性下标拆成各维坐标，再按 strides 求偏移。
        int64_t *coord = (int64_t *)calloc(ndim ? ndim : 1, sizeof(int64_t));
        for (size_t k = 0; k < numel; k++) {
          int64_t off = 0;
          for (size_t d = 0; d < ndim; d++) off += coord[d] * strides[d];
          if (dtype == MLMultiArrayDataTypeFloat32) {
            data[k] = ((const float *)bytes)[off];
          } else if (dtype == MLMultiArrayDataTypeFloat16) {
            data[k] = (float)((const __fp16 *)bytes)[off];
          } else {
            data[k] = (float)((const double *)bytes)[off];
          }
          for (ssize_t d = (ssize_t)ndim - 1; d >= 0; d--) {
            if (++coord[d] < shape[d]) break;
            coord[d] = 0;
          }
        }
        free(coord);
      }];
      free(strides);
      outs[i].data = data;
      outs[i].shape = shape;
      outs[i].ndim = ndim;
      outs[i].dtype = 0;
    }
    *out_tensors = outs;
    *n_out = m->n_outputs;
    return 0;
  }
}

void cml_tensors_free(CmlTensor *tensors, size_t n) {
  if (!tensors) return;
  for (size_t i = 0; i < n; i++) {
    free(tensors[i].data);
    free((void *)tensors[i].shape);
  }
  free(tensors);
}
