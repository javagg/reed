# Reed 与 libCEED 概念对照

这份文档用于维护 Reed 与 libCEED 之间的概念、数据类型和接口映射，作为后续设计、实现和示例迁移时的参考。

## 1. 总体定位

- libCEED：面向高阶有限元离散化的可移植算子库，核心强调 `Ceed` 上下文、后端资源、向量、基函数、自由度限制、QFunction 与 Operator 的组合。
- Reed：用 Rust 复现类似抽象层次，核心强调 trait 抽象、后端可替换、类型安全和面向算子组合的接口设计。

当前可以认为：

- libCEED 的 C API 对象模型，对应 Reed 的 Rust trait + struct 对象模型。
- libCEED 的后端资源字符串，对应 Reed 的 `Reed::init(resource)` 与 `Backend` 实现。

### 1.1 资源字符串语义（约定）

- 资源路径表示 **执行位置与能力**：主机 CPU（`/cpu/…`）、GPU（`/gpu/…`）、将来可扩展为 ISA 档次（例如 `/cpu/x86/avx512`，具体以实现为准）。
- Reed **不包含** Hypre、PETSc、MUMPS、MKL 等代数求解器的资源路由；线性/非线性求解在更上层选择。
- `/gpu/cuda`、`/gpu/hip` 为保留占位符（可解析并出现在选型报告中）；具体执行后端未实现前，`init` 会返回错误。

## 2. 顶层对象映射

| libCEED | Reed | 说明 |
|---|---|---|
| `Ceed` | `Reed<T>` | 顶层上下文，负责创建后端对象 |
| `CeedInit()` | `Reed::init()` | 从资源字符串初始化上下文 |
| `Ceed` backend/resource | `Backend<T>` trait + `CpuBackend<T>` | 后端抽象与具体实现 |
| `CeedVector` | `dyn VectorTrait<T>` | 抽象数值向量 |
| `CeedElemRestriction` | `dyn ElemRestrictionTrait<T>` | 单元到全局自由度映射 |
| `CeedBasis` | `dyn BasisTrait<T>` | 参考单元基函数与求值 |
| `CeedQFunction` | `dyn QFunctionTrait<T>` | 积分点点态算符 |
| `CeedOperator` | `dyn OperatorTrait<T>` / `CpuOperator<'a, T>` | 完整离散算符 |
| `CeedCompositeOperator` | `CompositeOperator<T>` | 加法型组合；`apply` / `apply_add` / `linear_assemble_diagonal` 为子算子之和 |

## 3. 数据类型映射

### 3.1 标量与整数

| libCEED | Reed | 说明 |
|---|---|---|
| `CeedScalar` | `T: Scalar` | Reed 当前用泛型 trait 支持 `f32/f64` |
| `CeedInt` | `usize` / `i32` | 尺寸多用 `usize`，restriction 偏移和 stride 当前使用 `i32` |
| `CEED_EPSILON` | `T::EPSILON` | 通过 `Scalar` trait 提供 |

说明：

- libCEED 中整数类型更统一，Reed 当前在“尺寸”和“索引偏移”之间分成 `usize` 与 `i32` 两套表示。
- 后续如果要减少示例迁移成本，建议明确一套统一约定，例如：尺寸统一 `usize`，offset/stride 统一 `i32`。

### 3.2 枚举与模式

| libCEED | Reed | 说明 |
|---|---|---|
| `CeedMemType` | `MemType` | 主机/设备内存位置 |
| `CeedEvalMode` | `EvalMode` | `None/Interp/Grad/Div/Curl/Weight` |
| `CeedQuadMode` | `QuadMode` | `Gauss/GaussLobatto` |
| `CeedTransposeMode` | `TransposeMode` | 转置与否 |
| `CeedNormType` | `NormType` | 向量范数类型 |
| `CeedElemTopology` | `ElemTopology` | 单元拓扑 |

当前对应关系基本完整，语义上与 libCEED 保持一致。

## 4. 核心抽象映射

### 4.1 Vector

libCEED 概念：

- `CeedVectorCreate`
- `CeedVectorSetValue`
- `CeedVectorGetArrayRead/Write`
- `CeedVectorNorm`
- `CeedVectorAXPY`

Reed 对应：

- `Reed::vector()`
- `VectorTrait::set_value()`
- `VectorTrait::as_slice()` / `as_mut_slice()`
- `VectorTrait::norm()`
- `VectorTrait::axpy()`
- `VectorTrait::scale()`
- `Reed::vector_from_slice()`
- `VectorTrait::copy_from_slice()` / `copy_to_slice()`

说明：

- libCEED 的数组访问接口更接近显式内存管理。
- Reed 当前偏向 Rust 风格的 slice 访问。

### 4.2 ElemRestriction

libCEED 概念：

- `CeedElemRestrictionCreate`
- `CeedElemRestrictionCreateStrided`
- `CeedElemRestrictionApply`

Reed 对应：

- `Reed::elem_restriction()`
- `Reed::strided_elem_restriction()`
- `Reed::elem_restriction_at_points()`（语义同 offset restriction，`elemsize = npoints_per_elem`，对齐 libCEED `CeedElemRestrictionCreateAtPoints` 命名）
- `ElemRestrictionTrait::apply()`

说明：

- 数学语义一致：
  - `E`: global -> local
  - `E^T`: local -> global accumulate
- Reed 当前直接在 trait 中暴露 `TransposeMode`，对应 libCEED 的 restriction apply 语义。
- **WGPU**（`f32`）：offset 型已实现 **gather** 与 **transpose scatter**（与 CPU 一致）；**strided** 型仍走 CPU。`elem_restriction_at_points` 与 offset 共用实现，GPU 路径同上。

### 4.3 Basis

libCEED 概念：

- `CeedBasisCreateTensorH1Lagrange`
- `CeedBasisApply`
- `CeedBasisGetQWeights`
- `CeedBasisGetQRef`

Reed 对应：

- `Reed::basis_tensor_h1_lagrange()`
- `BasisTrait::apply()`
- `BasisTrait::q_weights()`
- `BasisTrait::q_ref()`

说明：

- 当前 Reed 主要实现了 tensor H1 Lagrange basis（CPU），并对 `EvalMode::Div` 在 **向量场且 `ncomp == dim`** 时提供散度及其转置（与 `Grad` 组合，满足离散伴随恒等式）。
- `EvalMode::Curl`：在 **(dim,ncomp)=(2,2)** 时为标量（平面旋度）；** (3,3)** 时为三维向量旋度；正/转置均通过 `Grad` 组合实现。
- **WGPU**：`WgpuBasis` 对 tensor H1 Lagrange 的 **`Interp` / `Grad`**（含转置）在原生 `f32` 上走 compute；**`Div` / `Curl`** 在 `f32` 上为 **`basis_post` 准备（注入 / 旋度对偶权重）+ 稠密 `Grad` 或 `Gradᵀ`**，与 CPU `LagrangeBasis` 的积分点布局 `qcomp = ncomp·dim`、`comp·dim + dir` 一致。
- 单纯形 `basis_h1_simplex` 上同样支持 **`Div` / `Curl`**（与 Lagrange 张量积相同的索引约定）；与 libCEED 的 H(div)/H(curl) **专用有限元基** 相比，此处仍是 **H1 向量场的微分算子**，并非 Nédélec 等空间。

### 4.4 QFunction

libCEED 概念：

- `CeedQFunctionCreateInterior`
- `CeedQFunctionAddInput`
- `CeedQFunctionAddOutput`
- `CeedQFunctionContext`
- `CeedQFunctionByName`

Reed 对应：

- `Reed::q_function_interior()`（`context_byte_len` + 闭包接收 `ctx: &[u8]`）
- `QFunctionField`
- `QFunctionTrait`（`context_byte_len()` + `apply(ctx, ...)`）
- `QFunctionContext`（字节缓冲，`read_f64_le` / `write_f64_le` 等）
- `ClosureQFunction`
- `Reed::q_function_by_name()`
- `OperatorBuilder::qfunction_context()`（长度须与 `context_byte_len()` 一致）

当前对应情况：

| libCEED 能力 | Reed 状态 | 说明 |
|---|---|---|
| 用户自定义 interior qfunction | 已支持 | 通过闭包构造 |
| 命名 gallery qfunction | 已支持 | 当前仅 CPU gallery 的有限集合 |
| QFunction field 描述 | 已支持 | `QFunctionField { name, num_comp, eval_mode }` |
| QFunction context | 已支持（MVP） | `QFunctionContext` + 算子执行时传入 `apply`；尚无设备侧同步 API |

说明：

- 与 libCEED 一样，context 为定长字节块；gallery 默认 `context_byte_len() == 0`。

### 4.5 Operator

libCEED 概念：

- `CeedOperatorCreate`
- `CeedOperatorSetField`
- `CeedOperatorApply`
- `CeedOperatorApplyAdd`
- `CeedOperatorLinearAssembleDiagonal`

Reed 对应：

- `OperatorBuilder`
- `OperatorBuilder::field()`
- `OperatorBuilder::qfunction_context()`
- `OperatorTrait::apply()`
- `OperatorTrait::apply_add()`
- `OperatorTrait::linear_assemble_diagonal()`
- `FieldVector<'a, T>`
- `CompositeOperator<T>`（`y = sum_i A_i x`，对应 libCEED 加法型 `CeedCompositeOperator` 的 apply 语义）

`FieldVector` 与 libCEED field vector 语义映射：

| libCEED | Reed |
|---|---|
| `CEED_VECTOR_ACTIVE` | `FieldVector::Active` |
| 被动输入向量 | `FieldVector::Passive(&dyn VectorTrait<T>)` |
| `CEED_VECTOR_NONE` | `FieldVector::None` |

这是当前 Reed 与 libCEED 最贴近的一层接口设计。

## 5. 当前 Reed gallery QFunction 与 libCEED 示例对照

libCEED 内置 gallery 名称见上游 `gallery/ceed-gallery-list.h`（`CeedQFunctionRegister_*`）。Reed 通过 `Reed::q_function_by_name` 提供同名或等价实现。

| Reed / `q_function_by_name` 名称 | libCEED 注册名 | 说明 |
|---|---|---|
| `Mass1DBuild` / `Mass2DBuild` / `Mass3DBuild` | 同名 | 质量矩阵积分数据 |
| `MassApply` | 同名 | 标量质量作用 |
| `Poisson1DApply` / `Poisson2DBuild` / `Poisson2DApply` / `Poisson3DBuild` / `Poisson3DApply` | 同名 | Poisson 路径 |
| `Identity` | `Identity` | 插值场逐点拷贝（默认 1 分量；`Identity::with_components(n)` 用于多分量） |
| `Identity to scalar` | `Identity to scalar` | 保留每点第一分量（默认输入 3 分量；`IdentityScalar::with_input_components(n)`） |
| `Scale` | `Scale` | `output = alpha * input`，`alpha` 为 8 字节 `f64` LE 上下文 |
| `Scale (scalar)` | `Scale (scalar)` | 与 `Scale` 同核，保留 libCEED 双注册名 |
| `Vector3MassApply` | 同名 | 3 分量质量作用 |
| `Vector3Poisson1DApply` / `Vector3Poisson2DApply` / `Vector3Poisson3DApply` | 同名 | 3 分量 Poisson 梯度作用 |
| `Vec2Dot` / `Vec3Dot` | （Reed 扩展） | 插值向量点积，便于示例与测试 |

说明：

- **`Vector3Poisson2DApply`**：`qdata` 为 **4** 分量 / 点，与 Reed 已有 `Poisson2DApply` / `Poisson2DBuild` 一致；libCEED 注册为 **3** 对称分量，迁移时需注意打包格式。
- **复合算子**：`CompositeOperator` 实现子算子之和的 `apply` / `apply_add` / `linear_assemble_diagonal`（与 libCEED 加法组合一致）；不含 libCEED 的其它组合模式（如嵌套网格专用 API）。

## 6. API 风格对照

### libCEED 风格

- 偏 C 风格对象句柄
- 显式 destroy / restore
- 通过函数调用组合对象
- 运行时动态配置为主

### Reed 风格

- Rust trait + struct 抽象
- 通过所有权和借用管理生命周期
- 通过 builder 和 trait object 组合对象
- 泛型标量类型 `T: Scalar`

设计上可以理解为：

- libCEED 更像“C 运行时对象系统”
- Reed 更像“Rust 类型系统承载的 libCEED 对象模型”

## 7. 术语建议

今后在代码、文档、issue 和示例里，建议统一使用如下术语：

| 建议术语 | 含义 |
|---|---|
| context | `Reed<T>` / `Ceed` 顶层上下文 |
| backend | 资源后端，如 `/cpu/self` |
| vector | 全局或局部数值向量 |
| restriction | 自由度限制/散布算子 |
| basis | 参考单元基函数 |
| qfunction | 积分点点态算子 |
| operator | 完整离散算子 |
| qdata | 积分点辅助数据 |
| active/passive field | 算子字段的数据流角色 |

## 8. 当前差距清单

以下是 Reed 相对 libCEED 当前还未完全对齐、但后续很重要的点：

| 能力 | libCEED | Reed 当前状态 |
|---|---|---|
| 多维 gallery qfunction | 完整 | 已覆盖 libCEED `ceed-gallery-list.h` 中列出的名称（外加 `Vec2Dot`/`Vec3Dot`）；AtPoints 专用核等仍少 |
| 复合算子 | `CeedCompositeOperator` | 已有加法型 `CompositeOperator` |
| QFunctionContext | 支持 | 已有字节上下文；设备同步/注册字段未做 |
| 更丰富 basis 类型 | 支持 | tensor H1 Lagrange + simplex H1；`EvalMode::{Div,Curl}`（H1 向量微分；非 Nédélec 空间） |
| 更丰富 backend | 多 backend | CPU 为主；WGPU 渐进；CUDA/HIP 占位 |
| 更完整 resource 兼容 | 丰富 | 当前主要 `/cpu/self`、`/cpu/self/ref`，可选 `/gpu/wgpu` |
| 更复杂 operator 组合 | 支持 | 当前基础能力已具备，但接口和示例还需扩展 |
| WGPU 与 CPU 张量 H1 基 | — | `Interp` / `Grad` / `Div` / `Curl`（含转置、交错积分点布局）在 `f32` 上由集成测试与 CPU 对齐；算子内 QFunction 等仍以 CPU 为主 |
| WGPU 与 CPU 元限制 | — | offset 型 `ElemRestriction` 的 `NoTranspose` / `Transpose` 在 `f32` 上可走 GPU；`Transpose` 为单线程 scatter（与 `atomicCompareExchange` 不可用的后端兼容） |

### 8.1 后续 libCEED 对齐优先级（建议）

面向示例迁移与接口完备性，建议按依赖顺序推进：

1. **QFunction / 算子设备路径**：在保持 gallery 与 `ClosureQFunction` CPU 正确的前提下，为 WGPU 规划 qdata 与 `QFunctionContext` 的设备驻留与回读约定（对齐 libCEED 的 context / field 注册思路）。
2. **Strided `ElemRestriction` on WGPU**：在 offset 路径稳定后，为 strided gather / transpose 增加 GPU 实现或明确仅 CPU 的迁移注意事项。
3. **AtPoints 与边界算子**：将 `elem_restriction_at_points`、表面/体积算子组合等写入独立迁移笔记（与 `examples/` 中 ex2 类路径对应）。
4. **Operator 组合示例**：在现有 `OperatorBuilder` 与 `CompositeOperator` 上增加与 libCEED 示例结构对应的最小范例（多子域、多块相加）。
5. **整数与索引约定**：评估将「尺寸」与「restriction 偏移」统一为与 libCEED 更接近的整型策略，降低 C 示例机械翻译成本。

以上不改变当前 trait 语义；落地时逐项更新本文件 §4–§8 与 `readme.md`。

## 9. 后续维护建议

每当 Reed 新增或调整以下能力时，应同步更新本文件：

- 新增一个 libCEED 对应对象或 trait
- 新增一个 gallery qfunction
- 修改 resource 命名或 backend 支持范围
- 修改 `VectorTrait`、`BasisTrait`、`ElemRestrictionTrait`、`QFunctionTrait`、`OperatorTrait` 的核心语义
- 新增与 libCEED 示例迁移相关的公共约定

建议维护原则：

- 优先记录“概念是否等价”而不是只记录“函数名是否相似”。
- 对暂未实现的对应项明确标记为“缺失”或“部分缺失”。
- 如果 Reed 有意偏离 libCEED 设计，应在这里记录原因。
