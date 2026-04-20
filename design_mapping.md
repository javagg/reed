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
- `ElemRestrictionTrait::apply()`

说明：

- 数学语义一致：
  - `E`: global -> local
  - `E^T`: local -> global accumulate
- Reed 当前直接在 trait 中暴露 `TransposeMode`，对应 libCEED 的 restriction apply 语义。

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

- 当前 Reed 主要实现了 tensor H1 Lagrange basis。
- 这与 libCEED 的基础示例路径是对齐的，但还没有覆盖更广泛 basis 家族。

### 4.4 QFunction

libCEED 概念：

- `CeedQFunctionCreateInterior`
- `CeedQFunctionAddInput`
- `CeedQFunctionAddOutput`
- `CeedQFunctionContext`
- `CeedQFunctionByName`

Reed 对应：

- `Reed::q_function_interior()`
- `QFunctionField`
- `QFunctionTrait`
- `ClosureQFunction`
- `Reed::q_function_by_name()`

当前对应情况：

| libCEED 能力 | Reed 状态 | 说明 |
|---|---|---|
| 用户自定义 interior qfunction | 已支持 | 通过闭包构造 |
| 命名 gallery qfunction | 已支持 | 当前仅 CPU gallery 的有限集合 |
| QFunction field 描述 | 已支持 | `QFunctionField { name, num_comp, eval_mode }` |
| QFunction context | 部分缺失 | 当前没有与 libCEED 等价的独立 context 对象 |

说明：

- `QFunctionContext` 是 Reed 后续扩展的关键点，尤其在迁移 libCEED 示例时会频繁使用。

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
- `OperatorTrait::apply()`
- `OperatorTrait::apply_add()`
- `OperatorTrait::linear_assemble_diagonal()`
- `FieldVector<'a, T>`

`FieldVector` 与 libCEED field vector 语义映射：

| libCEED | Reed |
|---|---|
| `CEED_VECTOR_ACTIVE` | `FieldVector::Active` |
| 被动输入向量 | `FieldVector::Passive(&dyn VectorTrait<T>)` |
| `CEED_VECTOR_NONE` | `FieldVector::None` |

这是当前 Reed 与 libCEED 最贴近的一层接口设计。

## 5. 当前 Reed gallery QFunction 与 libCEED 示例对照

| Reed gallery 名称 | 对应 libCEED 概念 | 当前状态 |
|---|---|---|
| `Mass1DBuild` | 1D 质量算子 qdata 构建 | 已实现 |
| `MassApply` | 质量算子作用 | 已实现 |
| `Poisson1DApply` | 1D Poisson/扩散作用 | 已实现 |

说明：

- 当前 Reed 的 gallery 主要覆盖 1D 示例所需能力。
- 若要继续对标 libCEED 的 `ex1/ex2/ex3` 多维版本，通常需要补齐：
  - 2D/3D `MassBuild`
  - 2D/3D `PoissonBuild`
  - 2D/3D `PoissonApply`
  - 更明确的组合算子支持策略

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
| 多维 gallery qfunction | 完整 | 部分缺失 |
| QFunctionContext | 支持 | 缺失 |
| 更丰富 basis 类型 | 支持 | 主要是 tensor H1 Lagrange |
| 更丰富 backend | 多 backend | 当前主要 CPU |
| 更完整 resource 兼容 | 丰富 | 当前只接受 `/cpu/self`、`/cpu/self/ref` |
| 更复杂 operator 组合 | 支持 | 当前基础能力已具备，但接口和示例还需扩展 |

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
