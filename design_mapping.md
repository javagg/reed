# Reed 与 libCEED 概念对照

这份文档用于维护 Reed 与 libCEED 之间的概念、数据类型和接口映射，作为后续设计、实现和示例迁移时的参考。

**对齐度快照、风险、WASM 能力矩阵与 `CeedInt` 约定摘要**：见仓库根目录 [`libceed_alignment_assessment.md`](./libceed_alignment_assessment.md)（附录 A / B）。

## 0.1 CPU 对齐速查（发布口径）

该速查仅针对 **`/cpu/self`**，用于快速判断“Reed CPU 后端对 libCEED 的可发布对齐范围”：

| 维度 | 状态 | 口径 |
|---|---|---|
| Vector / Restriction / Basis | ✅ 已对齐（子集） | 覆盖常见示例路径（offset/strided/at-points + tensor/simplex H1 常用能力）。 |
| QFunction（interior + context） | ✅ 已对齐（子集） | `apply(ctx,...)`、gallery 命名解析与 context 字节读写语义稳定。 |
| Operator apply / apply_add | ✅ 已对齐（子集） | 单/多 active 场 apply 路径可用，满足典型离散算子迁移。 |
| Operator `Adjoint`（离散伴随） | ⚠️ 条件对齐 | 依赖 `apply_operator_transpose`；`Weight` 等高级组合仍有边界。 |
| 线装配（Diagonal / Dense / CSR，含 add） | ✅ 已对齐（子集） | `linear_assemble_*` + `linear_assemble_*_add`（dense 与 CSR）可用并有测试覆盖。 |
| FDM inverse API 形状 | ⚠️ API 对齐，实现替代 | 现有小 `n` 稠密逆（`CpuFdmDenseInverseOperator`）+ Jacobi 近似逆（`CpuFdmJacobiInverseOperator`），均非 libCEED 原生 tensor-FDM。 |
| `CeedMatrix` 对象语义 | ⚠️ 部分对齐 | 已有 `CeedMatrix`/`CeedMatrixStorage` 句柄与 symbolic/numeric 状态、dense/CSR set+add 装配语义；仍非 libCEED 后端托管矩阵对象的 1:1 全量实现。 |
| CompositeOperator | ✅ 已对齐（子集） | 加法 apply/diag-add 对齐；矩阵装配与 FDM 在复合算子上显式 `Err`。 |

**一句话建议**：若发布目标是“CPU 主机端离散算子迁移与常见教学示例”，可按 **中高对齐** 口径对外说明；若目标是“libCEED 全 API 逐项等价”，仍需补 `CeedMatrix` 托管模型细节与原生 tensor-FDM。

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
| `CeedCompositeOperator` | `CompositeOperator<T>` | 加法型组合；`apply` / `apply_add` / `linear_assemble_diagonal` / **`linear_assemble_add_diagonal`** 为子算子之和（后者 **累加** 到调用方缓冲）。**稠密** `linear_assemble_symbolic` / `linear_assemble` / **`linear_assemble_add`** 与 **CSR** `linear_assemble_csr_matrix` / **`linear_assemble_csr_matrix_add`** 在复合上 **`Err`**（无共享矩阵槽；须在子算子上分别调用） |

## 3. 数据类型映射

### 3.1 标量与整数

| libCEED | Reed | 说明 |
|---|---|---|
| `CeedScalar` | `T: Scalar` | Reed 当前用泛型 trait 支持 `f32/f64` |
| `CeedInt` | `CeedInt`（`i32`）/ `CeedSize`（`usize`） | `CeedInt` 用于索引/偏移语义，`CeedSize` 用于尺寸；`i64` 绑定桥接与溢出规则见 [`libceed_alignment_assessment.md`](./libceed_alignment_assessment.md) **附录 B** |
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
- **`CeedInt` / `i64` 桥接**（与 §3.2 一致）：`Reed::elem_restriction_ceed_int_offsets`、`elem_restriction_at_points_ceed_int_offsets`、`strided_elem_restriction_ceed_int_strides` — 将 `i64` 安全转换为 `i32` 后委托上述工厂（便于 C 绑定、`int64_t` 缓冲区零拷贝意图下的显式收窄）
- `ElemRestrictionTrait::apply()`

说明：

- 数学语义一致：
  - `E`: global -> local
  - `E^T`: local -> global accumulate
- Reed 当前直接在 trait 中暴露 `TransposeMode`，对应 libCEED 的 restriction apply 语义。
- **CPU**：集成测试 `test_cpu_elem_restriction_at_points_matches_elem_restriction` 校验 `elem_restriction_at_points` 与同名参数的 `elem_restriction` 在 `NoTranspose` / `Transpose` 下数值完全一致（实现上前者委托后者，对齐 libCEED 命名）。
- **WGPU**：offset / strided 的 gather 与 `f32` transpose scatter 等见 `crates/wgpu` 与 §8；`elem_restriction_at_points` 与 offset 共用实现。

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
- **单纯形 `basis_h1_simplex`**：除三角形 / 四面体外，已实现 **参考线段 `ElemTopology::Line` 上 P1–P3**（1D simplex）；`Pyramid` / `Prism` 等枚举见 [`ElemTopology`](crates/core/src/enums.rs) 文档说明（占位与后续 FE 扩展）。
- `EvalMode::Curl`：在 **(dim,ncomp)=(2,2)** 时为标量（平面旋度）；** (3,3)** 时为三维向量旋度；正/转置均通过 `Grad` 组合实现。
- **WGPU**：`WgpuBasis` 对 tensor H1 Lagrange 的 **`Interp` / `Grad`**（含转置）在原生 `f32` 上走 compute；**标量 `EvalMode::Weight` 且 `transpose`** 与 CPU 一致，**复用 `Interpᵀ` 的 compute 核**（`ncomp != 1` 仍走 CPU 并报错）；**`Div` / `Curl`** 在 `f32` 上为 **`basis_post` 准备（注入 / 旋度对偶权重）+ 稠密 `Grad` 或 `Gradᵀ`**，与 CPU `LagrangeBasis` 的积分点布局 `qcomp = ncomp·dim`、`comp·dim + dir` 一致。
- 单纯形 `basis_h1_simplex` 上同样支持 **`Div` / `Curl`**（与 Lagrange 张量积相同的索引约定）；与 libCEED 的 H(div)/H(curl) **专用有限元基** 相比，此处仍是 **H1 向量场的微分算子**，并非 Nédélec 等空间。

### 4.4 QFunction

libCEED 概念：

- `CeedQFunctionCreateInterior` / **`CeedQFunctionCreateInteriorByName`**
- **`CeedQFunctionCreateExterior`**（及 exterior 变体；与 interior 在迁移/面算子语义上区分）
- `CeedQFunctionAddInput`
- `CeedQFunctionAddOutput`
- `CeedQFunctionContext`
- `CeedQFunctionByName`

Reed 对应：

- `Reed::q_function_interior()` / **`Reed::q_function_exterior()`**（二者校验相同；后者构造 **`QFunctionCategory::Exterior`** 的 `ClosureQFunction`，前者为 **`Interior`**）
- **`QFunctionCategory`**（`Interior` / `Exterior`）与 **`QFunctionTrait::q_function_category()`**（gallery 与默认闭包为 `Interior`）
- `QFunctionField`
- `QFunctionTrait`（`context_byte_len()` + `apply(ctx, ...)`）
- `QFunctionContext`（定长字节缓冲；`read_*_le` / `write_*_le`；**`read_*_le_bytes` / `write_*_le_bytes`** 作用于任意 `&[u8]` / `&mut [u8]`；**`from_field_layout` + `QFunctionContextField` / `QFunctionContextFieldKind`** 提供 libCEED 风格的字段元数据与 `read_field_f64` 等命名访问；**`host_needs_device_upload` / `mark_host_synced_to_device`** 为 GPU 后端预留 host↔device 一致性标记）
- `ClosureQFunction`
- `Reed::q_function_by_name()`
- `OperatorBuilder::qfunction_context()`（长度须与 `context_byte_len()` 一致）

当前对应情况：

| libCEED 能力 | Reed 状态 | 说明 |
|---|---|---|
| 用户自定义 interior qfunction | 已支持 | 通过 `q_function_interior` 闭包构造 |
| 用户自定义 exterior qfunction（**元数据**） | 已支持 | `q_function_exterior`：`apply` 与 interior 闭包 **同一路径**；`q_function_category() == Exterior`，便于迁移脚本与上层与 libCEED exterior 注册 **概念对齐**（**无** 自动面专用积分点或独立设备核） |
| 命名 gallery qfunction | 已支持 | CPU gallery 有限集合；各核实现 `QFunctionTrait<T>`，`T: Scalar`（`f32` / `f64`）；**`q_function_category() == Interior`** |
| QFunction field 描述 | 已支持 | `QFunctionField { name, num_comp, eval_mode }` |
| QFunction context | 已支持（扩展） | 字节缓冲 + 可选字段注册 + host 脏标记（WGPU 算子路径仍待接线上传） |

说明：

- 与 libCEED 一样，context 为定长字节块；gallery 默认 `context_byte_len() == 0`。
- **标量类型**：命名 gallery 与 `reed_cpu::q_function_by_name::<T>` / `Reed<T>::q_function_by_name` 对齐同一 `T`；算子中向量与 QFunction 标量须一致。独立调用工厂函数且无法推断 `T` 时需显式 `::<f32>` / `::<f64>`。
- 闭包 / 自定义核里收到的 `ctx: &[u8]` 与 `QFunctionContext::as_bytes()` 布局相同；优先使用 `QFunctionContext::read_f64_le_bytes(ctx, offset)` 等（及对应的 `write_*_le_bytes`），与实例方法共享边界检查与错误信息；亦可自行 `from_le_bytes` 或拷入 `QFunctionContext::from_bytes` 再调用 `read_*_le`。

### 4.5 Operator

libCEED 概念：

- `CeedOperatorCreate`
- `CeedOperatorSetField`
- `CeedOperatorApply`
- `CeedOperatorApplyAdd`
- `CeedOperatorLinearAssembleDiagonal`
- `CeedOperatorLinearAssembleAddDiagonal`
- **`CeedOperatorLinearAssembleSymbolic` / `CeedOperatorLinearAssemble`**（稀疏/稠密矩阵装配）
- **`CeedOperatorLinearAssembleAdd`**（在已有矩阵结构中 **累加** 数值；Reed：稠密 **`linear_assemble_add`**、CSR **`linear_assemble_csr_matrix_add`**）
- **`CeedOperatorCreateFDMElementInverse`**（FDM 单元逆等）
- **装配数据释放**（libCEED 常通过 **`CeedMatrixDestroy`** 等与矩阵对象解绑；Reed 稠密槽见下 **`CpuOperator::clear_dense_linear_assembly`**）

Reed 对应：

- `OperatorBuilder`
- `OperatorBuilder::field()`
- `OperatorBuilder::qfunction_context()`
- `OperatorBuilder::operator_label()`（对齐 `CeedOperatorSetName` / 调试日志）
- **`CpuOperator::dense_linear_assembly_n`** / **`dense_linear_assembly_numeric_ready`**（查询稠密槽是否已 symbolic / 是否已完成数值列装配；**`reed-cpu`** **`dense_linear_assembly_probes_track_symbolic_numeric_and_clear`**）
- **`CpuOperator::clear_dense_linear_assembly`**（**Reed 扩展**：释放稠密 **`O(n²)`** 装配槽；§4.5.1 表；**`reed-cpu`** 单测 **`clear_dense_linear_assembly_idempotent_without_slot`**）
- `OperatorTrait::apply()`
- `OperatorTrait::apply_add()`
- `OperatorTrait::apply_with_transpose()` / `apply_add_with_transpose()`（`OperatorTransposeRequest`：`Forward` 委托 `apply` / `apply_add`；`Adjoint`：`CpuOperator` 在单缓冲或命名场映射、单 active 输入/输出场、`QFunctionTrait::supports_operator_transpose()` 为真时实现离散伴随；向量场 `EvalMode::Weight` 仍不支持；其它类型默认 `Adjoint`→`Err`）
- `QFunctionTrait::apply_operator_transpose_with_primal()`（可选扩展：在 `apply_operator_transpose` 基础上获取最近一次前向积分点输入；默认回退到 `apply_operator_transpose`）
- `OperatorTrait::operator_label()`（对齐 `CeedOperatorGetName` 语义）
- `OperatorTrait::apply_field_buffers()` / `apply_add_field_buffers()`（多 active 场 / libCEED 多向量 `Apply`；`CpuOperator` 与 `CompositeOperator*` 已实现，默认 trait 仍为未实现）
- `OperatorTrait::apply_field_buffers_with_transpose()` / `apply_add_field_buffers_with_transpose()`（`Forward` 委托上两行；`Adjoint`：`CpuOperator` 按场名 range/domain cotangent 缓冲实现离散伴随，`CompositeOperator*` 在此基础上对子算子结果求和；默认类型 `Adjoint`→`Err`）
- `OperatorTrait::linear_assemble_diagonal()`
- `OperatorTrait::linear_assemble_add_diagonal()`（对齐 **`CeedOperatorLinearAssembleAddDiagonal`**；**不** 清零 `assembled`，**`CpuOperator` / `CompositeOperator*`** 已实现）
- **`OperatorTrait::linear_assemble_add()`**（对齐 **`CeedOperatorLinearAssembleAdd`**，稠密槽 **`+=`** 列；**`CpuOperator`**；复合 **`Err`**）
- **`OperatorTrait::linear_assemble_csr_matrix_add()`**（同上 libCEED API 的 CSR 侧；**`&mut CsrMatrix`**；**`CpuOperator`**；复合 **`Err`**）
- **`OperatorAssembleKind`**（`Diagonal` / `LinearSymbolic` / `LinearNumeric` / **`LinearCsrNumeric`** / `FdmElementInverse`；`#[non_exhaustive]`）与 **`OperatorTrait::operator_supports_assemble`**（**`CpuOperator`**：`Diagonal` / `LinearSymbolic` / `LinearNumeric` / **`LinearCsrNumeric`** 为 `true` **当且仅当** `active_global_dof_len` 有定义（**`LinearNumeric` / `LinearCsrNumeric`** 同时覆盖 **set** 与 **add** 装配：`linear_assemble` / **`linear_assemble_add`**、`linear_assemble_csr_matrix` / **`linear_assemble_csr_matrix_add`**，以及 `CeedMatrix` 句柄路径 `linear_assemble_ceed_matrix` / `linear_assemble_add_ceed_matrix`）；**`FdmElementInverse`** 为 `true` **当且仅当** `active_global_dof_len ≤ FDM_DENSE_MAX_N`（默认 256）。**`CompositeOperator` / `CompositeOperatorBorrowed`**：**`FdmElementInverse`** 与 **`LinearCsrNumeric`** 恒 **`false`**；其余 probe 为子算子 **逻辑与**；**`operator_create_fdm_element_inverse`** / **`linear_assemble_csr_matrix`** / **`linear_assemble_csr_matrix_add`** / **`linear_assemble_symbolic`** / **`linear_assemble`** / **`linear_assemble_add`** / **`linear_assemble_ceed_matrix`** / **`linear_assemble_add_ceed_matrix`** 对复合算子固定 `Err`。自定义算子仍可用 trait 默认：仅 `Diagonal` 为 `true`。）
- **`OperatorTrait::linear_assemble_symbolic` / `linear_assemble` / `linear_assemble_add`**：**`CpuOperator`** 在 `active_global_dof_len` 有定义时实现 **稠密全局 `n×n`**（**列主序** `a[row + col*n]`，内存 `O(n²)`；每次 **`linear_assemble_symbolic`** **分配或替换** 稠密槽并重置 **`numeric_done`**，直至下一次 **`linear_assemble` / `linear_assemble_add`**；`linear_assemble` **写入**列、`linear_assemble_add` **累加**列，均需先 **`linear_assemble_symbolic`**；由 `n` 次前向 `apply` 得列 `A e_j`，**仅当** `apply` 对 active 未知为线性时等于整体 Jacobian）。**`CpuOperator::clear_dense_linear_assembly`**：释放稠密槽（**不影响** `apply` / CSR 装配；集成测 **`test_cpu_operator_libceed_dense_linear_assemble_and_fdm_stub`**）。**`CompositeOperator` / `CompositeOperatorBorrowed`** 对稠密 `linear_assemble*`（含 **`linear_assemble_add`**）返回 **`Err`**（无共享矩阵槽）。**`operator_create_fdm_element_inverse`**：**`CpuOperator`** 在 `n ≤ FDM_DENSE_MAX_N` 时 **`Ok`**，创建时在本地缓冲按 `n` 次前向 `apply` 组装规范 \(A\)（**不读取也不改写** 稠密槽，因此不受先前 `linear_assemble_add` 累加槽影响）；内部 Gauss–Jordan 得 **`CpuFdmDenseInverseOperator`**（**非** libCEED 张量 FDM，小 `n` 下为 \(A^{-1}\)）；`n` 过大或奇异则 **`Err`**；复合算子显式拒绝（`reed-cpu`：**`fdm_creation_does_not_mutate_dense_slot`**）。 
- **`OperatorTrait::linear_assemble_ceed_matrix` / `linear_assemble_add_ceed_matrix`**：面向 `CeedMatrix` 句柄执行 dense/CSR 的 set+add 装配（当前由 `CpuOperator` 实现；`CompositeOperator*` 显式 `Err`）。
- **`operator_create_fdm_element_inverse_jacobi`**：对角装配后形成 `CpuFdmJacobiInverseOperator`（结构化近似逆，轻量替代；仍非 libCEED tensor-FDM）。
- `OperatorTrait::check_ready()`（对齐 `CeedOperatorCheckReady`；[`CpuOperator`](crates/cpu/src/operator.rs) 校验各侧 active restriction 一致、被动场全局长度、`num_elements` / `num_qpoints` 与基一致；[`CompositeOperator`](crates/cpu/src/composite_operator.rs) 逐子算子委托）
- `FieldVector<'a, T>`
- `CompositeOperator<T>`（`y = sum_i A_i x`，对应 libCEED 加法型 `CeedCompositeOperator` 的 apply 语义）

`FieldVector` 与 libCEED field vector 语义映射：

| libCEED | Reed |
|---|---|
| `CEED_VECTOR_ACTIVE` | `FieldVector::Active` |
| 被动输入向量 | `FieldVector::Passive(&dyn VectorTrait<T>)` |
| `CEED_VECTOR_NONE` | `FieldVector::None` |

这是当前 Reed 与 libCEED 最贴近的一层接口设计。

#### 4.5.1 算子迁移对齐：本阶段「可交付」范围（closure）

以下能力视为 **与 libCEED 离散算子迁移路径对齐的闭合里程碑**（在 `reed` + `reed_cpu` 当前语义下可依赖；后续 WGPU 全算子等为独立里程碑）：

| libCEED 能力 | Reed 状态 |
|---|---|
| `CeedOperatorCreate` / `SetField` | `OperatorBuilder::field` + `qfunction` / `qfunction_context`；可选 `OperatorBuilder::operator_label`（`CeedOperatorSetName`） |
| `CeedOperatorGetName` / `SetName` | `OperatorTrait::operator_label`；`CpuOperator` 存 builder 设置的字符串 |
| `CeedOperatorApply` + `CeedTransposeMode` | `OperatorTrait::apply_with_transpose` / `apply_add_with_transpose`（`Forward`≈`CEED_NOTRANSPOSE`；`Adjoint`≈`CEED_TRANSPOSE`：**`CpuOperator` 在 v1 约束下实现离散伴随**——qfunction 须 `apply_operator_transpose`、单缓冲或命名场映射、标量 `Weight` 与 basis 约定见 §4.5.1；其余类型或未满足条件时仍 `Err`） |
| `CeedOperatorApply` / `ApplyAdd` | `OperatorTrait::apply` / `apply_add`（单 active 输入缓冲 + 单 active 输出缓冲）；**多 active 场**（qfunction 输入/输出侧出现多个不同 `field_index` 的 `FieldVector::Active`）时用 `OperatorTrait::apply_field_buffers` / `apply_add_field_buffers`（`CpuOperator` 提供实现；其它类型默认返回 `Operator` 错误；按场名 `&[(&str, &dyn VectorTrait)]` / `&mut [(&str, &mut dyn VectorTrait)]`）；`apply` 前仍对单缓冲路径校验 `active_input_global_len` / `active_output_global_len` |
| `CeedOperatorLinearAssembleDiagonal` | `OperatorTrait::linear_assemble_diagonal`（要求方阵：`active_global_dof_len` 与 `assembled.len()` 一致；**多 active 场** 时 `active_global_dof_len` 为 `Err`，对角装配需在单全局向量空间上编排） |
| `CeedOperatorLinearAssembleAddDiagonal` | `OperatorTrait::linear_assemble_add_diagonal`（同上尺寸约束；**累加** `diag(A)`；**`CompositeOperator`**：各子算子 **`linear_assemble_add_diagonal`** 依次累加；集成测 **`test_cpu_operator_libceed_dense_linear_assemble_and_fdm_stub`** 校验与稠密对角一致） |
| `CeedOperatorLinearAssembleSymbolic` / `CeedOperatorLinearAssemble` | **`CpuOperator`**：`linear_assemble_symbolic` **每次替换** 稠密槽（清零、**`numeric_done`** 待后续 **`linear_assemble`**）；`linear_assemble` 填 **稠密** 列主序缓冲（`assembled_linear_matrix_col_major`）；**`reed-cpu`** **`second_linear_assemble_symbolic_resets_numeric_state`**。另：**`csr_sparsity_from_offset_restriction`** + **`ElemRestrictionTrait::assembled_csr_pattern`** + **[`OperatorTrait::linear_assemble_csr_matrix`]** / **`CpuOperator::linear_assemble_csr_matrix`**；便捷 **`CpuOperator::linear_assemble_csr_from_elem_restriction`**。**`CsrMatrix::mul_vec`** 与 `apply` 在集成测 **`test_mass_csr_assembly_matches_dense_columns`**、**`test_poisson_1d_csr_assembly_matches_dense_columns`** 中对齐。**`CompositeOperator` / `CompositeOperatorBorrowed`**：稠密 `linear_assemble*`、**`linear_assemble_csr_matrix`**、**`linear_assemble_csr_matrix_add`** 均为 `Err`（须在各子算子上分别装配） |
| `CeedOperatorLinearAssembleAdd` | **`CpuOperator`**：**稠密** **`OperatorTrait::linear_assemble_add`**（在已有 symbolic 缓冲上对每列 **`+=`**，与 **`linear_assemble`** 相同前置检查）；**CSR** **`OperatorTrait::linear_assemble_csr_matrix_add`** / **`CpuOperator::linear_assemble_csr_matrix_add(&mut CsrMatrix)`**（对 pattern 内 **`values` 累加**；维数与 `values.len()==nnz` 校验）。**`LinearNumeric` / `LinearCsrNumeric`** 的 **`operator_supports_assemble`** 与 **set** 装配一致。**`CompositeOperator*`**：二者均 **`Err`**。集成测：稠密 **`2A`**（`test_cpu_operator_libceed_dense_linear_assemble_and_fdm_stub`，于 FDM 验证之后 **`linear_assemble` 重置再 `linear_assemble_add`**）、CSR 从零 **`add`** 对齐 **`linear_assemble_csr_matrix`** 与 **二次 `add`**（**`test_mass_csr_assembly_matches_dense_columns`**、**`test_poisson_1d_csr_assembly_matches_dense_columns`**） |
| （装配矩阵 / `CeedMatrix` 生命周期：显式释放；无与稠密槽一一对应的单一 C 符号） | **`CeedMatrix` 句柄**：`CeedMatrixStorage::{DenseColMajor,Csr}` + symbolic/numeric 状态；`OperatorTrait::linear_assemble_ceed_matrix` / `linear_assemble_add_ceed_matrix`（由 `CpuOperator` 实现）执行 set+add 装配。**`CpuOperator::clear_dense_linear_assembly`** 仍可释放内部稠密槽（`O(n²)`，不影响 CSR 与 apply）。 |
| `CeedOperatorCheckReady` | `OperatorTrait::check_ready`；`CpuOperator` 校验各侧 restriction、被动场长度、`num_elements` / `num_qpoints`；`CompositeOperator` / `CompositeOperatorBorrowed` 委托子算子 |
| `CeedCompositeOperator`（加法） | `CompositeOperator` / `composite_operator_refs` + `global_vector_len_hint` 严格合并；子算子可为单缓冲或命名缓冲路径。若任一子算子 `requires_field_named_buffers == true`，复合算子的单缓冲 `apply*` / `apply*_with_transpose(Adjoint)` 返回提示错误，应改用 `apply_field_buffers*`（含 `Adjoint`）。`CeedMatrix` 句柄装配在复合算子上返回 `Err`，迁移时可回退为对子算子逐个 `linear_assemble_add_ceed_matrix` 求和（示例：`examples/composite_operator_refs.rs`；集成测：`test_composite_operator_refs_ceed_matrix_fallback_sum_suboperators`）。 |
| 非对称 build（输入/输出不同全局维，如 `Mass*DBuild`） | `active_input_global_len` ≠ `active_output_global_len` 为预期情形；`active_global_dof_len` 仅在同维方阵算子上有 `Ok` |

**本阶段明确不包含（需在其它里程碑或上层编排实现）：**

- **libCEED 式 C 句柄与独立 exterior 运行时**：已有 **`q_function_exterior` + `QFunctionCategory`** 做 **分类元数据**；**无** `CeedQFunction` 级独立句柄、**无** 自动面元 / 外法向积分点编排（与 interior 闭包 **同一 CPU `apply`**）。
- **libCEED `CeedMatrix` 对象与后端装配 API 全量对齐** 与 **张量 FDM**（`CeedOperatorCreateFDMElementInverse` 的 libCEED 原生实现）：仍为缺口；Reed 已提供 `CeedMatrix` 句柄 + dense/CSR set+add 装配语义、**小 `n` 稠密逆（`CpuFdmDenseInverseOperator`）** 与 **Jacobi 近似逆（`CpuFdmJacobiInverseOperator`）**，但仍非 libCEED 后端托管矩阵与原生 tensor-FDM 的 1:1。
- **`CeedTransposeMode` / `CEED_TRANSPOSE` 下任意算子的完整离散伴随**：`CpuOperator` 的 `Adjoint` 要求 **qfunction 实现 `apply_operator_transpose`**（gallery 已含标量/向量 **Poisson apply**、`Identity` / `Identity to scalar`、`MassApply`、**`MassApplyInterpTimesWeight`**（第二路 `Weight` 被动槽）、**`Scale` / `Scale (scalar)`**、向量 Mass 等；闭包 QF 等默认仍 `Err`）。**单缓冲**：`apply_with_transpose` / `apply_add_with_transpose`。**多 active 场**：`OperatorTrait::apply_field_buffers_with_transpose` / `apply_add_field_buffers_with_transpose`，其中 `Adjoint` 时 `inputs` 为各 **active 输出场** 上的 range cotangent，`outputs` 为各 **active 输入场** 上的 domain cotangent（**被动 / `None` 输入槽** 不要求出现在 `outputs` 映射中）。**`CompositeOperator` / `CompositeOperatorBorrowed`** 的单缓冲 `Adjoint` 为各子算子伴随之和；其 **`apply_field_buffers*`（含 `Adjoint`）** 也已实现为对子算子对应路径求和。一般非线性或依赖 active 场在 qp 上前向值的伴随不在此 v1 范围；自定义 `OperatorTrait` 可覆盖。
- **WGPU** 上整条 `OperatorTrait::apply`：**qfunction 与算子编排仍在 CPU**（`CpuOperator`）；**restriction / basis / 向量** 在资源为 `/gpu/wgpu` 时可选用 WGPU 实现，数据经各对象的 host 切片与内部 GPU 路径同步。另：**`GpuRuntime::dispatch_mass_apply_qp_f32`** / **`mass_apply_qp_f32_host`** 等提供 **`MassApply` 点态** 的独立 compute（见 §8 表），尚未由 `CpuOperator` 自动调用。集成测（`tests/integration.rs`，需 `wgpu-backend`）：**`test_wgpu_hybrid_mass_operator_apply_matches_cpu`** 校验与纯 CPU 栈的 `MassApply` **前向** `apply` 一致；**`test_wgpu_hybrid_mass_operator_transpose_matches_cpu`** 校验 **`apply_with_transpose`（`Forward` / `Adjoint`）** 与 CPU 参考及对称伴随时 **GPU 混合栈** 一致。

## 5. 当前 Reed gallery QFunction 与 libCEED 示例对照

libCEED 内置 gallery 名称见上游 [`gallery/ceed-gallery-list.h`](https://github.com/CEED/libCEED/blob/main/gallery/ceed-gallery-list.h)（`CEED_GALLERY_QFUNCTION` 注册顺序）。Reed 通过 `Reed::q_function_by_name` 提供 **同名** 实现，并额外暴露 **`reed_cpu::QFUNCTION_LIBCEED_MAIN_GALLERY_NAMES`**（与上游列举 **18** 个 interior 核一一对应，含文档链接）供迁移脚本做 **子集校验**；**扩展名**（`Vector2*`、`Vec2Dot`/`Vec3Dot`、`MassApplyInterpTimesWeight*`、AtPoints 别名、`IdentityScalar` / `ScaleScalar` 别名等）由 **`QFUNCTION_INTERIOR_GALLERY_NAMES`**（与 `q_function_by_name` 同步）统一列出。

| Reed / `q_function_by_name` 名称 | libCEED 注册名 | 说明 |
|---|---|---|
| `Mass1DBuild` / `Mass2DBuild` / `Mass3DBuild` | 同名 | 质量矩阵积分数据 |
| `MassApply` | 同名 | 标量质量作用 |
| `MassApplyInterpTimesWeight` | （Reed 扩展） | 与 `MassApply` 相同的 `v=u·w` 核与伴随；第二输入声明为 **`EvalMode::Weight`**（便于覆盖算子组装里 **Weight 被动槽** 路径） |
| `Poisson1DBuild` / `Poisson1DApply` / `Poisson2DBuild` / `Poisson2DApply` / `Poisson3DBuild` / `Poisson3DApply` | 同名 | Poisson 路径（1D build：`qdata = w/J`） |
| `Identity` | `Identity` | 插值场逐点拷贝（默认 1 分量；`Identity::with_components(n)` 用于多分量） |
| `Identity to scalar` | `Identity to scalar` | 保留每点第一分量（默认输入 3 分量；`IdentityScalar::with_input_components(n)`） |
| `Scale` | `Scale` | `output = alpha * input`，`alpha` 为 8 字节 `f64` LE 上下文 |
| `Scale (scalar)` | `Scale (scalar)` | 与 `Scale` 同核，保留 libCEED 双注册名 |
| `Vector3MassApply` | 同名 | 3 分量质量作用 |
| `Vector3Poisson1DApply` / `Vector3Poisson2DApply` / `Vector3Poisson3DApply` | 同名 | 3 分量 Poisson 梯度作用 |
| `Vector2MassApply` / `Vector2Poisson1DApply` / `Vector2Poisson2DApply` | （Reed 扩展） | 2 分量向量场；布局与 `Vector3*` 相同规则（2D：`du` 为 `2×dim` 分量 / 点） |
| `Vec2Dot` / `Vec3Dot` | （Reed 扩展） | 插值向量点积，便于示例与测试 |
| `MassApplyAtPoints` / `MassApplyInterpTimesWeightAtPoints` / `ScaleAtPoints` / `IdentityAtPoints` / `Poisson2DApplyAtPoints` | （Reed 别名） | 与 `MassApply` / **`MassApplyInterpTimesWeight`** / `Scale` / `Identity` / `Poisson2DApply` **同一 CPU 实现**；命名用于与 AtPoints restriction 组合的 libCEED 风格迁移 |

说明：

- **`Scale` / `Scale (scalar)`**：与 libCEED 相同，上下文仍为 **8 字节 `f64` 小端**；Reed 读入后按 `T` 转换再参与乘法（与双精度 libCEED 示例二进制兼容）。
- **`Vector3Poisson2DApply`**：`qdata` 为 **4** 分量 / 点，与 Reed 已有 `Poisson2DApply` / `Poisson2DBuild` 一致；libCEED 注册为 **3** 对称分量，迁移时需注意打包格式。
- **复合算子**：`CompositeOperator` 实现子算子之和的 `apply` / `apply_add` / `linear_assemble_diagonal` / **`linear_assemble_add_diagonal`**（与 libCEED 加法组合一致）；**稠密 / CSR 矩阵装配（含 `linear_assemble_add` / `linear_assemble_csr_matrix_add`）** 须在子算子上分别调用。不含 libCEED 的其它组合模式（如嵌套网格专用 API）。

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
| 多维 gallery qfunction | 完整 | **`QFUNCTION_LIBCEED_MAIN_GALLERY_NAMES`** 与上游 `gallery/ceed-gallery-list.h`（main）**逐项一致**；`QFUNCTION_INTERIOR_GALLERY_NAMES` 另含 **`Vector2*`**、`Vec2Dot`/`Vec3Dot`、**`MassApplyInterpTimesWeight`**、**`IdentityScalar`/`ScaleScalar`**（等价于 libCEED 用户可见名 **`Identity to scalar`/`Scale (scalar)`**）、及 **`MassApplyAtPoints` / `MassApplyInterpTimesWeightAtPoints` / `ScaleAtPoints` / `IdentityAtPoints` / `Poisson2DApplyAtPoints`** 等 AtPoints 迁移别名 |
| 复合算子 | `CeedCompositeOperator` | 已有加法型 `CompositeOperator` |
| QFunctionContext | 支持 | 字段注册 + host 脏标记；**`reed_wgpu::GpuRuntime::sync_qfunction_context_to_buffer`**（及 `write_qfunction_context_to_buffer`）可把上下文写入任意 `COPY_DST` 缓冲并在成功后清脏；**`GpuRuntime::dispatch_mass_apply_qp_f32`** / **`dispatch_mass_apply_qp_transpose_accumulate_f32`**（已有缓冲）及 **`mass_apply_qp_f32_host`** / **`mass_apply_qp_transpose_accumulate_f32_host`**（主机切片往返）提供与 gallery **`MassApply`** 一致的 **`f32` 积分点点态**（`reed_wgpu` 单测 + `tests/integration.rs`：`test_wgpu_gpu_runtime_mass_apply_qp_host_bridge`）；与 **`CpuOperator`** 的自动接线仍待完成 |
| 更丰富 basis 类型 | 支持 | tensor H1 Lagrange + **线段** / 三角形 / 四面体 simplex H1–**P3**；`EvalMode::{Div,Curl}`（H1 向量微分；非 Nédélec 空间）。棱柱/楔、金字塔参考元上的 H1 基仍为后续扩展 |
| 更丰富 backend | 多 backend | CPU 为主；WGPU 渐进；CUDA/HIP 占位 |
| 更完整 resource 兼容 | 丰富 | 当前主要 `/cpu/self`、`/cpu/self/ref`，可选 `/gpu/wgpu` |
| 更复杂 operator 组合 | 支持 | **§4.5.1** 所列 CPU 侧迁移项已闭合；**多 active 场** 的 `Apply` / **`apply_field_buffers_with_transpose(Adjoint)`** 已由 `apply_field_buffers*` 覆盖；**被动输入槽**（含 `EvalMode::Weight`）不要求出现在 `Adjoint` 的命名 `outputs` 中（与 `reed_cpu::CpuOperator` 实现一致，见 `MassApplyInterpTimesWeight` 集成测） |
| WGPU 与 CPU 张量 H1 基 | — | `Interp` / `Grad` / `Div` / `Curl`（含转置、交错积分点布局）在 `f32` 上由集成测试与 CPU 对齐；算子级 **gallery QFunction** 仍以 **CPU** 为主；**`MassApply` 点态乘** 已有 **独立 `GpuRuntime` dispatch**（见上行） |
| WGPU 与 CPU 元限制 | — | offset 与 strided：`NoTranspose`（gather）在 **`f32` 与 `f64`** 上可走 GPU（`f64` 为 `u32` 对按 IEEE 位复制，无 WGSL `f64` 算术）；`Transpose`（scatter）在 `f32` 上为单线程 scatter，`f64` 仍走 CPU（需双精度加法） |

### 8.1 后续 libCEED 对齐优先级（建议）

面向示例迁移与接口完备性，建议按依赖顺序推进：

1. **QFunction / 算子设备路径**：在保持 gallery 与 `ClosureQFunction` CPU 正确的前提下，为 WGPU 规划 qdata 与 `QFunctionContext` 的设备驻留与回读约定（对齐 libCEED 的 context / field 注册思路）。**进展**：`GpuRuntime` 已提供 **`MassApply` 标量 `f32` qp 核**（前向与转置累加）；下一步是把 restriction/basis 产出的 **device 缓冲** 与该核及回读 **接入 `CpuOperator` 可选路径**（或专用设备算子壳层）。
2. **Strided `ElemRestriction` on WGPU**：strided 的 gather / transpose（`f32`）与 **gather（`f64`，位复制）** 已由专用 compute 着色器实现；`f64` 的 `Transpose`、Basis、向量代数等仍主要回落 CPU（或见各模块说明）。
3. **AtPoints 与边界算子**：将 `elem_restriction_at_points`、表面/体积算子组合等写入独立迁移笔记（与 `examples/` 中 ex2 类路径对应）。
4. ~~**Operator 组合示例**~~：**§4.5.1** 与 `examples/mass_operator.rs`、`composite_operator*.rs` 已覆盖典型迁移路径；多子域编排留在应用层。
5. **整数与索引约定**：评估将「尺寸」与「restriction 偏移」统一为与 libCEED 更接近的整型策略；已提供 **`i64` → `i32` 的 restriction 工厂**（`*_ceed_int_*`）与 **`QFUNCTION_INTERIOR_GALLERY_NAMES`** 名称表降低迁移成本，全局 `usize` 与 `CeedInt` 完全统一仍待评估。

以上不改变当前 trait 语义；落地时逐项更新本文件 §4–§8 与 `readme.md`。

### 8.2 迁移笔记：AtPoints、复合算子、整型（对照 libCEED 示例）

**AtPoints 与边界/体积**

| libCEED | Reed |
|---|---|
| `CeedElemRestrictionCreateAtPoints` | `Reed::elem_restriction_at_points(nelem, npoints_per_elem, ncomp, compstride, lsize, offsets)` |

语义与 **offset 型** `elem_restriction` 相同，仅将「每单元局部点数」参数命名为 `npoints_per_elem` 以贴近上游命名。积分点布局仍由 `Basis` / 算子字段决定；表面离散、体积–边界耦合可参照 `examples/ex2_surface.rs` 与 libCEED surface 类示例对照迁移。

**示例对照速查（CPU）**

| libCEED 示例语义 | Reed 示例 |
|---|---|
| ex1-volume（体积积分 / 质量路径） | `examples/ex1_volume.rs` |
| ex1-volume（AtPoints 命名迁移） | `examples/ex1_volume_at_points.rs` |
| ex2-surface（边界/表面积路径） | `examples/ex2_surface.rs` |
| ex3-volume-combined（质量 + Poisson 组合） | `examples/ex3_volume_combined.rs` |
| Poisson（1D/2D/3D） | `examples/poisson.rs` |
| Poisson（2D AtPoints 命名迁移） | `examples/poisson_at_points.rs` |
| 向量质量 gallery 路径（`Vector2MassApply`） | `examples/vector_mass_operator.rs` |
| 复合算子（代数型） | `examples/composite_operator.rs` |
| 复合算子（借用型 / 同作用域网格对象） | `examples/composite_operator_refs.rs` |

**复合算子（加法型）**

| libCEED | Reed |
|---|---|
| `CeedCompositeOperator` 将多个子算子 `apply` 相加 | `Reed::composite_operator(Vec<Box<dyn OperatorTrait<T>>>)` → `CompositeOperator`：`y = Σ_i A_i x`，`linear_assemble_diagonal` / **`linear_assemble_add_diagonal`** 为各子算子对角线之和（后者累加到已有向量）；**`linear_assemble_symbolic` / `linear_assemble` / `linear_assemble_add` / `linear_assemble_csr_matrix` / `linear_assemble_csr_matrix_add`** 在复合上 **`Err`**。若任一子算子 `requires_field_named_buffers == true`，需走复合算子的 `apply_field_buffers*`（含 `Adjoint`） |

注意：`Box<dyn OperatorTrait<T>>` 在 Rust 中默认带 **`'static`**。由 `OperatorBuilder` 得到的 `CpuOperator<'a, T>` 若含有对网格对象（restriction、basis）的借用，则 **不能** 直接装箱进 `composite_operator`。

**对齐 libCEED 的组合方式**：使用 **`Reed::composite_operator_refs`** → **`CompositeOperatorBorrowed<'a, T>`**，以 `&dyn OperatorTrait<T>` 存储子算子，在同一作用域与网格对象共存即可（见 **`examples/composite_operator_refs.rs`**）。纯代数、无借用的子算子仍可用 **`examples/composite_operator.rs`** 的 `Box<dyn>` 路径。**`CompositeOperator`** 与 **`CompositeOperatorBorrowed`** 对稠密 / CSR 矩阵 **`linear_assemble*`**（含 **`linear_assemble_add`** / **`linear_assemble_csr_matrix_add`**）和 `CeedMatrix` 句柄装配（`linear_assemble_ceed_matrix*`）均返回 **`Err`**；迁移策略是“先探针/尝试，再回退到对子算子逐个 `linear_assemble_add_ceed_matrix` 求和”（示例见 **`examples/composite_operator_refs.rs`**）。

**向量空间一致性**：子算子若实现 **`OperatorTrait::global_vector_len_hint`**（`CpuOperator` 由活动场 restriction 推断全局长度），`CompositeOperator` / `CompositeOperatorBorrowed` 在构造时会校验所有非 `None` 的 hint 一致，避免 libCEED 侧也易犯的「子算子空间不匹配」错误。

**`CeedInt` 与尺寸**

| 概念 | libCEED（C） | Reed（Rust） | 迁移 |
|---|---|---|---|
| 全局长度、单元数、`p`、`q` 等尺寸 | `CeedInt` | `CeedSize`（`usize`） | C → Rust：`try_into()`；Rust → C：`as _`（注意范围） |
| restriction 的 `offsets`、strided 的 stride | `CeedInt *` / `int64_t` 绑定 | `&[CeedInt]` / `[CeedInt; 3]`；或 **`&[i64]` / `[i64; 3]`** 经 `*_ceed_int_*` 工厂收窄为 `CeedInt` | 与 32 位索引缓冲区一一对应时可零拷贝共用；`i64` 路径越界返回 `InvalidArgument` |

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
