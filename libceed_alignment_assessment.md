# Reed 与 libCEED 对齐程度评估（基于当前代码）

本文档在 **`design_mapping.md` 概念对照** 之外，单独给出截至当前仓库实现的 **对齐度评估**，便于迁移 libCEED 示例、做路线图或对外说明。评估依据为 `reed`、`reed-core`、`reed-cpu`、`reed-wgpu` 源码与集成测试；**不绑定** libCEED 某一具体发行版号（上游 API 以 `Ceed*` / `ceed-gallery-list.h` 为准）。

---

## 1. 结论摘要

| 层级 | 与 libCEED 的大致对齐度 | 一句话 |
|------|-------------------------|--------|
| 对象模型与资源路由 | **高** | `Reed<T>` / `Backend`、`/cpu/self`、`/gpu/wgpu`（可选 feature）、CUDA/HIP 占位与 `design_mapping` 一致。 |
| Vector / Restriction（CPU） | **高** | `VectorTrait`、`ElemRestrictionTrait`、offset / strided、`elem_restriction_at_points`、`*_ceed_int_*` 工厂与 libCEED 语义一致并有测试。 |
| Basis（CPU） | **中高** | 张量 H1 Lagrange + simplex H1（至 P3）+ `Interp/Grad/Div/Curl/Weight`；Div/Curl 为 **H1 向量上的微分**，非 Nédélec/RT 单元；部分拓扑为枚举占位。 |
| Basis（WGPU） | **中** | `f32` 上 `Interp`/`Grad`/`Div`/`Curl` 及转置与 CPU 对齐；标量 **`Weight` 转置** 复用 `Interpᵀ` 核；`f64` / 无 runtime 回落 CPU。 |
| QFunction（CPU） | **中高** | Interior **命名 gallery** + `q_function_interior`；**`q_function_exterior` + `QFunctionCategory`** 提供与 libCEED exterior 注册 **概念对齐的元数据**（`apply` 仍与 interior 闭包同路径）；**无** `CeedQFunction` 级 C 句柄、**无** 自动面专用积分点编排。 |
| Operator（CPU） | **中高** | `OperatorBuilder` + `CpuOperator`：`apply`/`apply_add`、`apply_with_transpose(Adjoint)`（离散伴随 v1，含可选 `apply_operator_transpose_with_primal` 扩展）、**多 active 场** `apply_field_buffers*`、**被动/`None` 槽在命名 `Adjoint` 中不要求出现在 `outputs`**；**稠密** `linear_assemble*` / **`linear_assemble_add`**、**自建 CSR** `linear_assemble_csr_matrix` / **`linear_assemble_csr_matrix_add`**、**`CeedMatrix` 句柄 set/add 装配**（`linear_assemble_ceed_matrix*`）与 **FDM 替代路径**（小 `n` 稠密逆 + Jacobi 近似逆）在单全局 active 空间上可用；**libCEED 张量 FDM** 仍未实现。依赖 `apply_operator_transpose` 与标量 `Weight` 约定。 |
| Operator（WGPU） | **低** | 整条算子仍在 CPU；WGPU 目前覆盖 restriction / basis 片段，非完整 `CeedOperator` 设备路径。 |
| CompositeOperator | **中高** | 加法型组合与对角装配与 libCEED 子集一致；支持 `apply_field_buffers*`（含 `Adjoint`）对子算子路径求和。 |
| WASM | **中偏低** | `OperatorTrait` / `QFunctionTrait` 等在 `wasm32` 上裁剪（如 `Send + Sync`）；WGPU basis 在 wasm 上无 `BasisTrait` 实现等，与桌面路径不完全对称。 |

**总体**：在 **主机 CPU 离散算子迁移** 方向上，Reed 已覆盖 libCEED 教学/示例中最常见的 **restriction + tensor/simplex H1 basis + interior gallery QFunction + operator apply / transpose（伴随约束内）** 路径，并具备 **Reed 侧 CSR 装配与 SpMV**、**`CeedMatrix` 句柄语义（dense/CSR set+add）** 与 **FDM 替代路径（稠密逆 + Jacobi）**；**设备端完整算子**、**面元专用 quadrature / exterior 全语义**、**与 libCEED 托管式 `CeedMatrix` 的 1:1 后端生命周期**、**libCEED 原生张量 FDM**、**libCEED 全部 gallery / OCCA 后端** 等仍为明显缺口。

---

## 1.1 CPU 对齐发布清单（快速判断）

以下清单仅面向 **`/cpu/self`** 路径，帮助快速回答“CPU 后端能否作为 libCEED 迁移目标发布”：

| 项目 | 状态 | 说明 |
|---|---|---|
| 向量 / restriction / basis 基础路径 | ✅ **已对齐（子集）** | `VectorTrait`、`ElemRestrictionTrait`（offset/strided/at-points）、Lagrange/Simplex basis（P1–P3）与常见 libCEED 示例语义对齐。 |
| QFunction（interior + context） | ✅ **已对齐（子集）** | `QFunctionField`、`apply(ctx,...)`、gallery 名称解析、context 字节布局与 LE 读写路径稳定。 |
| Operator 前向/累加 apply | ✅ **已对齐（子集）** | `apply` / `apply_add`、`check_ready`、多 active 场 `apply_field_buffers*`。 |
| Operator 离散伴随（Adjoint） | ⚠️ **条件对齐** | 依赖 `QFunctionTrait::apply_operator_transpose`；向量 `Weight` 等高级情形仍有边界。 |
| 线装配（Diagonal / AddDiagonal） | ✅ **已对齐（子集）** | `linear_assemble_diagonal` / `linear_assemble_add_diagonal` 已稳定。 |
| 线装配（Dense / CSR，set + add） | ✅ **已对齐（子集）** | `linear_assemble_symbolic` / `linear_assemble` / `linear_assemble_add` 与 `linear_assemble_csr_matrix` / `_add`；测试覆盖 Mass/Poisson。 |
| FDM inverse 形状 (`CeedOperatorCreateFDMElementInverse`) | ⚠️ **API 对齐，实现替代** | 采用小 `n` 全局稠密逆（`CpuFdmDenseInverseOperator`）+ Jacobi 近似逆（`CpuFdmJacobiInverseOperator`），非 libCEED 原生 tensor-FDM。 |
| `CeedMatrix` 对象级 1:1 语义 | ⚠️ **部分对齐** | 已有 `CeedMatrix` 句柄（dense/CSR + symbolic/numeric 状态）与 `CpuOperator` 的 set/add 装配；仍非 libCEED 后端托管矩阵对象的完整 1:1 模型。 |
| 复合算子（加法） | ✅ **已对齐（子集）** | `CompositeOperator*` 的 apply/diag-add 行为稳定；矩阵装配/FDM 在复合上显式 `Err`。 |

**发布建议（CPU）**：若目标是“迁移主机离散算子与常见示例工作流”，可按 **中高对齐** 口径发布；若目标是“libCEED 全 API 逐项等价”，则仍需补 `CeedMatrix` 后端托管语义与原生 tensor-FDM。

---

## 2. 分维度说明

### 2.1 顶层与后端

- **对齐**：资源字符串、`Reed::init`、`Backend` trait、`/cpu/self`（及文档中的 `/cpu/self/ref`）、可选 **`wgpu-backend`** 的 `/gpu/wgpu`。
- **部分对齐**：`/gpu/cuda`、`/gpu/hip` 可解析与报告，**无执行实现**（占位）。
- **差异**：无 libCEED 式 OCCA / 多 vendor 运行时枚举；后端矩阵由 Rust feature + 资源串表达。

### 2.2 类型与枚举

- **对齐**：`EvalMode`（含 `Weight`）、`QuadMode`、`TransposeMode`、`ElemTopology` 等与 libCEED 概念对应；`ElemTopology` 文档明确 **Pyramid/Prism** 等为占位。
- **差异**：Reed 明确提供 `CeedInt`（`i32`）与 `CeedSize`（`usize`）双别名；与 libCEED 单一整型策略仍非 1:1，但通过 `*_ceed_int_*` 工厂覆盖常见绑定桥接。

### 2.3 `ElemRestriction`

- **对齐（CPU）**：`NoTranspose` / `Transpose` 与 gather/scatter 语义；strided；`elem_restriction_at_points` 与 offset 实现一致（集成测覆盖）。
- **对齐（WGPU）**：offset / strided 的 GPU 路径；`f64` gather 的 **位复制** 与 `f64` transpose 仍 CPU 等（与 `design_mapping` §8 一致）。
- **风险点**：与 libCEED 示例混用 `int64` 缓冲区时，仍需通过 **`elem_restriction_ceed_int_*`** 等工厂显式收窄到 `CeedInt`。

### 2.4 `Basis`

**CPU（`LagrangeBasis` / `SimplexBasis`）**

- **高对齐**：`Interp` / `Grad` 及转置；Gauss / GaussLobatto；`q_weights` / `q_ref`。
- **中高对齐**：`Div` / `Curl`（含转置与离散伴随恒等式类测试）；语义为 **H1 向量笛卡尔分量上的算子**，**非** libCEED 中独立 H(div)/H(curl) 元的 Nédélec/RT 基。
- **中高对齐**：**标量** `EvalMode::Weight` 的 **转置** 与 `Interp` 转置同构（basis + 算子伴随路径已接）；**向量 `Weight`** 仍不支持。
- **中**：Simplex 线/三角/四面体 **P1–P3**；张量积 Lagrange 覆盖 libCEED 常见示例维数组合。

**WGPU（`WgpuBasis`）**

- **中**：`f32` 上 `Interp`/`Grad`/`Div`/`Curl`（含转置）与 CPU 对齐（集成测）；**标量 `Weight`+transpose** 走 `Interpᵀ` 核；否则回落 `LagrangeBasis` CPU。
- **限制**：`wasm32` 上 **`BasisTrait` 未为 `WgpuBasis` 实现**（见 `crates/wgpu/src/basis.rs` 注释），与 native 不对称。

### 2.5 `QFunction`

- **高对齐（能力子集）**：`QFunctionField`、`apply(ctx,…)`、`context_byte_len`；`QFunctionContext` 命名字段与 LE 读写；gallery **`q_function_by_name`** 与 `QFUNCTION_INTERIOR_GALLERY_NAMES` 同步自检。
- **中（B）**：**interior / exterior 闭包**：`q_function_interior` / **`q_function_exterior`** + **`QFunctionTrait::q_function_category`**（**`QFunctionCategory`**）；与 libCEED interior/exterior **注册分类** 对齐；**执行路径相同**，无独立面 quadrature 或句柄类型。
- **中**：命名 gallery 均为 **interior** 语义；**无** `CeedQFunctionCreateInteriorByName` 的 C 级独立句柄。
- **中**：`ClosureQFunction` **默认无** `apply_operator_transpose` → 用于 **算子 `Adjoint`** 时需 gallery 或自研 `QFunctionTrait`。
- **Reed 扩展**：`MassApplyInterpTimesWeight`（及 `MassApplyInterpTimesWeightAtPoints`）用于 **被动 `Weight` 槽** 与伴随测试；非 libCEED 注册名。

### 2.6 `Operator` / `CompositeOperator`

**`CpuOperator`（`OperatorBuilder`）**

- **高对齐（子集）**：单 active 输入 + 单 active 输出下的 `apply` / `apply_add`；`check_ready`；非对称 build 的 `active_input_global_len` / `active_output_global_len` 与文档说明。
- **高对齐（子集）**：**离散伴随** — `apply_with_transpose(Adjoint)` 与 `apply_field_buffers_with_transpose(Adjoint)`，在 **qfunction 实现 `apply_operator_transpose`**（或覆写 `apply_operator_transpose_with_primal`）且满足 **单缓冲或命名场映射**、**basis/Weight 约定** 时工作。
- **明确约束（相对 libCEED 一般性）**：
  - 非线性或 **依赖 active 场前向值** 的 qp 核伴随 **不在 v1 范围**（与 `design_mapping` 一致）。
  - **向量场 `EvalMode::Weight`** 在算子伴随中仍不支持。
  - `CompositeOperator*` 已支持 `apply_field_buffers*`（含 `Adjoint`）对子算子路径求和；若子算子需要命名缓冲，单缓冲 `apply*` 路径会返回提示错误并引导改用命名缓冲接口。
- **已修正行为（与 libCEED 文档习惯对齐）**：命名 **`Adjoint` 的 `outputs` 仅需各 active 输入场**；**被动 / `None` 输入槽** 不要求出现在 `outputs`（实现与 `reed_core::OperatorTrait` 文档一致）。
- **Reed 扩展（内存管理）**：**`CpuOperator::dense_linear_assembly_n`** / **`dense_linear_assembly_numeric_ready`** 查询稠密槽状态；**`clear_dense_linear_assembly`** 释放稠密装配槽（**不影响** `apply` / CSR 装配；集成测 **`test_cpu_operator_libceed_dense_linear_assemble_and_fdm_stub`**；**`reed-cpu`** 探针单测 **`dense_linear_assembly_probes_track_symbolic_numeric_and_clear`**）。
- **部分对齐（稠密 + 自建 CSR + `CeedMatrix` 句柄 + FDM API）**：**`CpuOperator::linear_assemble_symbolic` / `linear_assemble`** 写入 **列主序稠密 `n×n`**（`Mutex` 缓冲，`n` 次 `apply`）；**`linear_assemble_add`** 在已有槽上 **累加列**（**`CeedOperatorLinearAssembleAdd`**）；**非线性** `apply` 下 **不保证** 为全局 Jacobian。**`csr_sparsity_from_offset_restriction`** / **`assembled_csr_pattern`** / **`linear_assemble_csr_matrix`** / **`linear_assemble_csr_matrix_add`** / **`CsrMatrix::mul_vec`**：与 libCEED **稀疏拓扑 + 数值（含累加）** 概念对齐。并已提供 **`CeedMatrix` 句柄**（dense/CSR，symbolic/numeric 状态）与 **`linear_assemble_ceed_matrix` / `linear_assemble_add_ceed_matrix`**。**`linear_assemble_add_diagonal`**：对齐 **`CeedOperatorLinearAssembleAddDiagonal`**（`CpuOperator` / **`CompositeOperator*`** / **`CpuFdmDenseInverseOperator`**）。**`operator_create_fdm_element_inverse`**：小 `n` 下 **`Ok(CpuFdmDenseInverseOperator)`**（创建时在本地缓冲按 `n` 次前向 `apply` 组装规范 Jacobian \(A\)，**不读取也不改写** 稠密槽；全局稠密 \(A^{-1}\)，**非** libCEED 张量 FDM）；另有 **`operator_create_fdm_element_inverse_jacobi`**（`CpuFdmJacobiInverseOperator`）提供结构化近似逆。**`operator_supports_assemble`**：`LinearCsrNumeric` 与稠密线装同为 **`active_global_dof_len` 有定义**（**set** 与 **add** 共用）；**复合算子**对 **`LinearCsrNumeric`** 与 **`FdmElementInverse`** 恒 **`false`**（对应 **`linear_assemble_csr_matrix`** / **`linear_assemble_csr_matrix_add`** / **`operator_create_fdm_element_inverse`** 为 **`Err`**）；`FdmElementInverse` 在 `n ≤ FDM_DENSE_MAX_N` 时于 **`CpuOperator`** 为 `true`。稠密 **`linear_assemble_symbolic` / `linear_assemble` / `linear_assemble_add`** 在复合上 **`Err`**。集成测：`test_cpu_operator_ceed_matrix_handle_and_jacobi_inverse_paths`、`test_cpu_operator_libceed_dense_linear_assemble_and_fdm_stub`、`test_mass_csr_assembly_matches_dense_columns`、`test_poisson_1d_csr_assembly_matches_dense_columns`（含 matvec 与 CSR **`add`**）、复合 Mass 中的 FDM / `supports`。

**`CompositeOperator`**

- **中**：加法、`apply_add*`、`linear_assemble_diagonal`、单缓冲 **`Adjoint` 为子算子伴随之和**；与 libCEED 组合模式的部分子集一致。

### 2.7 WGPU 与「整条算子在设备上」

- **低对齐**：无 **`CeedOperator` 级** WGPU apply（qfunction 仍在 CPU，算子未端到端 GPU）。
- **中**：restriction / basis 片段与 `GpuRuntime`、context **同步到 buffer** 等已存在；与 libCEED device 驻留 qdata/context 的 **完整故事** 仍有差距。
- **进展（qp 核）**：`reed_wgpu::GpuRuntime` 提供 **`MassApply` 标量 `f32`** 在积分点上的 **compute 前向 / 转置累加**（`dispatch_mass_apply_qp_*` 已有缓冲；**`mass_apply_qp_f32_host`** / **`mass_apply_qp_transpose_accumulate_f32_host`** 主机切片往返），与 CPU gallery 语义一致（`reed_wgpu` 单测 + `test_wgpu_gpu_runtime_mass_apply_qp_host_bridge`）；**尚未**自动并入 `CpuOperator`。
- **混合算子路径（已测）**：`CpuOperator` + `/gpu/wgpu` 下 `WgpuVector` / `WgpuElemRestriction` / `WgpuBasis` 与 `/cpu/self` 全 CPU 对象在 **`apply`** 及 **`apply_with_transpose`（`Forward` / 对称 `MassApply` 的 `Adjoint`）** 上 **数值一致**（`tests/integration.rs`：`test_wgpu_hybrid_mass_operator_apply_matches_cpu`、`test_wgpu_hybrid_mass_operator_transpose_matches_cpu`，feature `wgpu-backend`）。

### 2.8 WASM

- **Operator / QFunction**：`reed_core` 在 `wasm32` 上 **弱化** `Send + Sync` 等约束；与 native `OperatorTrait` 非同一 cfg 块。
- **WGPU Basis**：见 2.4，`WgpuBasis` 在 wasm 上 **无** `BasisTrait`。
- **对齐策略**：以 **主机 + 可选 wgpu** 为主；与 libCEED wasm 示例同构时需对照 **附录 A** 能力矩阵（向量/restriction 仍可走 WGPU，张量 H1 Lagrange 基在 wasm 上走 CPU）。

---

## 3. Gallery 名称覆盖（相对 `ceed-gallery-list.h`）

- **实现源**：`crates/cpu/src/lib.rs` 中 **`QFUNCTION_LIBCEED_MAIN_GALLERY_NAMES`**（与上游 `gallery/ceed-gallery-list.h` **18** 个 interior 注册 **顺序与名称** 对齐）、**`QFUNCTION_INTERIOR_GALLERY_NAMES`** 与 `q_function_by_name`（当前 **31** 个扩展表项，另含 `IdentityScalar`/`ScaleScalar` 别名、`Vec2Dot`/`Vec3Dot`、`Vector2*`、`MassApplyInterpTimesWeight*`、AtPoints 别名等）。
- **对齐度**：与 libCEED **体积** interior 示例常用核 **高度重叠**；**不声称**与上游 header 中每一条注册名 1:1 完备（上游会随版本增减）。
- **迁移注意**（已在 `design_mapping` 强调，此处列为风险）：
  - **`Vector3Poisson2DApply`**：Reed 侧 `qdata` **4 分量/点** 与部分 libCEED 注册描述 **3 分量** 的打包差异 — 迁移须核对布局。

---

## 4. 测试作为对齐证据（当前仓库）

下列测试类别支撑上表判断（非穷举）：

- **Restriction**：`elem_restriction` vs `elem_restriction_at_points`、strided、`*_ceed_int_*`（`tests/integration.rs` 等）。
- **Basis**：Lagrange / Simplex 恒等式、div/curl 伴随、`Weight` 转置与 `Interp` 转置一致性（`reed-cpu` unit + `reed` integration）。
- **Operator**：Mass / Poisson 对称伴随、`MassApplyInterpTimesWeight` 单缓冲与 **命名缓冲 `Adjoint`**、多 active 场 `apply_field_buffers`、`CompositeOperator`（`tests/integration.rs` + `reed-cpu`）；**稠密 `LinearAssemble*` / `linear_assemble_add`**、**CSR `linear_assemble_csr_matrix_add`**、**`dense_linear_assembly_n` / `dense_linear_assembly_numeric_ready` / `clear_dense_linear_assembly`**（集成测 + **`reed-cpu`** **`dense_linear_assembly_probes_track_symbolic_numeric_and_clear`**、**`clear_dense_linear_assembly_idempotent_without_slot`**）。
- **QFunction**：**exterior 闭包分类**（`test_qfunction_exterior_closure_reports_exterior_category`）；gallery **interior 分类**（`reed-cpu`：`gallery_mass_apply_is_interior_category`）。
- **WGPU**：`wgpu-backend` feature 下 basis / elem_restriction 与 CPU 对齐（`tests/integration.rs`）；`reed-wgpu` 内 basis 单元测（需 adapter 时跳过）。

---

## 5. 对齐度分级（建议用法）

| 等级 | 含义 | 典型用途 |
|------|------|----------|
| **A** | 语义与路径与 libCEED 常见示例 **等价或可机械迁移** | 1D/2D/3D Poisson/Mass + tensor Lagrange + offset restriction |
| **B** | 功能有，但 **API 形状或类型细节不同** 或 **仅 CPU** | 多向量 `apply_field_buffers`、`*_ceed_int_*`、闭包 QFunction、**稠密** `LinearAssemble*` / **`linear_assemble_add`**、**`OperatorTrait::linear_assemble_csr_matrix`** / **`linear_assemble_csr_matrix_add`**、**自建 CSR**（非 libCEED 持有的 `CeedMatrix` 句柄） |
| **C** | **部分** 与 libCEED 同名或同概念，**语义子集或扩展** | `Vector2*` gallery、Reed 扩展 `MassApplyInterpTimesWeight` |
| **D** | **未实现** 或 **仅占位** | CUDA/HIP 执行、整条 Operator on GPU、**libCEED 托管 `CeedMatrix` / 原生张量 FDM**、面元专用 exterior **全语义**（Reed 仅有 **元数据** `QFunctionCategory::Exterior`） |

可将迁移中的每个 libCEED 调用映射到上表某一格，再决定是否需要上层胶水代码。

---

## 6. 建议的后续对齐优先级（与 `design_mapping` §8.1 一致，略作压缩）

1. **WGPU 算子端到端**：qp 数据与 `QFunctionContext` 设备驻留、与 restriction/basis 管线衔接（最大缺口）。
2. **整型与 `CeedInt` 策略**：是否在公共 API 层固定 `i32` 索引 + `usize` 尺寸文档化，减少与 libCEED C 示例的摩擦（**当前落盘约定见附录 B**）。
3. **WASM 能力矩阵**：为 `wasm32` 单独维护「支持 / 不支持」表，与 libCEED wasm 路径期望对齐（**见附录 A**）。
4. **Gallery 缺口**：按上游 `ceed-gallery-list.h` 做 diff，补缺或显式标注「不计划支持」。

---

## 7. 与 `design_mapping.md` 的关系

- **`design_mapping.md`**：长期维护的 **概念 ↔ API 映射表** 与约定说明。
- **本文档**：基于 **当前实现快照** 的 **对齐度与风险** 评估，可随大版本或里程碑更新；不必逐行重复映射表。
- **附录 A**：`wasm32` 目标下各 trait / 工厂与 WGPU 路径的 **能力矩阵**（与 `reed_core` / `reed_wgpu` 中 `cfg` 一致）。
- **附录 B**：与 libCEED `CeedInt` 互操作的 **整型桥接** 与 Reed 侧 **索引类型** 约定摘要。

若二者冲突，以 **源码与测试** 为准，并应回写修正 `design_mapping.md`。

---

## 附录 A. WASM（`target_arch = "wasm32"`）能力矩阵

依据 `reed_core` 与 `reed_wgpu` 源码中的 `#[cfg(target_arch = "wasm32")]` 拆分整理；**非 wasm** 列表示桌面/原生线程模型下的对照。

| 能力域 | 非 wasm（native） | wasm32 | 说明 |
|--------|-------------------|--------|------|
| `reed_core::Backend` | `Send + Sync` | 无 `Send + Sync` | 浏览器中 `wgpu::Device` 非 `Send + Sync`，故后端工厂 trait 在 wasm 上放宽边界（`reed.rs` / `reed_wgpu` 的 `Backend` 注释）。 |
| `VectorTrait` / `ElemRestrictionTrait` / `BasisTrait`（对象侧） | 多为 `Send + Sync` | 多为无 `Send + Sync` | 与 `reed_core` 中各 trait 的 wasm 变体一致；便于持有 `dyn` 后端对象。 |
| `QFunctionTrait` / `QFunctionClosure` | trait 与闭包要求 `Send + Sync` | 不要求 `Send + Sync` | `qfunction.rs`：wasm 上闭包可为单线程捕获。 |
| `OperatorTrait` | `Send + Sync` | 无 | `operator.rs`：`CpuOperator` 在 wasm 上仍实现完整算子 API（含 `apply_field_buffers` 与伴随相关路径），仅 trait 对象边界不同。 |
| `WgpuBackend::create_vector` | `WgpuVector` | `WgpuVector` | wasm 上仍走 WebGPU 缓冲路径（非纯 CPU `CpuVector`）。 |
| `WgpuBackend::create_elem_restriction` / `create_strided_elem_restriction` | `WgpuElemRestriction` | 同上 | GPU gather/scatter 路径在 wasm 上仍可用（受 WebGPU 实现与浏览器策略约束）。 |
| `WgpuBackend::create_basis_tensor_h1_lagrange` | `WgpuBasis`（`BasisTrait`） | **CPU `LagrangeBasis`（`cpu_backend` 工厂）** | `WgpuBasis` 的 `BasisTrait` 实现 **仅** `#[cfg(not(target_arch = "wasm32"))]`（`basis.rs`）；wasm 上 `WgpuBackend` 委托 `cpu_backend` 创建张量 Lagrange。 |
| `WgpuBasis`：`EvalMode::Weight` 转置 GPU 捷径 | `f32` 且标量时可复用 interpᵀ 核 | **不适用** | 依赖 `BasisTrait for WgpuBasis`，wasm 上不存在该 impl。 |
| WGPU 相关集成测 / basis 单测 | 默认运行 | 多处 `#[cfg(not(target_arch = "wasm32"))]` 跳过 | 例如 `basis.rs` 末尾部分测试仅在 native 编译。 |

**迁移提示**：在 wasm 上若资源选 `/gpu/wgpu`，张量 H1 Lagrange **基求值在 CPU**，restriction/vector 仍可走 WGPU；与 libCEED「全对象同后端」的想象不完全一致，上层若需统一性能模型应显式分支或固定用 CPU 后端。

---

## 附录 B. 整型与 `CeedInt` 桥接（当前约定）

| 概念 | Reed 运行时类型 | 与 libCEED 的桥 |
|------|-----------------|----------------|
| 全局长度、单元数、`elemsize`、`ncomp`、`lsize`、多项式阶等 **尺寸** | `CeedSize`（`usize`） | C 示例里常为 `CeedInt`；从 Rust 调用 Reed 时使用 `CeedSize` / `usize`。 |
| restriction **偏移**、**strided 三整数** | `&[CeedInt]` / `[CeedInt; 3]`（`CeedInt = i32`） | 与 32 位索引的 GPU / WGSL 路径一致；超大网格需自行确认不溢出 `CeedInt`。 |
| 自 **i64 / `int64_t` 绑定** 迁入 | `Reed::elem_restriction_ceed_int_offsets`、`elem_restriction_at_points_ceed_int_offsets`、`strided_elem_restriction_ceed_int_strides` | `reed.rs` 内 `ceed_int_*_to_i32`：每项必须落入 `i32`，否则 `InvalidArgument`。 |

**未决（与 §6.2 一致）**：是否在文档层统一写死「Reed 公共 API 以 `CeedSize` + `CeedInt` 为规范，与 libCEED 64 位 `CeedInt` 的差异由上述 `*_ceed_int_*` 入口吸收」，并在更多工厂上对称提供 `*_ceed_int_*` 变体，仍属产品化决策。

---

## 8. 修订历史

| 日期 | 说明 |
|------|------|
| 2026-04-21 | 首版：综合当前 `reed` 工作区算子伴随、`Weight`、WGPU basis、命名缓冲 `Adjoint` 与 gallery 状态撰写。 |
| 2026-04-21 | 增补附录 A（WASM 能力矩阵）、附录 B（`CeedInt` / 尺寸约定）；§6 与 §7 交叉引用。 |
| 2026-04-21 | §2.7：记录 WGPU 混合 `CpuOperator` 前向与 CPU 栈对齐的集成测；`design_mapping` §4.5 同步。 |
| 2026-04-21 | 混合算子：增补 `apply_with_transpose`（Forward / Adjoint）与 CPU 交叉验证的集成测；`design_mapping` §4.5 同步。 |
| 2026-04-21 | WGPU：`GpuRuntime` 增加 `MassApply` 标量 `f32` qp 前向/转置 compute 与单测；§2.7、`design_mapping` §8 / §8.1 同步。 |
| 2026-04-21 | WGPU：补充 `mass_apply_qp_*_host` 主机往返 API、集成测；文档同步。 |
| 2026-04-21 | CPU：`QFUNCTION_LIBCEED_MAIN_GALLERY_NAMES` 对齐 libCEED main `ceed-gallery-list.h`；`IdentityScalar`/`ScaleScalar` 别名；集成测与 `design_mapping` §5 / §8 同步。 |
| 2026-04-21 | QFunction：`QFunctionCategory` / `q_function_exterior`；Operator：`OperatorAssembleKind` 与 `LinearAssemble*` / FDM trait 占位；`design_mapping` §4.4–4.5、集成测与 §2.5–2.6、分级表 **D** 行更新。 |
| 2026-04-21 | Operator：`CpuOperator::operator_supports_assemble(Diagonal)` 与 `active_global_dof_len` 一致；复合算子 **supports 逻辑与**、顺序调用 `linear_assemble_*`、**FDM 显式拒绝**；`design_mapping` / 集成测与 `reed-cpu` 单测更新。 |
| 2026-04-21 | `CpuOperator`：**稠密** `linear_assemble_symbolic` / `linear_assemble`（`Mutex` 缓冲、列主序）、`assembled_linear_matrix_col_major`；复合算子对 `linear_assemble*` **`Err`**；`OperatorAssembleKind` 文档与 `design_mapping` / 本表 **B/D** 行更新。 |
| 2026-04-21 | CSR：`reed_core` 的 `CsrPattern` / `CsrMatrix`、`csr_sparsity_from_offset_lnodes`；**`ElemRestrictionTrait::assembled_csr_pattern`**（`CpuElemRestriction` offset / `ncomp==1`）；**`CpuOperator::linear_assemble_csr_matrix`**；集成测 `test_mass_csr_assembly_matches_dense_columns`；`design_mapping` / 分级 **B** 行更新。 |
| 2026-04-21 | CSR：**`csr_sparsity_from_offset_restriction`**（`ncomp`×`compstride` 与 offset gather 一致）；**`CsrMatrix::mul_vec` / `mul_vec_add`**；集成测中 **CSR matvec vs `apply`**；`reed-cpu` 单测 `assembled_csr_pattern_ncomp2_one_segment`。 |
| 2026-04-21 | CPU：**`operator_create_fdm_element_inverse`** 小 `n` 稠密逆（`CpuFdmDenseInverseOperator`、`FDM_DENSE_MAX_N`）；复合算子 **`FdmElementInverse` supports 恒 false**；`design_mapping` / 集成测更新。 |
| 2026-04-21 | **`OperatorTrait::linear_assemble_csr_matrix`**（默认 `Err`；`CpuOperator` 委托）；复合算子显式 **`Err`**；FDM 创建与 CSR trait 路径集成测同步。 |
| 2026-04-21 | **`OperatorAssembleKind::LinearCsrNumeric`** + **`operator_supports_assemble`**（`CpuOperator` / 复合）；`design_mapping` / 集成测更新。 |
| 2026-04-21 | **`CpuOperator::linear_assemble_csr_from_elem_restriction`**；集成测 **`test_poisson_1d_csr_assembly_matches_dense_columns`**（Poisson1D + CSR/dense/`mul_vec`）。 |
| 2026-04-21 | **`OperatorTrait::linear_assemble_add_diagonal`**（`CeedOperatorLinearAssembleAddDiagonal`）；`CpuOperator` / **`CompositeOperator*`** / **`CpuFdmDenseInverseOperator`**；`design_mapping` / 集成测与 `reed-cpu` 单测。 |
| 2026-04-21 | **`OperatorTrait::linear_assemble_add`** / **`linear_assemble_csr_matrix_add`**（`CeedOperatorLinearAssembleAdd` 稠密与 CSR）；`CpuOperator` 实现；**`CompositeOperator*`** 显式 **`Err`**；集成测与 **`composite_linear_assemble_add_errors`**；`design_mapping` §4.5.1 / §8.2、§2.6 与分级 **B** 行更新。 |
| 2026-04-21 | 集成测：**`test_poisson_1d_csr_assembly_matches_dense_columns`** 增加 CSR **`linear_assemble_csr_matrix_add`**（与 Mass 同构）；**`composite_linear_assemble_csr_matrix_add_errors`**；`design_mapping` §4.5.1 **`LinearAssembleAdd`** 行、§4 测试列举同步。 |
| 2026-04-21 | **`CompositeOperatorBorrowed`**：`composite_borrowed_linear_assemble_add_errors`、**`composite_borrowed_linear_assemble_csr_matrix_add_errors`**（与 **`CompositeOperator`** 对称，锁定错误前缀与 API 名）。 |
| 2026-04-21 | **`CpuOperator::clear_dense_linear_assembly`**：释放稠密 `Mutex` 装配缓冲；`assembled_linear_matrix_col_major` 在清后 **`None`**；集成测与 **`assembly_dense`** 模块注释、`design_mapping` §4.5 同步。 |
| 2026-04-21 | **`clear_dense_linear_assembly_idempotent_without_slot`**（`reed-cpu`）；`design_mapping` §4.5 / §4.5.1 表（与 **`CeedMatrixDestroy`** 等 **概念** 对照）、`libceed_alignment` §4 测试列举。 |
| 2026-04-21 | **`CpuOperator::dense_linear_assembly_n`** / **`dense_linear_assembly_numeric_ready`**；**`dense_linear_assembly_probes_track_symbolic_numeric_and_clear`**；`clear_dense_linear_assembly` 文档交叉引用；`design_mapping` / §2.6 同步。 |
| 2026-04-21 | **`OperatorTrait::linear_assemble_symbolic`** 文档：**`CpuOperator`** 每次调用 **替换** 稠密槽并重置数值完成标志；**`second_linear_assemble_symbolic_resets_numeric_state`**；`reed_core` / `reed_cpu` crate 文档与 **`design_mapping`** §4.5 同步。 |
| 2026-04-21 | **`operator_create_fdm_element_inverse`**：改为创建时在本地缓冲组装规范 \(A\)（不读取/改写稠密槽，避免受到 `linear_assemble_add` 累加槽影响）；集成测 `test_cpu_operator_libceed_dense_linear_assemble_and_fdm_stub` 增加 “after add 仍回到 \(A^{-1}\)” 与“FDM 创建不改 dense 槽”校验。 |
| 2026-04-21 | `reed-cpu` 单测 **`fdm_creation_does_not_mutate_dense_slot`**：在可逆 `MassApply` 上锁定 “FDM 创建后 dense 槽保持 `linear_assemble_add` 累加态” 的无副作用语义。 |
