# Reed 离散化库 — AI Agent 编码规范文档

## 1. 项目概述

**项目名称**：`reed`（Rust Efficient Extensible Discretization）

**设计目标**：用纯 Rust 实现一个类 [libCEED](https://github.com/CEED/libCEED) 的有限元离散化库，提供高性能、内存安全、后端可扩展的有限元算符组合框架。

**核心设计哲学**：
- 将 PDE 的离散化分解为若干**正交的数学对象**（向量、基函数、自由度限制、积分点算符、算符组合）。
- 每个对象通过 **Rust trait** 描述，通过 **`dyn Trait`** 或泛型实现后端分发。
- 充分利用 Rust 的所有权系统保证**内存安全与无悬垂指针**。
- 零成本抽象：内层热路径（张量积）使用泛型单态化；外层 API 使用 `dyn Trait`。

---

## 2. 工作区布局（Workspace）

```
reed/
├── Cargo.toml                   # workspace 根配置
├── crates/
│   ├── reed-core/               # 核心抽象 trait 与公共类型
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── scalar.rs        # Scalar trait（f32/f64 泛型）
│   │       ├── error.rs         # ReedError、ReedResult
│   │       ├── enums.rs         # EvalMode、ElemTopology、QuadMode、TransposeMode、NormType、MemType
│   │       ├── vector.rs        # VectorTrait
│   │       ├── basis.rs         # BasisTrait
│   │       ├── elem_restriction.rs  # ElemRestrictionTrait
│   │       ├── qfunction.rs     # QFunctionTrait、QFunctionInputs、QFunctionOutputs
│   │       ├── operator.rs      # OperatorTrait、OperatorField
│   │       └── reed.rs          # Reed 顶层 context
│   ├── reed-cpu/                # CPU 后端（默认实现）
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── vector.rs        # CpuVector<T>
│   │       ├── basis_lagrange.rs    # LagrangeBasis<T>（张量积高斯积分）
│   │       ├── elem_restriction.rs  # CpuElemRestriction<T>
│   │       └── operator.rs          # CpuOperator<T>
│   └── reed-cuda/               # CUDA GPU 后端（feature = "cuda"，可选）
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           └── vector.rs        # CudaVector<T>
├── examples/
│   ├── mass_operator.rs         # 质量算符端到端示例
│   └── poisson_1d.rs            # 1D Poisson 方程示例
└── tests/
    └── integration.rs
```

---

## 3. 核心类型规范

### 3.1 Workspace Cargo.toml

```toml name=Cargo.toml
[workspace]
members = [
    "crates/reed-core",
    "crates/reed-cpu",
]
resolver = "2"

[workspace.dependencies]
thiserror = "2"
num-traits = "0.2"
rayon     = "1"
```

---

### 3.2 `reed-core/src/scalar.rs`

```rust name=crates/reed-core/src/scalar.rs
use num_traits::{Float, NumAssign};
use std::fmt::Debug;

/// 标量类型约束，支持 f32 / f64
pub trait Scalar:
    Float + NumAssign + Send + Sync + Copy + Debug + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const EPSILON: Self;
}

impl Scalar for f32 {
    const ZERO: Self    = 0.0_f32;
    const ONE: Self     = 1.0_f32;
    const EPSILON: Self = f32::EPSILON;
}

impl Scalar for f64 {
    const ZERO: Self    = 0.0_f64;
    const ONE: Self     = 1.0_f64;
    const EPSILON: Self = f64::EPSILON;
}
```

---

### 3.3 `reed-core/src/error.rs`

```rust name=crates/reed-core/src/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReedError {
    #[error("Vector error: {0}")]
    Vector(String),

    #[error("Basis error: {0}")]
    Basis(String),

    #[error("ElemRestriction error: {0}")]
    ElemRestriction(String),

    #[error("QFunction error: {0}")]
    QFunction(String),

    #[error("Operator error: {0}")]
    Operator(String),

    #[error("Backend not supported: {0}")]
    BackendNotSupported(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

pub type ReedResult<T> = Result<T, ReedError>;
```

---

### 3.4 `reed-core/src/enums.rs`

```rust name=crates/reed-core/src/enums.rs
/// 内存位置类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemType {
    Host,
    Device,
}

/// 基函数求值模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    None,
    Interp,   // 插值 B·u
    Grad,     // 梯度 ∇B·u
    Div,      // 散度 ∇·B·u
    Curl,     // 旋度 ∇×B·u
    Weight,   // 积分权重 w_q
}

/// 积分点分布类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadMode {
    Gauss,
    GaussLobatto,
}

/// 转置模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeMode {
    NoTranspose,
    Transpose,
}

/// 向量范数类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    One,   // L1 范数
    Two,   // L2 范数
    Max,   // L∞ 范数
}

/// 单元拓扑类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemTopology {
    Line,
    Triangle,
    Quad,
    Tet,
    Pyramid,
    Prism,
    Hex,
}
```

---

### 3.5 `reed-core/src/vector.rs`

```rust name=crates/reed-core/src/vector.rs
use crate::{error::ReedResult, scalar::Scalar, enums::NormType};

/// 抽象数值向量 trait
/// - 实现者：CpuVector<T>、CudaVector<T>
pub trait VectorTrait<T: Scalar>: Send + Sync {
    /// 向量长度
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool { self.len() == 0 }

    /// 将数据复制进向量（来自主机切片）
    fn copy_from_slice(&mut self, data: &[T]) -> ReedResult<()>;

    /// 将数据复制出向量（到主机切片）
    fn copy_to_slice(&self, data: &mut [T]) -> ReedResult<()>;

    /// 将所有元素设为常量
    fn set_value(&mut self, val: T) -> ReedResult<()>;

    /// AXPY: self = alpha * x + self
    fn axpy(&mut self, alpha: T, x: &dyn VectorTrait<T>) -> ReedResult<()>;

    /// 向量缩放: self *= alpha
    fn scale(&mut self, alpha: T) -> ReedResult<()>;

    /// 计算向量范数
    fn norm(&self, norm_type: NormType) -> ReedResult<T>;
}
```

---

### 3.6 `reed-core/src/basis.rs`

```rust name=crates/reed-core/src/basis.rs
use crate::{error::ReedResult, scalar::Scalar, enums::EvalMode};

/// 参考单元基函数 trait
/// - 支持 H1、H(div)、H(curl) 等各类有限元空间
pub trait BasisTrait<T: Scalar>: Send + Sync {
    /// 空间拓扑维度
    fn dim(&self) -> usize;

    /// 每单元自由度数量（插值节点数）
    fn num_dof(&self) -> usize;

    /// 每单元积分点数量
    fn num_qpoints(&self) -> usize;

    /// 分量数量（标量场为 1）
    fn num_comp(&self) -> usize;

    /// 应用基函数算符
    ///
    /// - `transpose = false`：正向，u_local[elem,dof] → v_qpt[elem,qpt]
    /// - `transpose = true` ：转置，v_qpt[elem,qpt] → u_local[elem,dof]
    /// - `eval_mode`：指定求值类型（插值/梯度/散度/旋度/权重）
    /// - `num_elem`：批量处理的单元数
    fn apply(
        &self,
        num_elem: usize,
        transpose: bool,
        eval_mode: EvalMode,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<()>;

    /// 积分权重（长度 = num_qpoints()）
    fn q_weights(&self) -> &[T];

    /// 积分点坐标（参考单元，行主序 [nqpts × dim]）
    fn q_ref(&self) -> &[T];
}
```

---

### 3.7 `reed-core/src/elem_restriction.rs`

```rust name=crates/reed-core/src/elem_restriction.rs
use crate::{error::ReedResult, scalar::Scalar, enums::TransposeMode};

/// 自由度限制算符 trait
///
/// 数学含义：
///   E  (NoTranspose): u_local[e,i] = u_global[offsets[e,i]]
///   Eᵀ (Transpose)  : u_global[offsets[e,i]] += u_local[e,i]  （集成/组装）
pub trait ElemRestrictionTrait<T: Scalar>: Send + Sync {
    /// 单元总数
    fn num_elements(&self) -> usize;

    /// 每单元自由度数
    fn num_dof_per_elem(&self) -> usize;

    /// 全局向量自由度数
    fn num_global_dof(&self) -> usize;

    /// 分量数
    fn num_comp(&self) -> usize;

    /// 应用限制算符
    ///
    /// - NoTranspose: u_global → u_local （限制/提取）
    /// - Transpose  : u_local  → u_global（集成/组装，累加）
    fn apply(
        &self,
        t_mode: TransposeMode,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<()>;
}
```

---

### 3.8 `reed-core/src/qfunction.rs`

```rust name=crates/reed-core/src/qfunction.rs
use crate::{error::ReedResult, scalar::Scalar, enums::EvalMode};

/// QFunction 字段描述符
#[derive(Debug, Clone)]
pub struct QFunctionField {
    pub name:      String,
    pub num_comp:  usize,
    pub eval_mode: EvalMode,
}

/// 积分点点态算符 trait
///
/// 数学含义（弱形式体积项）：
///   ⟨v, F(u)⟩ = ∫_Ω v·f₀(u,∇u) + ∇v:f₁(u,∇u)
///
/// `apply` 在所有积分点上同时求值，inputs/outputs 均为
///   扁平切片，布局为 [num_qpoints × num_comp]（行主序）
pub trait QFunctionTrait<T: Scalar>: Send + Sync {
    /// 输入字段描述
    fn inputs(&self) -> &[QFunctionField];

    /// 输出字段描述
    fn outputs(&self) -> &[QFunctionField];

    /// 在 `q` 个积分点上求值
    ///
    /// - `inputs[i]`  长度 = q × inputs()[i].num_comp
    /// - `outputs[i]` 长度 = q × outputs()[i].num_comp
    fn apply(
        &self,
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()>;
}

/// 用户闭包类型别名
pub type QFunctionClosure<T> = dyn Fn(usize, &[&[T]], &mut [&mut [T]]) -> ReedResult<()>
    + Send
    + Sync;
```

---

### 3.9 `reed-core/src/operator.rs`

```rust name=crates/reed-core/src/operator.rs
use crate::{error::ReedResult, scalar::Scalar, vector::VectorTrait};

/// 完整弱形式算符 trait
///
/// 组合 ElemRestriction + Basis + QFunction，实现：
///   v = A(u) = Eᵀ Bᵀ D B E u
/// 其中 D 由 QFunction 确定
pub trait OperatorTrait<T: Scalar>: Send + Sync {
    /// 应用算符：output = A(input)
    fn apply(
        &self,
        input:  &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()>;

    /// 累加应用算符：output += A(input)
    fn apply_add(
        &self,
        input:  &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()>;

    /// 组装线性算符的对角线
    fn linear_assemble_diagonal(
        &self,
        assembled: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()>;
}
```

---

### 3.10 `reed-core/src/reed.rs`（顶层 Context）

```rust name=crates/reed-core/src/reed.rs
use std::sync::Arc;
use crate::{
    error::{ReedError, ReedResult},
    scalar::Scalar,
    vector::VectorTrait,
    basis::BasisTrait,
    elem_restriction::ElemRestrictionTrait,
    qfunction::QFunctionTrait,
    operator::OperatorTrait,
    enums::*,
};

/// 后端工厂 trait（各后端实现此 trait）
pub trait Backend<T: Scalar>: Send + Sync {
    fn resource_name(&self) -> &str;

    fn create_vector(&self, size: usize) -> ReedResult<Box<dyn VectorTrait<T>>>;

    fn create_elem_restriction(
        &self,
        nelem:      usize,
        elemsize:   usize,
        ncomp:      usize,
        compstride: usize,
        lsize:      usize,
        offsets:    &[i32],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>;

    fn create_strided_elem_restriction(
        &self,
        nelem:    usize,
        elemsize: usize,
        ncomp:    usize,
        lsize:    usize,
        strides:  [i32; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>;

    fn create_basis_tensor_h1_lagrange(
        &self,
        dim:    usize,
        ncomp:  usize,
        p:      usize,   // 插值节点数（多项式阶 = p-1）
        q:      usize,   // 积分点数
        qmode:  QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;
}

/// Reed 顶层库上下文
///
/// 对应 libCEED 的 `Ceed`，通过资源字符串选择后端：
///   "/cpu/self"   → CPU 后端
///   "/gpu/cuda"   → CUDA 后端（feature = "cuda"）
pub struct Reed<T: Scalar> {
    backend: Arc<dyn Backend<T>>,
}

impl<T: Scalar> Reed<T> {
    /// 使用资源字符串初始化
    pub fn init(resource: &str) -> ReedResult<Self> {
        let backend: Arc<dyn Backend<T>> = match resource {
            "/cpu/self" | "/cpu/self/ref" => {
                #[cfg(feature = "cpu")]
                { Arc::new(reed_cpu::CpuBackend::<T>::new()) }
                #[cfg(not(feature = "cpu"))]
                { return Err(ReedError::BackendNotSupported(resource.into())); }
            }
            other => return Err(ReedError::BackendNotSupported(other.into())),
        };
        Ok(Self { backend })
    }

    pub fn resource(&self) -> &str {
        self.backend.resource_name()
    }

    // ── Vector 工厂 ──────────────────────────────────────────────────────────

    pub fn vector(&self, n: usize) -> ReedResult<Box<dyn VectorTrait<T>>> {
        self.backend.create_vector(n)
    }

    pub fn vector_from_slice(&self, data: &[T]) -> ReedResult<Box<dyn VectorTrait<T>>> {
        let mut v = self.backend.create_vector(data.len())?;
        v.copy_from_slice(data)?;
        Ok(v)
    }

    // ── ElemRestriction 工厂 ─────────────────────────────────────────────────

    pub fn elem_restriction(
        &self,
        nelem:      usize,
        elemsize:   usize,
        ncomp:      usize,
        compstride: usize,
        lsize:      usize,
        offsets:    &[i32],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        self.backend.create_elem_restriction(nelem, elemsize, ncomp, compstride, lsize, offsets)
    }

    pub fn strided_elem_restriction(
        &self,
        nelem:    usize,
        elemsize: usize,
        ncomp:    usize,
        lsize:    usize,
        strides:  [i32; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        self.backend.create_strided_elem_restriction(nelem, elemsize, ncomp, lsize, strides)
    }

    // ── Basis 工厂 ───────────────────────────────────────────────────────────

    pub fn basis_tensor_h1_lagrange(
        &self,
        dim:   usize,
        ncomp: usize,
        p:     usize,
        q:     usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        self.backend.create_basis_tensor_h1_lagrange(dim, ncomp, p, q, qmode)
    }

    // ── QFunction 工厂 ───────────────────────────────────────────────────────
    // QFunction 由用户直接 impl QFunctionTrait，无需工厂方法。
    // 标准内置算符（质量���Poisson 等）在 reed-cpu/src/gallery/ 中提供。
}
```

---

## 4. CPU 后端实现规范

### 4.1 `reed-cpu/src/vector.rs`

**实现要求**：
- 内部存储：`Vec<T>`
- 实现 `VectorTrait<T>` 全部方法
- `axpy`、`scale`、`norm` 可使用 `rayon` 并行化

### 4.2 `reed-cpu/src/basis_lagrange.rs`

**实现要求**：
- 存储张量积矩阵：`interp: Vec<T>`（Q×P）、`grad: Vec<T>`（Q×P×dim）、`weights: Vec<T>`（Q）
- `apply` 方法实现张量积分解（先在每个维度依次作用），维度循环用 `rayon` 并行
- 提供工厂函数 `gauss_lobatto_nodes`、`gauss_quadrature` 自动计算节点和权重

```rust
// 张量积应用伪代码（1D → nD 的推广）
// interp[q,p] 存储为行主序，q 是积分点索引，p 是 DOF 索引
fn tensor_contract(B: &[T], u: &[T], v: &mut [T], Q: usize, P: usize, transpose: bool) {
    // v[q] = Σ_p B[q,p] * u[p]  (NoTranspose)
    // v[p] = Σ_q B[q,p] * u[q]  (Transpose)
}
```

### 4.3 `reed-cpu/src/elem_restriction.rs`

**实现要求**：
- 存储 `offsets: Vec<i32>`（形状 `[nelem × elemsize]`）
- NoTranspose：`u_local[e*elemsize+i] = u_global[offsets[e*elemsize+i]]`
- Transpose：`u_global[offsets[e*elemsize+i]] += u_local[e*elemsize+i]`（原子加或串行累加）

### 4.4 CPU 内置 QFunction（Gallery）

在 `reed-cpu/src/gallery/` 提供以下标准算符：

| 名称 | 描述 | 输入 | 输出 |
|------|------|------|------|
| `Mass1DBuild` | 构建 1D 质量矩阵积分数据 | `dx[1]`、`weights[1]` | `qdata[1]` |
| `MassApply`   | 应用质量算符 | `u[1]`、`qdata[1]` | `v[1]` |
| `Poisson1DApply` | 应用 1D Poisson 算符 | `du[1]`、`qdata[1]` | `dv[1]` |

每个 Gallery QFunction 实现 `QFunctionTrait<f64>`，可通过 `Reed::q_function_by_name("Mass1DBuild")` 获取。

---

## 5. 用户自定义 QFunction 规范

AI Agent 在生成用户侧代码时，QFunction 应通过闭包或结构体实现：

```rust
// 方式 A：闭包风格（简洁，推荐用于内联算符）
let my_qf = reed.q_function_interior(
    1, // vector length (SIMD width hint)
    Box::new(|q: usize, inputs: &[&[f64]], outputs: &mut [&mut f64]| {
        let u       = inputs[0];
        let weights = inputs[1];
        let v       = &mut outputs[0];
        for i in 0..q {
            v[i] = u[i] * weights[i];
        }
        Ok(())
    }),
)?;

// 方式 B：结构体实现（推荐用于复杂 PDE 算符）
struct MyQFunction;
impl QFunctionTrait<f64> for MyQFunction {
    fn inputs(&self) -> &[QFunctionField] { /* ... */ }
    fn outputs(&self) -> &[QFunctionField] { /* ... */ }
    fn apply(&self, q: usize, inputs: &[&[f64]], outputs: &mut [&mut [f64]]) -> ReedResult<()> {
        // 物理算符实现
        Ok(())
    }
}
```

---

## 6. Operator 组合规范

Operator 通过 **Builder 模式** 链式组合字段，调用顺序：

```rust
let reed = Reed::<f64>::init("/cpu/self")?;

// 1. 创建基础对象
let x_coord  = reed.vector_from_slice(&node_coords)?;
let mut qdata = reed.vector(nelem * nqpts)?;
qdata.set_value(0.0)?;

// 2. 构造限制算符
let r_x = reed.elem_restriction(nelem, 2, 1, 1, nelem + 1, &ind_x)?;
let r_u = reed.elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)?;
let r_q = reed.strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])?;

// 3. 构造基函数
let b_x = reed.basis_tensor_h1_lagrange(1, 1, 2, q, QuadMode::Gauss)?;
let b_u = reed.basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)?;

// 4. 构建 QData（几何信息预计算）
reed.operator_builder()
    .qfunction(reed.q_function_by_name("Mass1DBuild")?)
    .field("dx",      &*r_x, &*b_x, FieldVector::Active)
    .field("weights", FieldRestriction::None, &*b_x, FieldVector::None)
    .field("qdata",   &*r_q, FieldBasis::None, FieldVector::Active)
    .build()?
    .apply(&x_coord, &mut qdata)?;

// 5. 构建最终算符
let op_mass = reed.operator_builder()
    .qfunction(reed.q_function_by_name("MassApply")?)
    .field("u",     &*r_u, &*b_u, FieldVector::Active)
    .field("qdata", &*r_q, FieldBasis::None, FieldVector::Passive(qdata))
    .field("v",     &*r_u, &*b_u, FieldVector::Active)
    .build()?;

// 6. 执行算符
let u = reed.vector_from_slice(&vec![1.0_f64; ndofs])?;
let mut v = reed.vector(ndofs)?;
v.set_value(0.0)?;
op_mass.apply(&*u, &mut *v)?;
```

**字段向量类型（`FieldVector` 枚举）**：

```rust
pub enum FieldVector<'a, T: Scalar> {
    Active,                          // 输入/输出主向量
    Passive(&'a dyn VectorTrait<T>), // 预计算数据（只读）
    None,                            // 积分权重字段（无向量）
}
```

---

## 7. 错误处理规范

- **所有公开 API** 返回 `ReedResult<T>`（即 `Result<T, ReedError>`）。
- 内部实现使用 `?` 运算符传播错误，禁止在库内部 `panic!`（除非是不变量违反的 `unreachable!()`）。
- 错误消息应包含足够上下文，例如：
  ```rust
  return Err(ReedError::InvalidArgument(
      format!("offsets length {} != nelem*elemsize {}", offsets.len(), nelem * elemsize)
  ));
  ```

---

## 8. 后端扩展规范

新增后端步骤：

1. 新建 crate `reed-{backend}/`。
2. 实现 `struct {Backend}Backend<T: Scalar>`。
3. 为其 `impl Backend<T>`，实现所有工厂方法。
4. 在 `reed-core/src/reed.rs` 的 `Reed::init` 中添加资源字符串匹配分支（通过 feature flag 条件编译）。

```rust
// reed-core/src/reed.rs
#[cfg(feature = "cuda")]
"/gpu/cuda" => Arc::new(reed_cuda::CudaBackend::<T>::new()?),
```

---

## 9. 测试规范

每个模块须包含：

1. **单元测试**（`#[cfg(test)] mod tests`）：测试各个 trait 实现的数值正确性。
2. **集成测试**（`tests/integration.rs`）：端到端测试，验证算符组合后的数值结果。

**数值正确性基准**（质量算符）：
- 1D 区间 `[-1, 1]` 上的质量积分 = 2.0
- 误差阈值：`< 50 * T::EPSILON`

```rust
#[test]
fn test_mass_1d_integral() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    // ... 构造算符 ...
    let sum: f64 = v.iter().sum();
    assert!((sum - 2.0).abs() < 50.0 * f64::EPSILON);
}
```

---

## 10. 命名约定

| 对象 | Rust 类型名 | 说明 |
|------|-------------|------|
| 库上下文 | `Reed<T>` | 对应 libCEED 的 `Ceed` |
| 向量 | `dyn VectorTrait<T>` | |
| 基函数 | `dyn BasisTrait<T>` | |
| 自由度限制 | `dyn ElemRestrictionTrait<T>` | |
| 积分点算符 | `dyn QFunctionTrait<T>` | |
| 弱形式算符 | `dyn OperatorTrait<T>` | |
| CPU 向量 | `CpuVector<T>` | |
| CPU Lagrange 基 | `LagrangeBasis<T>` | |
| CPU 限制算符 | `CpuElemRestriction<T>` | |
| 错误类型 | `ReedError` | |
| 结果类型 | `ReedResult<T>` | |

---

## 11. 关键约束与禁止事项

- **禁止**在 `reed-core` 中依赖任何数值计算 crate（保持零依赖）。
- **禁止** `unsafe` 代码出现在 `reed-core`；`reed-cpu` 中仅在 SIMD 优化处允许，必须封装在安全函数内。
- **禁止**在 trait 方法中分配堆内存（热路径），所有 scratch buffer 通过调用方传入或预分配。
- **必须**为所有公开 API 编写文档注释（`///`），并包含可运行的 doctest 示例。
- **必须**遵循 `rustfmt` 默认格式，通过 `clippy --deny warnings` 检查。

