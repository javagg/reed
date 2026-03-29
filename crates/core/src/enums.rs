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
    /// 插值 B·u
    Interp,
    /// 梯度 ∇B·u
    Grad,
    /// 散度 ∇·B·u
    Div,
    /// 旋度 ∇×B·u
    Curl,
    /// 积分权重 w_q
    Weight,
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
    /// L1 范数
    One,
    /// L2 范数
    Two,
    /// L∞ 范数
    Max,
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
