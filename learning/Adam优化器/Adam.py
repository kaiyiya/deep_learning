import math
from typing import Callable, Tuple

def quadratic_function(theta: Tuple[float, float]) -> float:
    """
    一个简单的二维二次函数:
    f(x, y) = (x - 3)^2 + (y + 1)^2
    其最小值在 (3, -1) 处, 用来演示 Adam 优化器如何快速逼近最优点。
    """
    x, y = theta
    return (x - 3) ** 2 + (y + 1) ** 2


def quadratic_gradient(theta: Tuple[float, float]) -> Tuple[float, float]:
    """
    解析梯度:
    ∂f/∂x = 2(x - 3)
    ∂f/∂y = 2(y + 1)
    """
    x, y = theta
    return 2 * (x - 3), 2 * (y + 1)


def adam_optimize(
        grad_fn: Callable[[Tuple[float, float]], Tuple[float, float]],
        init_theta: Tuple[float, float],
        lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        steps: int = 50,
) -> Tuple[Tuple[float, float], list]:
    """
    手写 Adam 优化过程, 展示:
    1. 一阶矩估计 m_t (动量)
    2. 二阶矩估计 v_t (均方梯度)
    3. 偏差校正 m_t_hat, v_t_hat

    返回最终参数以及每一步的轨迹, 便于观察收敛速度。
    """
    theta = init_theta
    m = (0.0, 0.0)
    v = (0.0, 0.0)
    trajectory = [theta]

    for t in range(1, steps + 1):
        g = grad_fn(theta)

        m = tuple(beta1 * m_i + (1 - beta1) * g_i for m_i, g_i in zip(m, g))
        v = tuple(beta2 * v_i + (1 - beta2) * (g_i ** 2) for v_i, g_i in zip(v, g))

        # 偏差校正
        m_hat = tuple(m_i / (1 - beta1 ** t) for m_i in m)
        v_hat = tuple(v_i / (1 - beta2 ** t) for v_i in v)

        # 参数更新
        theta = tuple(
            theta_i - lr * m_hat_i / (math.sqrt(v_hat_i) + eps)
            for theta_i, m_hat_i, v_hat_i in zip(theta, m_hat, v_hat)
        )

        trajectory.append(theta)

    return theta, trajectory


if __name__ == "__main__":
    init_point = (-4.0, 5.0)
    best_theta, trace = adam_optimize(quadratic_gradient, init_point, lr=0.001, steps=60)
    print("Adam 优化演示 - 目标函数 f(x,y) = (x-3)^2 + (y+1)^2")
    print(f"初始点: {init_point}")
    print(f"最终参数: {best_theta}")
    print(f"最终函数值: {quadratic_function(best_theta):.6f}")
    print("\n前几次迭代轨迹 (x, y):")
    for i, (x, y) in enumerate(trace[:5]):
        print(f"Step {i:02d}: ({x:.4f}, {y:.4f})")

    # 学习率 η / lr：控制每步前进幅度。调大能更快逼近，但易振荡或发散；调小更稳但收敛慢。常用 1e-3，也可按任务/阶段自适应调度，比如 warmup→cosine decay。
    # 动量衰减 β₁：决定历史梯度保留多少。默认 0.9。调高（接近 1）会让方向更平滑但响应慢，适合噪声大场景；调低会更敏感、可能抖动，适合快速捕捉突变。
    # 均方梯度衰减 β₂：控制二阶矩平滑程度。默认 0.999。如果梯度变化剧烈，可稍微调低（如 0.98~0.995）让它更快反映最新方差信息；若任务本身噪声大，保持高值能稳定步长。
    # 数值稳定项 ε：默认 1e-8。几乎不动，但在低精度训练或梯度极小的网络里，适当调大（如 1e-7）避免分母过小；过大则会削弱自适应效果。
    # 权重衰减 / L2 正则（若实现中提供 weight_decay 或 λ）：抑制过拟合。调大会迫使参数更接近零，但学习太慢；调小或关掉适合数据量大或正则化另有安排时。
    # 学习率调度：虽然不是 Adam 内部的参数，但常搭配。例如 warmup、余弦退火、StepLR 等，可在保持 β₁/β₂ 默认的情况下，用调度器解决不同训练阶段的速度与稳定性平衡。
    # 梯度裁剪阈值（若框架配合使用）：设置最大梯度范数，防止爆炸。配合 Adam 可在序列模型或 GAN 里提升稳定性。
    # 批大小 / 噪声水平：间接影响 Adam 表现。梯度噪声大时，适度调高 β₁、β₂ 或减小 lr；批量增大时，可考虑稍提 lr 或使用 β₁=0.9, β₂=0.999 的常规值。
    # 调参实践建议：先用默认 (lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-8)，观察损失曲线；若震荡就减 lr 或调大 β₁/β₂；若收敛慢可小幅提 lr 或降 β₁；遇到长尾噪声就尝试调低 β₂ 或加裁剪。

