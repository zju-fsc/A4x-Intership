#Parallel
# 如何实现并行
```python
import genesis as gs
import torch

########################## 初始化 ##########################
gs.init(backend=gs.gpu)

########################## 创建场景 ##########################
scene = gs.Scene(
    show_viewer    = True,
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3.5, -1.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
    ),
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
    ),
)

########################## 实体 ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

########################## 构建 ##########################

# 创建20个并行环境
B = 20
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

# 控制所有机器人
franka.control_dofs_position(
    torch.tile(
        torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0.02, 0.02], device=gs.device), (B, 1)
    ),
)

for i in range(1000):
    scene.step()
```

![](Pasted%20image%2020250104171134.png)


单一例子和[Getting Started](Getting%20Started.md)中一致

回想一下我们在之前的教程[Control Your Robots](Control%20Your%20Robots.md)中使用的API，例如`franka.control_dofs_position()`。现在你可以使用完全相同的API来控制批量机器人，只是输入变量需要一个额外的批量维度：

franka.control_dofs_position(torch.zeros(B, 9, device=gs.device))

由于我们在GPU上运行仿真，为了减少CPU和GPU之间的数据传输开销，我们可以使用通过`gs.device`选择的torch张量而不是numpy数组（但numpy数组也可以工作）。当你需要频繁发送一个具有巨大批量大小的张量时，这可以带来显著的性能提升。

上述调用将控制批量环境中的所有机器人。如果你只想控制某些环境，可以另外传入`envs_idx`，但请确保`position`张量的批量维度大小与`envs_idx`的长度匹配：

```python
# 只控制3个环境：1, 5和7。
franka.control_dofs_position(
    position = torch.zeros(3, 9, device=gs.device),
    envs_idx = torch.tensor([1, 5, 7], device=gs.device),
)
```

此调用将仅向3个选定的环境发送零位置命令。


# 关闭显示
注意在关闭显示的情况下能超高速运作：
```python
import torch
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer   = False,
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

scene.build(n_envs=30000)

# 控制所有机器人
franka.control_dofs_position(
    torch.tile(
        torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0.02, 0.02], device=gs.device), (30000, 1)
    ),
)

for i in range(1000):
    scene.step()
```


### **`torch.tile` 的作用**

`torch.tile` 是 PyTorch 中的一个函数，用于将张量（Tensor）沿指定的维度重复（tile）一定次数，从而构造一个更大的张量。可以理解为 **“复制并拼接张量”** 的操作，它类似于 NumPy 中的 `np.tile`。

---

### **语法**
```python
torch.tile(input, dims)
```

- **`input`**: 输入的张量。
- **`dims`**: 一个元组或列表，指定每个维度上重复的次数。

---

### **例子**

#### **单维度重复**
```python
import torch

x = torch.tensor([1, 2, 3])
result = torch.tile(x, (2,))
print(result)  # 输出: tensor([1, 2, 3, 1, 2, 3])
```
- 原始张量 `x` 有 1 个维度。
- `(2,)` 表示沿第 0 维重复 2 次，因此结果是 `[1, 2, 3, 1, 2, 3]`。

#### **多维度重复**
```python
x = torch.tensor([[1, 2], [3, 4]])
result = torch.tile(x, (2, 3))
print(result)
# 输出:
# tensor([[1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4],
#         [1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4]])
```
- 原始张量 `x` 的形状是 `(2, 2)`。
- `(2, 3)` 表示：
  - 沿第 0 维（行）重复 2 次。
  - 沿第 1 维（列）重复 3 次。
- 结果的形状是 `(4, 6)`。

---

### **在代码中的作用**

```python
torch.tile(
    torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0.02, 0.02], device=gs.device), (B, 1)
)
```

#### **1. 输入张量**
```python
torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0.02, 0.02], device=gs.device)
```
- 这是一个一维张量，包含目标关节位置的值。
- 例如，对于 Franka Panda 机械臂：
  - `0, 0, 0, -1.0, 0, 0, 0`：对应 7 个旋转关节的目标位置。
  - `0.02, 0.02`：对应 2 个滑动关节（夹爪）的目标位置。
- `device=gs.device`：表示张量会被分配到指定的设备（如 GPU）。

#### **2. 重复操作**
```python
torch.tile(..., (B, 1))
```
- `(B, 1)` 表示沿张量维度重复：
  - 第 0 维（行）重复 `B` 次。
  - 第 1 维（列）保持不变。
- 如果原始张量的形状是 `(9,)`，重复后张量的形状会变成 `(B, 9)`。

#### **3. 示例**
假设 `B=3`，重复后的结果是：
```python
tensor([
    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.02, 0.02],
    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.02, 0.02],
    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.02, 0.02],
])
```
- 每一行都是同样的目标关节位置值。
- 这个操作通常用于 **批量控制**（Batch Control）：一次性为多个机器人（或者多个仿真实例）设置相同的目标关节位置。

