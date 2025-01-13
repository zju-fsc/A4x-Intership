# **PyTorch 的 `torch.nn.Module` 类介绍**

PyTorch 的 `torch.nn.Module` 是所有神经网络模块的基类。自定义的神经网络模型通常需要继承 `torch.nn.Module`，并实现其方法来定义网络的结构和行为。
### **1. `torch.nn.Module` 的核心功能**

`torch.nn.Module` 提供了以下主要功能：

#### **(1) 模型的定义**
- 它是一个容器，用于存放神经网络的层（如卷积层、全连接层）和其他模块。
- 你可以在 `__init__` 方法中定义网络的层次结构。

#### **(2) 模型的前向传播**
- 通过重写 `forward` 方法，定义数据在网络中的前向传播逻辑。

#### **(3) 参数管理**
- 自动注册模型中的参数（比如权重和偏置），通过 `parameters()` 方法可以轻松访问这些参数。

#### **(4) 存储和加载**
- 提供保存和加载模型的方法（如 `torch.save()` 和 `torch.load()`）。

#### **(5) 递归管理**
- 模块中的子模块（如嵌套的网络层）会被自动注册，便于组织复杂的网络结构。

---

### **2. `torch.nn.Module` 的常用方法**

#### **(1) `__init__`**
- 构造函数，定义模型的层和模块。
- 必须调用父类的 `__init__` 方法来正确初始化。

#### **(2) `forward`**
- 定义前向传播逻辑。
- 在调用模型实例时，PyTorch 会自动执行这个方法。

#### **(3) `parameters()`**
- 返回模型中所有需要优化的参数（如权重和偏置）。

#### **(4) `state_dict()`**
- 返回模型中所有的参数和缓冲区（如权重、偏置、均值、方差等）的状态字典。

#### **(5) `load_state_dict()`**
- 加载保存的参数字典，将模型的状态恢复到特定的训练阶段。

#### **(6) `eval()` 和 `train()`**
- 切换模型的状态：
  - `model.train()`：切换到训练模式（启用 dropout 和 batchnorm）。
  - `model.eval()`：切换到评估模式（禁用 dropout 和 batchnorm）。

#### **(7) `register_buffer()`**
- 注册一个缓冲区（如保存模型的中间变量或统计量），不会被优化。

#### **(8) `cuda()` 和 `to()`**
- 将模型及其参数移动到 GPU 或其他设备。

---

### **3. `torch.nn.Module` 的典型用法**

#### **(1) 自定义模型**
通过继承 `torch.nn.Module` 来定义自己的神经网络结构：

```python
import torch
import torch.nn as nn

# 自定义一个简单的神经网络
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义网络结构
        self.fc1 = nn.Linear(10, 20)  # 输入 10，输出 20
        self.fc2 = nn.Linear(20, 5)   # 输入 20，输出 5

    def forward(self, x):
        # 定义前向传播逻辑
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
model = MyModel()

# 查看模型结构
print(model)

# 输入数据
x = torch.randn(1, 10)  # 输入一个形状为 (1, 10) 的张量
output = model(x)       # 调用 forward 方法
print(output)
```

#### **(2) 使用 `parameters` 方法**
查看模型中的所有参数（权重和偏置）：

```python
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, shape: {param.shape}")
```

#### **(3) 保存和加载模型**
- 保存模型：
  ```python
  torch.save(model.state_dict(), "model.pth")
  ```
- 加载模型：
  ```python
  model.load_state_dict(torch.load("model.pth"))
  ```

#### **(4) 切换训练和评估模式**
- 切换到训练模式（启用 dropout 和 batchnorm）：
  ```python
  model.train()
  ```
- 切换到评估模式（禁用 dropout 和 batchnorm）：
  ```python
  model.eval()
  ```

---

### **4. 常见内置方法和属性**

| **方法/属性**         | **描述**                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `__init__()`          | 定义网络的结构和子模块。                                                |
| `forward(x)`          | 定义前向传播逻辑。                                                     |
| `parameters()`        | 返回所有可训练的参数，用于优化器。                                       |
| `named_parameters()`  | 返回参数及其名称（键值对）。                                             |
| `children()`          | 返回所有直接子模块。                                                    |
| `named_children()`    | 返回子模块及其名称（键值对）。                                           |
| `modules()`           | 返回所有子模块（递归查询）。                                             |
| `named_modules()`     | 返回子模块及其名称（递归查询，键值对）。                                 |
| `state_dict()`        | 返回模块的所有参数和缓冲区的状态字典。                                   |
| `load_state_dict()`   | 加载保存的状态字典，用于恢复模型参数。                                    |
| `eval()`              | 切换到评估模式（禁用 dropout 等）。                                      |
| `train()`             | 切换到训练模式（启用 dropout 等）。                                      |
| `cuda()`              | 将模型移动到 GPU。                                                      |
| `cpu()`               | 将模型移动到 CPU。                                                      |
| `to(device)`          | 将模型移动到指定设备（例如 GPU）。                                       |
| `register_buffer()`   | 注册一个不会被优化的缓冲区（如均值、方差等统计数据）。                     |

---

### **5. 模块化模型的管理**

`torch.nn.Module` 的强大之处在于它支持模块化设计。你可以将一个复杂的神经网络拆分为多个子模块（每个子模块也是一个 `torch.nn.Module`），并将它们嵌套在一起。

#### **示例：模块化设计**
```python
class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return torch.relu(self.fc(x))

class MainModule(nn.Module):
    def __init__(self):
        super(MainModule, self).__init__()
        self.submodule1 = SubModule()
        self.submodule2 = SubModule()

    def forward(self, x):
        x = self.submodule1(x)
        x = self.submodule2(x)
        return x
```

这种模块化设计使得复杂的网络更容易维护和扩展。

---

### **6. `torch.nn.Module` 的优点**

1. **模块化和递归管理：**
   - 支持将复杂的网络拆分为多个子模块，便于管理和调试。

2. **参数管理：**
   - 自动注册网络中的参数，方便与优化器集成。
   - 提供统一的接口访问和管理所有参数。

3. **灵活性：**
   - 允许用户自定义任意复杂的前向传播逻辑。
   - 支持子模块的嵌套和递归调用。

4. **设备管理：**
   - 提供设备无关的接口（如 `to(device)`），便于在 CPU 和 GPU 之间切换。

5. **状态保存和恢复：**
   - 支持将模型的状态保存到文件中，并在需要时恢复。

---

# 类函数总结表格

`torch.nn.Module` 是 PyTorch 的核心模块，用于构建和管理神经网络。它提供了强大的功能来组织网络结构、管理参数和缓冲区、保存/加载模型，以及轻松切换设备（CPU/GPU）。

|**方法**|**作用**|
|---|---|
|`__init__`|初始化模块，定义网络结构和子模块。|
|`forward`|定义前向传播逻辑。|
|`add_module`|动态添加子模块并注册。|
|`cuda`|将模块及其参数移动到 GPU。|
|`cpu`|将模块及其参数移动到 CPU。|
|`__call__`|使模块可调用（自动调用 `forward` 方法）。|
|`parameters`|返回模块中所有需要优化的参数。|
|`named_parameters`|返回模块中所有参数及其名称。|
|`children`|返回所有直接子模块。|
|`named_children`|返回所有直接子模块及其名称。|
|`modules`|返回模块及其所有子模块（递归）。|
|`named_modules`|返回模块及其所有子模块及其名称（递归）。|
|`train`|切换到训练模式。|
|`eval`|切换到评估模式。|
|`zero_grad`|将所有参数的梯度清零。|
|`__repr__`|返回模块的字符串表示形式（如网络结构）。|
|`__dir__`|自定义模块属性的自动补全行为。|


# 钩子
### **钩子（Hooks）是什么？**

钩子（Hooks）是 PyTorch 中的一种机制，它允许你在特定的地方插入自定义的函数逻辑，比如在 **前向传播** 或 **反向传播** 时执行额外的操作。钩子可以用来调试、修改数据流、监控梯度等，非常灵活和强大。

### **PyTorch 提供的钩子种类**

PyTorch 中钩子（hooks）的种类可以分为 **前向传播相关钩子** 和 **反向传播相关钩子**，具体如下：

#### **1. 前向传播相关钩子**

| 钩子名称               | 描述                                        | 作用时机            |
| ------------------ | ----------------------------------------- | --------------- |
| `forward_pre_hook` | 在调用 `forward` 方法 **之前** 执行，可以修改输入数据。      | 发生在 `forward` 前 |
| `forward_hook`     | 在调用 `forward` 方法 **之后** 执行，可以修改输出数据或记录信息。 | 发生在 `forward` 后 |

---

#### **2. 反向传播相关钩子**

| 钩子名称                      | 描述                                                                                     | 作用时机                    |
|-------------------------------|------------------------------------------------------------------------------------------|-----------------------------|
| `register_backward_hook`      | **已被废弃，建议改用 `register_full_backward_hook`**。用于操作模块的梯度。                | 发生在梯度计算后，但传播前   |
| `register_full_backward_hook` | 替代 `register_backward_hook`，更稳定，操作模块的输入梯度和输出梯度。                     | 发生在梯度计算后，但传播前   |
| `tensor.register_hook`        | 注册在张量上的钩子，用于操作该张量的梯度。                                                | 发生在梯度计算后，但传播前   |

---

### **每种钩子的功能和示例**

#### **1. 前向传播钩子**

##### **1.1 `forward_pre_hook`**

- **执行时机**：在 `forward` 方法 **执行之前** 运行。
- **作用**：修改输入数据。

**示例：将输入数据乘以 2**
```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = MyModel()

def pre_hook_fn(module, inputs):
    print(f"Original Input: {inputs}")
    modified_input = inputs[0] * 2
    return (modified_input,)

handle = model.register_forward_pre_hook(pre_hook_fn)

x = torch.tensor([[1.0, 2.0]])
output = model(x)
print("Output:", output)

handle.remove()
```

---

##### **1.2 `forward_hook`**

- **执行时机**：在 `forward` 方法 **执行之后** 运行。
- **作用**：修改输出数据。

**示例：将输出数据加 10**
```python
def forward_hook_fn(module, inputs, output):
    print(f"Original Output: {output}")
    modified_output = output + 10
    return modified_output

handle = model.register_forward_hook(forward_hook_fn)

x = torch.tensor([[1.0, 2.0]])
output = model(x)
print("Final Output:", output)

handle.remove()
```

---

#### **2. 反向传播钩子**

##### **2.1 `tensor.register_hook`**

- **作用范围**：注册在 **张量** 上的钩子。
- **执行时机**：在该张量的梯度被计算后，传播给前面的模块之前。

**示例：查看梯度并对其进行修改**
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)

def tensor_hook_fn(grad):
    print(f"Original Gradient: {grad}")
    return grad * 2  # 修改梯度

x.register_hook(tensor_hook_fn)

y = x.sum()
y.backward()
```

##### **输出：**
```
Original Gradient: tensor([1., 1.])
```
梯度被修改后会传播给前面的计算图。

---

##### **2.2 `register_full_backward_hook`**

- **作用范围**：注册在 **模块** 上的钩子。
- **执行时机**：在模块计算梯度后，梯度传播给前面模块之前。
- **作用**：可以访问输入和输出的梯度，甚至修改它们。

**示例：修改输入和输出梯度**
```python
def full_backward_hook_fn(module, grad_input, grad_output):
    print(f"Gradient Input: {grad_input}")
    print(f"Gradient Output: {grad_output}")
    # 修改输入梯度
    modified_grad_input = tuple(g * 2 if g is not None else None for g in grad_input)
    return modified_grad_input

model = MyModel()
handle = model.register_full_backward_hook(full_backward_hook_fn)

x = torch.tensor([[1.0, 2.0]], requires_grad=True)
output = model(x).sum()
output.backward()

handle.remove()
```

##### **输出：**
```
Gradient Input: (...)
Gradient Output: (...)
```

---

### **关于 Full Hook**

在 PyTorch 中，"Full Hook" 通常指的是 **`register_full_backward_hook`**，它是对 `register_backward_hook` 的改进和替代。

#### **为什么叫 "Full Hook"？**
- **全功能**：`register_full_backward_hook` 提供了对 **模块输入梯度和输出梯度** 的访问和修改能力。
- **稳定性**：相比旧的 `register_backward_hook`，`register_full_backward_hook` 在分布式训练和复杂计算图中表现更稳定。

---

### **所有钩子总结**

| **钩子名称**                   | **作用范围**         | **作用时机**                | **能修改什么**               | **用途**                                         |
|--------------------------------|----------------------|-----------------------------|-----------------------------|-------------------------------------------------|
| `forward_pre_hook`             | 模块                | `forward` 调用前             | 输入（`*args`）              | 修改或记录输入                                   |
| `forward_hook`                 | 模块                | `forward` 调用后             | 输出                        | 修改或记录输出                                   |
| `tensor.register_hook`         | 张量                | 梯度计算后，传递前            | 当前张量的梯度              | 修改当前张量的梯度                               |
| `register_full_backward_hook`  | 模块                | 梯度计算后，传递前            | 输入梯度和输出梯度           | 修改模块的输入梯度或输出梯度                     |

---

### **是否有完整覆盖的 Hook（Full Hook Across Forward and Backward）？**

目前，PyTorch 的钩子分为两部分：
1. **前向传播相关**：`forward_pre_hook` 和 `forward_hook`。
2. **反向传播相关**：`tensor.register_hook` 和 `register_full_backward_hook`。

### **要实现 "Full Hook" 的效果**
你可以结合前向传播和反向传播钩子，实现从输入到输出，再到梯度的全流程监控和修改。

# add_module和__init__中使用self.x直接添加的区别
### **1. 直接赋值（`self.linear = nn.Linear(10, 10)`）**

#### **作用**
- 直接将一个子模块分配给模块的属性。
- PyTorch 的 `torch.nn.Module` 在重写的 `__setattr__` 方法中，会自动检测赋值的对象是否是 `Module` 类型，并将其注册到模块的 `_modules` 字典中。

#### **核心机制**
当你写以下代码：
```python
self.linear = nn.Linear(10, 10)
```
PyTorch 的 `__setattr__` 方法会自动执行：
1. 检测赋值的对象 `nn.Linear(10, 10)` 是否是一个 `torch.nn.Module`。
2. 如果是，它会将这个子模块注册到当前模块的 `_modules` 字典中。
3. 最终效果是与 `add_module` 相同。

---

#### **示例**
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)  # 自动注册子模块

model = MyModel()
print(model._modules)  # 输出：{'linear': Linear(in_features=10, out_features=10, bias=True)}
```

---

### **2. 使用 `add_module`**

#### **作用**
- 显式地将子模块注册到模块的 `_modules` 字典中。
- 不会触发 `__setattr__` 方法的逻辑，而是直接操作 `_modules` 字典。
- 更加灵活，适合动态添加子模块（特别是子模块的名称是运行时决定的）。

#### **核心机制**
当你写以下代码：
```python
self.add_module("linear", nn.Linear(10, 10))
```
代码直接将子模块注册到 `_modules` 中，而不会通过 `__setattr__` 方法。

---

#### **示例**
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module("linear", nn.Linear(10, 10))  # 显式注册子模块

model = MyModel()
print(model._modules)  # 输出：{'linear': Linear(in_features=10, out_features=10, bias=True)}
```

---

### **3. 区别**

| 特性                          | `self.linear = nn.Linear(...)`    | `self.add_module(...)`             |
|-------------------------------|------------------------------------|-------------------------------------|
| **调用机制**                  | 通过 `__setattr__` 自动注册       | 显式调用 `add_module` 方法          |
| **模块检测**                  | 自动检测是否为 `nn.Module`         | 必须手动传入 `nn.Module`            |
| **动态性**                    | 子模块名称必须在代码中显式写出     | 子模块名称可以动态生成              |
| **属性访问方式**              | 直接通过 `self.linear` 访问        | 直接通过 `self.linear` 访问         |
| **推荐场景**                  | 用于静态定义模型结构               | 用于动态添加子模块（如循环中添加）   |

---

### **4. 动态添加子模块**

#### **使用 `add_module` 的场景**
如果模块的名称在运行时动态生成，使用 `add_module` 更加方便。例如：
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(3):  # 动态添加多个子模块
            self.add_module(f"linear_{i}", nn.Linear(10, 10))

model = MyModel()
print(model._modules)
```

##### **输出：**
```
OrderedDict([
    ('linear_0', Linear(in_features=10, out_features=10, bias=True)),
    ('linear_1', Linear(in_features=10, out_features=10, bias=True)),
    ('linear_2', Linear(in_features=10, out_features=10, bias=True))
])
```

#### **用直接赋值的局限性**
直接赋值时，必须显式地写出子模块名称，无法动态创建变量：
```python
for i in range(3):
    self.linear_{i} = nn.Linear(10, 10)  # 无法这样写
```
上面的代码会导致语法错误，因为 Python 中不能动态定义变量名。

---

### **5. `_modules` 的作用**

无论使用 `self.linear = nn.Linear(...)` 还是 `self.add_module(...)`，最终子模块都会被存储在模块的 `_modules` 字典中。`_modules` 是 PyTorch 用于管理子模块的核心数据结构，所有子模块的名称和对象都会存储在这个字典中。

#### **示例**
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.add_module("linear2", nn.Linear(20, 20))

model = MyModel()
print(model._modules)
```

##### **输出：**
```
OrderedDict([
    ('linear1', Linear(in_features=10, out_features=10, bias=True)),
    ('linear2', Linear(in_features=20, out_features=20, bias=True))
])
```

---

### **6. 总结**

#### **相同点**
- `self.linear = nn.Linear(...)` 和 `self.add_module(...)` 最终都会将子模块添加到 `_modules` 字典中。
- 添加的子模块可以通过名称（如 `self.linear`）访问。

#### **不同点**
1. **调用方式：**
   - `self.linear = ...` 是通过 `__setattr__` 自动注册。
   - `add_module` 是显式调用，直接操作 `_modules`。

2. **动态性：**
   - `add_module` 更适合动态添加子模块（如循环中动态生成名称）。
   - 直接赋值更适合静态定义模块结构。

3. **代码语义：**
   - 如果子模块是模型结构的一部分，使用直接赋值更加直观。
   - 如果需要动态注册子模块，使用 `add_module` 更加灵活。

#### **推荐使用**
- **静态模块定义：**
  ```python
  self.linear = nn.Linear(10, 10)
  ```
- **动态模块添加：**
  ```python
  self.add_module(f"layer_{i}", nn.Linear(10, 10))
  ```

