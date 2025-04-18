# 流程图解

1. **Shadow Hand参数（28D = 22+3+3）** ↓
    
2. **Shadow Hand 21点点云** ↓
    
3. **作为优化目标** ↓
    
4. **NLOPT优化MANO Hand参数（22D）** ↓
    
5. **MANO Hand点云与目标点云距离最小**
    

# 骨架与自由度

你的问题非常好，下面我为你**系统地梳理一下“骨架点（keypoints）”与“关节参数（joint parameters）”的区别**，并且简要介绍一下**机械手的发展和骨架点的通用性**。

---

## 1. 机械手的发展概况

### 1.1 早期机械手

- **工业机械手**最早只关注抓取和重复动作，结构简单（2~6自由度），如大型机械臂。
    
- **刚性连接、末端执行器简单**，一般无复杂“手指”。
    

### 1.2 多指灵巧手

- **仿人五指手**（如Shadow Hand、Allegro Hand、Barrett Hand等）出现，模仿人手的复杂灵活运动。
    
- 伴随AI和机器人学发展，出现了越来越多自由度高、形态仿生的手型。
    

### 1.3 数据驱动与通用骨架

- 为了**用机器学习/感知方法描述手部动作**，学术界和工业界提出了“手部21骨架点”方案，广泛用于视觉和机器人任务（如OpenPose、MediaPipe、MANO模型等）。
    
- 这种21点方案和人类解剖学紧密相关，成为一种“事实标准”。
    

---

## 2. 21骨架点的来源与通用性

### 2.1 21点方案的由来

- 最初源自**人手关键点检测**任务，例如MediaPipe、OpenPose、MANO等3D手模型。
    
- 一般定义为：**手腕+每根手指4个关节**（5×4=20）= 21个点。
    

### 2.2 机械手是否都遵守？

- 不完全遵守
    
    ：机械手的实际关节数、布局、自由度可能和21点标准不同。
    
    - 例如Allegro Hand有16自由度，Shadow Hand有24自由度，Barrett Hand 9自由度。
        
    - 有的机械手没有拇指对掌，有的指根/指尖结构和人不同。
        
- **但21点作为“空间形状的通用表达”被广泛采用**，因为它方便与人手对齐、做视觉检测和跨平台重定向。
    

### 2.3 机械手与21点的关系

- **仿生五指手**通常可以通过前向运动学，计算出类似“21点骨架”来表示空间形态。
    
- **非仿生手**（如二指夹爪），不适用21点，但可以根据自身结构定义骨架点。
    

---

## 3. 骨架点（Keypoints） vs. 关节参数（Joint Parameters）

### 3.1 骨架点（keypoints）

- **定义**：手上若干有明确物理含义的空间点（如手腕、指尖、各指关节节点）。
    
- 用途
    
    ：
    
    - 作为视觉感知（检测/识别）的输出。
        
    - 描述手的形状、姿态。
        
    - 便于不同手型之间的“动作对齐”。
        
- 特性
    
    ：
    
    - 是三维空间中的点 (x, y, z)。
        
    - 点的数量和顺序通常是预先定义好的（如21点方案）。
        

### 3.2 关节参数（joint parameters）

- **定义**：描述机械手各关节的角度（或位移），即机械手的“动作空间参数”。
    
- 用途
    
    ：
    
    - 控制机械手运动的实际输入。
        
    - 用于机器人运动学、动力学建模。
        
- 特性
    
    ：
    
    - 通常是一组角度或位置值（如[θ1, θ2, ..., θn]）。
        
    - 不同手的参数数量、顺序、物理意义都可能不同。
        

### 3.3 二者关系

- 给定一组**关节参数**，通过**前向运动学（FK）*_可以唯一计算出一组*_骨架点**的位置。
    
- 反之，给定骨架点，通常无法唯一反推出关节参数（逆运动学问题，可能有多解或无解）。
    

---

## 4. 经典21点骨架的定义（以MANO为例）

|编号|含义|英文名|
|---|---|---|
|0|手腕|Wrist|
|1-4|食指4关节|Index finger (MCP, PIP, DIP, Tip)|
|5-8|中指4关节|Middle finger|
|9-12|无名指4关节|Ring finger|
|13-16|小指4关节|Little finger|
|17-20|拇指4关节|Thumb|

注：Tip为指尖，MCP/PIP/DIP为常用解剖学关节名。

---

## 5. 图示说明

- ![MANO 21点骨架图](https://pfst.cf2.poecdn.net/base/image/f6961f00ae640276641918bd4434226255543de224dc50d1490e9c7777dab9a3?w=161&h=81&pmaid=351467415) （图片仅为示意，具体点请参考你的21点编号图）
    

---

## 6. 总结

- 21点骨架是**人手/仿生手领域的事实通用标准**，便于视觉检测和跨域映射。
    
- 机械手的**关节参数**依赖于结构设计，不同手之间完全可能不同。
    
- **用21点骨架描述机械手的空间形状，便于和人手对齐、做动作迁移和重定向**。
    
- 但在机器人控制和运动学上，最终还是要用各自手的关节参数。

# Shadowhand Info
关节顺序
joint_names = [  
'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',  
'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',  
'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',  
'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',  
'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'  
]
21点顺序
shadow_seq_joints, _ = get_shadow_hand(seq_i, hand_model=shadow_hand_model) #(40, 21, 3)
这里`get_shadow_hand`函数输出了 `(帧数, 21, 3)` 的点坐标数组，
- 即每一帧有21个Shadow Hand的关节点（顺序由get_shadow_hand定义）。


# MANO INFO
21点顺序
- 0：手腕
- 1-4：食指（Index）从根到指尖
- 5-8：中指（Middle）从根到指尖
- 9-12：无名指（Ring）从根到指尖
- 13-16：小指（Little）从根到指尖
- 17-20：拇指（Thumb）从根到指尖


Shadowhand 到 MANO的映射：
mano_joints_order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
shadow_joints_order = [0, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
