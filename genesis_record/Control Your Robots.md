#xml #RobotControl
```xml
<mujoco model="panda">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <option integrator="implicitfast"/>

  <default>
    <default class="panda">
      <material specular="0.5" shininess="0.25"/>
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
      <default class="finger">
        <joint axis="0 1 0" type="slide" range="0 0.04"/>
      </default>

      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3"/>
        <default class="fingertip_pad_collision_1">
          <geom type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445"/>
        </default>
        <default class="fingertip_pad_collision_2">
          <geom type="box" size="0.003 0.002 0.003" pos="0.0055 0.002 0.05"/>
        </default>
        <default class="fingertip_pad_collision_3">
          <geom type="box" size="0.003 0.002 0.003" pos="-0.0055 0.002 0.05"/>
        </default>
        <default class="fingertip_pad_collision_4">
          <geom type="box" size="0.003 0.002 0.0035" pos="0.0055 0.002 0.0395"/>
        </default>
        <default class="fingertip_pad_collision_5">
          <geom type="box" size="0.003 0.002 0.0035" pos="-0.0055 0.002 0.0395"/>
        </default>
      </default>
    </default>
  </default>

  <asset>
    <material class="panda" name="white" rgba="1 1 1 1"/>
    <material class="panda" name="off_white" rgba="0.901961 0.921569 0.929412 1"/>
    <material class="panda" name="black" rgba="0.25 0.25 0.25 1"/>
    <material class="panda" name="green" rgba="0 1 0 1"/>
    <material class="panda" name="light_blue" rgba="0.039216 0.541176 0.780392 1"/>

    <!-- Collision meshes -->
    <mesh name="link0_c" file="link0.stl"/>
    <mesh name="link1_c" file="link1.stl"/>
    <mesh name="link2_c" file="link2.stl"/>
    <mesh name="link3_c" file="link3.stl"/>
    <mesh name="link4_c" file="link4.stl"/>
    <mesh name="link5_c0" file="link5_collision_0.obj"/>
    <mesh name="link5_c1" file="link5_collision_1.obj"/>
    <mesh name="link5_c2" file="link5_collision_2.obj"/>
    <mesh name="link6_c" file="link6.stl"/>
    <mesh name="link7_c" file="link7.stl"/>
    <mesh name="hand_c" file="hand.stl"/>

    <!-- Visual meshes -->
    <mesh file="link0_0.obj"/>
    <mesh file="link0_1.obj"/>
    <mesh file="link0_2.obj"/>
    <mesh file="link0_3.obj"/>
    <mesh file="link0_4.obj"/>
    <mesh file="link0_5.obj"/>
    <mesh file="link0_7.obj"/>
    <mesh file="link0_8.obj"/>
    <mesh file="link0_9.obj"/>
    <mesh file="link0_10.obj"/>
    <mesh file="link0_11.obj"/>
    <mesh file="link1.obj"/>
    <mesh file="link2.obj"/>
    <mesh file="link3_0.obj"/>
    <mesh file="link3_1.obj"/>
    <mesh file="link3_2.obj"/>
    <mesh file="link3_3.obj"/>
    <mesh file="link4_0.obj"/>
    <mesh file="link4_1.obj"/>
    <mesh file="link4_2.obj"/>
    <mesh file="link4_3.obj"/>
    <mesh file="link5_0.obj"/>
    <mesh file="link5_1.obj"/>
    <mesh file="link5_2.obj"/>
    <mesh file="link6_0.obj"/>
    <mesh file="link6_1.obj"/>
    <mesh file="link6_2.obj"/>
    <mesh file="link6_3.obj"/>
    <mesh file="link6_4.obj"/>
    <mesh file="link6_5.obj"/>
    <mesh file="link6_6.obj"/>
    <mesh file="link6_7.obj"/>
    <mesh file="link6_8.obj"/>
    <mesh file="link6_9.obj"/>
    <mesh file="link6_10.obj"/>
    <mesh file="link6_11.obj"/>
    <mesh file="link6_12.obj"/>
    <mesh file="link6_13.obj"/>
    <mesh file="link6_14.obj"/>
    <mesh file="link6_15.obj"/>
    <mesh file="link6_16.obj"/>
    <mesh file="link7_0.obj"/>
    <mesh file="link7_1.obj"/>
    <mesh file="link7_2.obj"/>
    <mesh file="link7_3.obj"/>
    <mesh file="link7_4.obj"/>
    <mesh file="link7_5.obj"/>
    <mesh file="link7_6.obj"/>
    <mesh file="link7_7.obj"/>
    <mesh file="hand_0.obj"/>
    <mesh file="hand_1.obj"/>
    <mesh file="hand_2.obj"/>
    <mesh file="hand_3.obj"/>
    <mesh file="hand_4.obj"/>
    <mesh file="finger_0.obj"/>
    <mesh file="finger_1.obj"/>
  </asset>

  <worldbody>
    <light name="top" pos="0 0 2" mode="trackcom"/>
    <body name="link0" childclass="panda">
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
        fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
      <geom mesh="link0_0" material="off_white" class="visual"/>
      <geom mesh="link0_1" material="black" class="visual"/>
      <geom mesh="link0_2" material="off_white" class="visual"/>
      <geom mesh="link0_3" material="black" class="visual"/>
      <geom mesh="link0_4" material="off_white" class="visual"/>
      <geom mesh="link0_5" material="black" class="visual"/>
      <geom mesh="link0_7" material="white" class="visual"/>
      <geom mesh="link0_8" material="white" class="visual"/>
      <geom mesh="link0_9" material="black" class="visual"/>
      <geom mesh="link0_10" material="off_white" class="visual"/>
      <geom mesh="link0_11" material="white" class="visual"/>
      <geom mesh="link0_c" class="collision"/>
      <body name="link1" pos="0 0 0.333">
        <inertial mass="4.970684" pos="0.003875 0.002081 -0.04762"
          fullinertia="0.70337 0.70661 0.0091170 -0.00013900 0.0067720 0.019169"/>
        <joint name="joint1"/>
        <geom material="white" mesh="link1" class="visual"/>
        <geom mesh="link1_c" class="collision"/>
        <body name="link2" quat="1 -1 0 0">
          <inertial mass="0.646926" pos="-0.003141 -0.02872 0.003495"
            fullinertia="0.0079620 2.8110e-2 2.5995e-2 -3.925e-3 1.0254e-2 7.04e-4"/>
          <joint name="joint2" range="-1.7628 1.7628"/>
          <geom material="white" mesh="link2" class="visual"/>
          <geom mesh="link2_c" class="collision"/>
          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
            <joint name="joint3"/>
            <inertial mass="3.228604" pos="2.7518e-2 3.9252e-2 -6.6502e-2"
              fullinertia="3.7242e-2 3.6155e-2 1.083e-2 -4.761e-3 -1.1396e-2 -1.2805e-2"/>
            <geom mesh="link3_0" material="white" class="visual"/>
            <geom mesh="link3_1" material="white" class="visual"/>
            <geom mesh="link3_2" material="white" class="visual"/>
            <geom mesh="link3_3" material="black" class="visual"/>
            <geom mesh="link3_c" class="collision"/>
            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="3.587895" pos="-5.317e-2 1.04419e-1 2.7454e-2"
                fullinertia="2.5853e-2 1.9552e-2 2.8323e-2 7.796e-3 -1.332e-3 8.641e-3"/>
              <joint name="joint4" range="-3.0718 -0.0698"/>
              <geom mesh="link4_0" material="white" class="visual"/>
              <geom mesh="link4_1" material="white" class="visual"/>
              <geom mesh="link4_2" material="black" class="visual"/>
              <geom mesh="link4_3" material="white" class="visual"/>
              <geom mesh="link4_c" class="collision"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.225946" pos="-1.1953e-2 4.1065e-2 -3.8437e-2"
                  fullinertia="3.5549e-2 2.9474e-2 8.627e-3 -2.117e-3 -4.037e-3 2.29e-4"/>
                <joint name="joint5"/>
                <geom mesh="link5_0" material="black" class="visual"/>
                <geom mesh="link5_1" material="white" class="visual"/>
                <geom mesh="link5_2" material="white" class="visual"/>
                <geom mesh="link5_c0" class="collision"/>
                <geom mesh="link5_c1" class="collision"/>
                <geom mesh="link5_c2" class="collision"/>
                <body name="link6" quat="1 1 0 0">
                  <inertial mass="1.666555" pos="6.0149e-2 -1.4117e-2 -1.0517e-2"
                    fullinertia="1.964e-3 4.354e-3 5.433e-3 1.09e-4 -1.158e-3 3.41e-4"/>
                  <joint name="joint6" range="-0.0175 3.7525"/>
                  <geom mesh="link6_0" material="off_white" class="visual"/>
                  <geom mesh="link6_1" material="white" class="visual"/>
                  <geom mesh="link6_2" material="black" class="visual"/>
                  <geom mesh="link6_3" material="white" class="visual"/>
                  <geom mesh="link6_4" material="white" class="visual"/>
                  <geom mesh="link6_5" material="white" class="visual"/>
                  <geom mesh="link6_6" material="white" class="visual"/>
                  <geom mesh="link6_7" material="light_blue" class="visual"/>
                  <geom mesh="link6_8" material="light_blue" class="visual"/>
                  <geom mesh="link6_9" material="black" class="visual"/>
                  <geom mesh="link6_10" material="black" class="visual"/>
                  <geom mesh="link6_11" material="white" class="visual"/>
                  <geom mesh="link6_12" material="green" class="visual"/>
                  <geom mesh="link6_13" material="white" class="visual"/>
                  <geom mesh="link6_14" material="black" class="visual"/>
                  <geom mesh="link6_15" material="black" class="visual"/>
                  <geom mesh="link6_16" material="white" class="visual"/>
                  <geom mesh="link6_c" class="collision"/>
                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="7.35522e-01" pos="1.0517e-2 -4.252e-3 6.1597e-2"
                      fullinertia="1.2516e-2 1.0027e-2 4.815e-3 -4.28e-4 -1.196e-3 -7.41e-4"/>
                    <joint name="joint7"/>
                    <geom mesh="link7_0" material="white" class="visual"/>
                    <geom mesh="link7_1" material="black" class="visual"/>
                    <geom mesh="link7_2" material="black" class="visual"/>
                    <geom mesh="link7_3" material="black" class="visual"/>
                    <geom mesh="link7_4" material="black" class="visual"/>
                    <geom mesh="link7_5" material="black" class="visual"/>
                    <geom mesh="link7_6" material="black" class="visual"/>
                    <geom mesh="link7_7" material="white" class="visual"/>
                    <geom mesh="link7_c" class="collision"/>
                    <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                      <geom mesh="hand_0" material="off_white" class="visual"/>
                      <geom mesh="hand_1" material="black" class="visual"/>
                      <geom mesh="hand_2" material="black" class="visual"/>
                      <geom mesh="hand_3" material="white" class="visual"/>
                      <geom mesh="hand_4" material="off_white" class="visual"/>
                      <geom mesh="hand_c" class="collision"/>
                      <body name="left_finger" pos="0 0 0.0584">
                        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                        <joint name="finger_joint1" class="finger"/>
                        <geom mesh="finger_0" material="off_white" class="visual"/>
                        <geom mesh="finger_1" material="black" class="visual"/>
                        <geom mesh="finger_0" class="collision"/>
                        <geom class="fingertip_pad_collision_1"/>
                        <geom class="fingertip_pad_collision_2"/>
                        <geom class="fingertip_pad_collision_3"/>
                        <geom class="fingertip_pad_collision_4"/>
                        <geom class="fingertip_pad_collision_5"/>
                      </body>
                      <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                        <joint name="finger_joint2" class="finger"/>
                        <geom mesh="finger_0" material="off_white" class="visual"/>
                        <geom mesh="finger_1" material="black" class="visual"/>
                        <geom mesh="finger_0" class="collision"/>
                        <geom class="fingertip_pad_collision_1"/>
                        <geom class="fingertip_pad_collision_2"/>
                        <geom class="fingertip_pad_collision_3"/>
                        <geom class="fingertip_pad_collision_4"/>
                        <geom class="fingertip_pad_collision_5"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <tendon>
    <fixed name="split">
      <joint joint="finger_joint1" coef="0.5"/>
      <joint joint="finger_joint2" coef="0.5"/>
    </fixed>
  </tendon>

  <equality>
    <joint joint1="finger_joint1" joint2="finger_joint2" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>

  <actuator>
    <general class="panda" name="actuator1" joint="joint1" gainprm="4500" biasprm="0 -4500 -450"/>
    <general class="panda" name="actuator2" joint="joint2" gainprm="4500" biasprm="0 -4500 -450"
      ctrlrange="-1.7628 1.7628"/>
    <general class="panda" name="actuator3" joint="joint3" gainprm="3500" biasprm="0 -3500 -350"/>
    <general class="panda" name="actuator4" joint="joint4" gainprm="3500" biasprm="0 -3500 -350"
      ctrlrange="-3.0718 -0.0698"/>
    <general class="panda" name="actuator5" joint="joint5" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <general class="panda" name="actuator6" joint="joint6" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"
      ctrlrange="-0.0175 3.7525"/>
    <general class="panda" name="actuator7" joint="joint7" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <!-- Remap original ctrlrange (0, 0.04) to (0, 255): 0.04 * 100 / 255 = 0.01568627451 -->
    <general class="panda" name="actuator8" tendon="split" forcerange="-100 100" ctrlrange="0 255"
      gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04" ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255"/>
  </keyframe>
</mujoco>
```
# 提取xml信息
以下是一个python脚本帮助提取xml文件信息：
```python
import xml.etree.ElementTree as ET

# 解析 XML 文件
def parse_mujoco_xml(file_path):
    # 解析 XML 文件
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # 初始化存储关节信息的列表
    joints = []
    keyframes = []

    # 找到 <worldbody> 中的所有 <joint
关节信息:
- 名称: None, 类型: 旋转关节, 范围: None, 轴线: None
- 名称: root, 类型: 旋转关节, 范围: None, 轴线: None
- 名称: hip_1, 类型: 旋转关节, 范围: -30 30, 轴线: 0 0 1
- 名称: ankle_1, 类型: 旋转关节, 范围: 30 70, 轴线: -1 1 0
- 名称: hip_2, 类型: 旋转关节, 范围: -30 30, 轴线: 0 0 1
- 名称: ankle_2, 类型: 旋转关节, 范围: -70 -30, 轴线: 1 1 0
- 名称: hip_3, 类型: 旋转关节, 范围: -30 30, 轴线: 0 0 1
- 名称: ankle_3, 类型: 旋转关节, 范围: -70 -30, 轴线: -1 1 0
- 名称: hip_4, 类型: 旋转关节, 范围: -30 30, 轴线: 0 0 1
- 名称: ankle_4, 类型: 旋转关节, 范围: 30 70, 轴线: 1 1 0> 标签
    for joint in root.findall(".//joint"):
        joint_info = {
            "name": joint.get("name"),
            "type": joint.get("type", "hinge"),  # 默认是 'hinge'（旋转关节）
            "axis": joint.get("axis"),          # 关节轴线
            "range": joint.get("range")         # 获取运动范围
        }
        joints.append(joint_info)

    # 提取关键帧中的关节角度
    for keyframe in root.findall(".//keyframe/key"):
        key_name = keyframe.get("name")
        qpos = keyframe.get("qpos")
        if qpos:
            keyframes.append({
                "name": key_name,
                "qpos": [float(angle) for angle in qpos.split()]  # 转换为浮点数
            })

    return joints, keyframes

# 打印关节信息和关键帧信息
def print_results(joints, keyframes):
    print("关节数量:", len(joints))
    print("\n关节信息:")
    for i, joint in enumerate(joints):
        joint_type = "滑动关节" if joint["type"] == "slide" else "旋转关节"
        print(f"- 名称: {joint['name']}, 类型: {joint_type}, 范围: {joint['range']}, 轴线: {joint['axis']}")

    print("\n关键帧信息:")
    for keyframe in keyframes:
        print(f"- 关键帧名称: {keyframe['name']}")
        for i, angle in enumerate(keyframe['qpos']):
            print(f"  关节 {joints[i]['name']}: {angle:.4f} 弧度/米")

# 主程序
if __name__ == "__main__":
    # 替换为你的 XML 文件路径
    xml_file_path = "panda.xml"

    # 解析 XML 文件
    joints, keyframes = parse_mujoco_xml(xml_file_path)

    # 打印结果
    print_results(joints, keyframes)
```

输出结果
```
关节数量: 9

关节信息:
- 名称: joint1, 类型: 旋转关节, 范围: None, 轴线: None
- 名称: joint2, 类型: 旋转关节, 范围: -1.7628 1.7628, 轴线: None
- 名称: joint3, 类型: 旋转关节, 范围: None, 轴线: None
- 名称: joint4, 类型: 旋转关节, 范围: -3.0718 -0.0698, 轴线: None
- 名称: joint5, 类型: 旋转关节, 范围: None, 轴线: None
- 名称: joint6, 类型: 旋转关节, 范围: -0.0175 3.7525, 轴线: None
- 名称: joint7, 类型: 旋转关节, 范围: None, 轴线: None
- 名称: finger_joint1, 类型: 滑动关节, 范围: 0 0.04, 轴线: 0 1 0
- 名称: finger_joint2, 类型: 滑动关节, 范围: 0 0.04, 轴线: 0 1 0

关键帧信息:
- 关键帧名称: home
  关节 joint1: 0.0000 弧度/米
  关节 joint2: 0.0000 弧度/米
  关节 joint3: 0.0000 弧度/米
  关节 joint4: -1.5708 弧度/米
  关节 joint5: 0.0000 弧度/米
  关节 joint6: 1.5708 弧度/米
  关节 joint7: -0.7853 弧度/米
  关节 finger_joint1: 0.0400 弧度/米
  关节 finger_joint2: 0.0400 弧度/米
```

# 如何使用
可选：设置控制增益
## 设置位置增益
```python
franka.set_dofs_kp(
    kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local = dofs_idx,
)
```
## 设置速度增益
```python
franka.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local = dofs_idx,
)
```
## 设置安全的力范围
```python
franka.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    dofs_idx_lofranka.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    dofs_idx_local = dofs_idx,
)cal = dofs_idx,
)
```
注意这些API通常需要两个值集作为输入：要设置的实际值和相应的自由度索引。大多数与控制相关的API遵循这种约定。

## 硬重置
```python
# 硬重置
for i in range(150):
    if i < 50:
        franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
    elif i < 100:
        franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
    else:
        franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

    scene.step()
```

## PD控制
可以实现速度控制和力控制
```python
# PD控制
for i in range(1250):
    if i == 0:
        franka.control_dofs_position(
            np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 250:
        franka.control_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 500:
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    elif i == 750:
        # 用速度控制第一个自由度，其余的用位置控制
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
            dofs_idx[1:],
        )
        franka.control_dofs_velocity(
            np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
            dofs_idx[:1],
        )
    elif i == 1000:
        franka.control_dofs_force(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    # 这是根据给定控制命令计算的控制力
    # 如果使用力控制，它与给定的控制命令相同
    print('控制力:', franka.get_dofs_control_force(dofs_idx))

    # 这是自由度实际经历的力
    print('内部力:', franka.get_dofs_force(dofs_idx))

    scene.step()
```
# 代码汇总
```python
import numpy as np

import genesis as gs

########################## 初始化 ##########################
gs.init(backend=gs.gpu)

########################## 创建场景 ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

########################## 实体 ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/franka_emika_panda/panda.xml',
    ),
)
########################## 构建 ##########################
scene.build()

jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

############ 可选：设置控制增益 ############
# 设置位置增益
franka.set_dofs_kp(
    kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local = dofs_idx,
)
# 设置速度增益
franka.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local = dofs_idx,
)
# 设置安全的力范围
franka.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    dofs_idx_local = dofs_idx,
)
# 硬重置
for i in range(150):
    if i < 50:
        franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
    elif i < 100:
        franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
    else:
        franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

    scene.step()

# PD控制
for i in range(1250):
    if i == 0:
        franka.control_dofs_position(
            np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 250:
        franka.control_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 500:
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    elif i == 750:
        # 用速度控制第一个自由度，其余的用位置控制
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
            dofs_idx[1:],
        )
        franka.control_dofs_velocity(
            np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
            dofs_idx[:1],
        )
    elif i == 1000:
        franka.control_dofs_force(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    # 这是根据给定控制命令计算的控制力
    # 如果使用力控制，它与给定的控制命令相同
    print('控制力:', franka.get_dofs_control_force(dofs_idx))

    # 这是自由度实际经历的力
    print('内部力:', franka.get_dofs_force(dofs_idx))

    scene.step()
```