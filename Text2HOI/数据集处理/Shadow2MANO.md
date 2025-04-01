### hand_pose_MANO
1 (whether annotate or not, 0: not annotate 1: annotate) + 3 translation values + 48 pose values + 10 shape values + 1 + 3 + 48 + 10 (right hand) <br>(这个3实际上是在48里面的)
First 59 numbers belong to the left hand. Next 59 numbers belong to the right hand 

MANO手的构建：
```python
out = hand_layer(  
    global_orient=hand_pose[:, :3],  
    hand_pose=hand_pose[:, 3:48],  
    betas=torch.zeros((duration, 10)).to(hand_pose.device)  
)
```
第一个参数global orient，就是腕部的旋转参数
第二个参数hand_pose，是手的每个关节的描述
第三个参数betas（形状参数），是PCA提取整个模型之后的结果，用于构建不同大小长短的人手，一般直接用全零值
```python
output = MANOOutput_C(  
    **super_output,   
    joints_w_tip=joints_w_tip,   
    skeletons=SKELETONS,  
    skeletons_w_tip=SKELETONS_W_TIP,  
)
```
这是调用build之后输出的，最好转换shadowhand之后能够得到这些信息
![[Pasted image 20250401195718.png]]


Shadowhand数据集中包含的东西：
3+3+22
手腕位移，手腕旋转，shadowhand的22个关节的关节角
其中手腕旋转是欧拉角的xyz表示