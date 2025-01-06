#installation #LuisaRender 
首先根据官方教程安装，我用的是ubuntu22.04 python 3.10 前面基本没有问题
https://genesis-doc-zh.readthedocs.io/zh-cn/latest/user_guide/overview/installation.html

问题出现在安装可选环境的  光线追踪渲染器 的安装
https://github.com/Genesis-Embodied-AI/Genesis/issues/42
第一个问题是`pip install genesis-world`包括genesis本身的仓库中并没有教程提到的`/ext/LuisaRender`
所以需要在/ext目录下
`git clone --recursive https://github.com/LuisaGroup/LuisaRender.git`
然后再进行官方教程中的步骤，其中遇到一些问题

`imgui.h`找不到，`_framebuffer`报错，Compilation Errors in `nfor.cpp`
在这个帖子中有详细解决方案
https://github.com/Genesis-Embodied-AI/Genesis/issues/42

安装较为简单，但是建议在有显卡的linux环境下安装，windows下不推荐
尝试了服务器上安装然后把图像传回PC，但是非常卡顿，效果不好
不过可以使用genesis的录制功能然后把mp4文件传回来看


推荐安装版本：ubuntu 20.04+   python3.9+   cuda 12.0+
以及安装最好有sudo的权限，如果没有会有很多报错无法处理
