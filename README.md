# VQE 3D VIS

以VQE的优化过程为线索，可视化优化的轨迹和参数空间（PCA降维）。基于MindQuantum构建。

两个横轴分别是PCA出来的两个主成分，纵轴则为loss函数值。

生成的是GIF图像。

# 快速测试

在你喜欢的虚拟环境中运行：
```
bash mkdir.sh
pip install -r requirements.txt
```

然后运行：
```
python sample_lih_uccsd.py
```
即可生成lih的优化轨迹图。
