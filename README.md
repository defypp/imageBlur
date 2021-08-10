# imageBlur
imageBlur functions Peel off from OpenCV

##TODO
+ CUDA support
+ gussian filter
+ bilateralblur
+ ...

##   快速中值滤波

### 原理

不同的kernel大小对应的不同的滤波方式3X3 和 5X5 采用核展开的形式进行比较;同时根据图像大小，设置动态kernel阈值，分别使用两种方式进行滤波，算法复杂度分别为O(m), O(1),其中m表示kernelsize

+   __3×3 原理__
```
I0 I1 I2
I3 I4 I5
I6 I7 I8
```
step1： 分别对三行进行比较排序，得到

I0 < I1 < I2
I3 < I4 < I5
I6 < I7 < I8

step2:  分别对第一列和第三列进行部分排序，即比较两次

（1）将第一列最大数交换到I6所在位置
I0 > I3,swap(I0,I3) 
I3 > I6,swap(I3,I6)
此时由于I0/I3 在第一列表明已经小于两个数，同时小于I6,表明，再次小于三个数，所以不可能成为中位数

（2）将第三列最小数交换到I2所在位置
I5 < I2, swap(I2,I5)
I8 < I2, swap(I8,I2)
由于I2 < I5, I2 < I8, 由于2/5/8在第三列，同时大于两个数，则I5/I8
大于5个数，不可能成为中位数

step3： 斜对角元素排列
对I2,I4,I6进行全排序，则I4位置所在的数为中位数


+ __5*5原理__

+ __medianBlur8uOm 算法复杂度O(m)__

构建两个直方图进行中值查找，分别为zone0[4][16] 和zone1[4][16*16],其中4表示图像的最大通道，16表示[0,256]的16次划分，16×16表示[0,256]的256次划分
列遍历查找过程
    图像边界replicated kernel/2 
    for x in width
        for y in height
            x 偶数情况
                [x,y]为中心， [m/2, m] 范围初始化zone（y=0，边界默认采用replicated，所以每个边界点的值 +（m/2+1); zone中必须记录满m^2个数）
                通过调整sptr_top / sptr_bottom, 动态的增加和删减zone中的值并通过累计zone0和zone1的方式来获取中值 最多计算32次

        
+   __medianBlur8u01 算法复杂度O(1)__

O(1)复杂度实现示意图
![O(1)中值滤波图示](./assets/mf.png)

paper: Median Filtering in Constant Time 

在复杂度O(m)的算法中，每横向移动一个像素，都需要在直方图最右侧和最左侧加上和减去像素，并没有充分利用行像素的不变性。考虑保留行像素的算法核心流程如下：
```
Input:Image X of size m X n, kernel_radius r
Output: Image Y of same size
    Init kernel histogram H(256bins，统计m^2像素)
    Init column hitograms h1/h2/../hn, h1统计第一列 列方向m个像素的直方图统计信息
    for i = 1 to m do
        for j = 1 to n do
            Remove X(i-r-1, j+r) from h(j+r)
            Add X(i+r,j+r) to h(j+r)
            H <- H + h(j+r) - h(j-r-1)
        end for
    end for
```

8位灰度图的OP如下
```
    “+”  1    更新top right histogram
    “-”  1    更新bottom right histogram
    “+”  256  kernel histogram 新增 最右侧 h
    “-”  256  kernel histogram 减去 最左侧 h
    “cmp” "+" 查找中值 平均情况 127 “+” 128 “cmp” 
```

针对上述OP，可以便捷的采用并行化加速提高运算速度和降低执行次数



