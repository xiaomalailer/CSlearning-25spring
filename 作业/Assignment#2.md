# Assignment #2: 深度学习与大语言模型

Updated 2204 GMT+8 Feb 25, 2025

2025 spring, Complied by <mark>马凱权 元培学院</mark>





## 1. 题目

### 18161: 矩阵运算

matrices, http://cs101.openjudge.cn/practice/18161



思路：

矩阵运算，先判断能不能完成，即A列与B行是否等同（乘法）及，形成的新矩阵A行B列是否同C矩阵，再进行运算；乘法运算是行*列，即A第一行乘B第一列、A第一行×B第二列...再到A下一行，因此有三个变量（A的行、A的列 B的列）

代码：

```python
A=[]
B=[]
C=[]
r_a,c_a=map(int,input().split())
for _ in range(r_a):
    A.append(list(map(int,input().split())))

r_b,c_b=map(int,input().split())
for _ in range(r_b):
    B.append(list(map(int,input().split())))

r_c,c_c=map(int,input().split())
for _ in range(r_c):
    C.append(list(map(int,input().split())))

if c_a!=r_b or r_a!=r_c or c_b!=c_c:
    print('Error!')
    exit()
D=[[0]*c_b for _ in range(r_a)]
for i in range(r_a):
    for j in range(c_b):
        for k in range(c_a):
            D[i][j]+=A[i][k]*B[k][j]
        D[i][j]+=C[i][j]
for ans in D:
    print(*ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-26%20092359.png?raw=true)




### 19942: 二维矩阵上的卷积运算

matrices, http://cs101.openjudge.cn/practice/19942/




思路：

由于把一堆for放在一起难看，所以开个函数进行运算，卷积矩阵不需要变化，但是原矩阵需要变，当处在结果矩阵的i、j时，原矩阵应也从i、j开始运算，且不超过p、q


代码：

```python

def juan(x,y):
    ans=0
    for i in range(p):
        for j in range(q):
            ans+=matrix_1[i+x][j+y]*matrix_2[i][j]
    return ans

m,n,p,q=map(int,input().split())
matrix_1=[]
matrix_2=[]
for i in range(m):
    matrix_1.append(list(map(int,input().split())))

for i in range(p):
    matrix_2.append(list(map(int,input().split())))

matrix_3=[[0]*(n+1-q) for _ in range(m+1-p)]

for i in range(m+1-p):
    for j in range(n+1-q):
        matrix_3[i][j]=juan(i,j)
for row in matrix_3:
    print(*row)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-26%20095841.png?raw=true)




### 04140: 方程求解

牛顿迭代法，http://cs101.openjudge.cn/practice/04140/

请用<mark>牛顿迭代法</mark>实现。

因为大语言模型的训练过程中涉及到了梯度下降（或其变种，如SGD、Adam等），用于优化模型参数以最小化损失函数。两种方法都是通过迭代的方式逐步接近最优解。每一次迭代都基于当前点的局部信息调整参数，试图找到一个比当前点更优的新点。理解牛顿迭代法有助于深入理解基于梯度的优化算法的工作原理，特别是它们如何利用导数信息进行决策。

> **牛顿迭代法**
>
> - **目的**：主要用于寻找一个函数 $f(x)$ 的根，即找到满足 $f(x)=0$ 的 $x$ 值。不过，通过适当变换目标函数，它也可以用于寻找函数的极值。
> - **方法基础**：利用泰勒级数的一阶和二阶项来近似目标函数，在每次迭代中使用目标函数及其导数的信息来计算下一步的方向和步长。
> - **迭代公式**：$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $ 对于求极值问题，这可以转化为$ x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)} $，这里 $f'(x)$ 和 $f''(x)$ 分别是目标函数的一阶导数和二阶导数。
> - **特点**：牛顿法通常具有更快的收敛速度（尤其是对于二次可微函数），但是需要计算目标函数的二阶导数（Hessian矩阵在多维情况下），并且对初始点的选择较为敏感。
>
> **梯度下降法**
>
> - **目的**：直接用于寻找函数的最小值（也可以通过取负寻找最大值），尤其在机器学习领域应用广泛。
> - **方法基础**：仅依赖于目标函数的一阶导数信息（即梯度），沿着梯度的反方向移动以达到减少函数值的目的。
> - **迭代公式**：$ x_{n+1} = x_n - \alpha \cdot \nabla f(x_n) $ 这里 $\alpha$ 是学习率，$\nabla f(x_n)$ 表示目标函数在 $x_n$ 点的梯度。
> - **特点**：梯度下降不需要计算复杂的二阶导数，因此在高维空间中相对容易实现。然而，它的收敛速度通常较慢，特别是当目标函数的等高线呈现出椭圆而非圆形时（即存在条件数大的情况）。
>
> **相同与不同**
>
> - **相同点**：两者都可用于优化问题，试图找到函数的极小值点；都需要目标函数至少一阶可导。
> - **不同点**：
>   - 牛顿法使用了更多的局部信息（即二阶导数），因此理论上收敛速度更快，但在实际应用中可能会遇到计算成本高、难以处理大规模数据集等问题。
>   - 梯度下降则更为简单，易于实现，特别是在高维空间中，但由于只使用了一阶导数信息，其收敛速度可能较慢，尤其是在接近极值点时。
>

如果用一阶导数和二阶导数的商貌似不太行，应该是原函数和一阶导数的商，初始值试了1，2，3，4，5..都大差不差，其中1e-6是最大误差容忍值（其实是x_n和x_n-1的差值，当小于所设值则不计误差），100设为最大迭代次数

牛顿法的核心思想是用切线逼近曲线，然后用切线的零点作为下一个近似解

代码：

```python

def newton_new(x):
    for i in range(100):
       d=f_x(x)/f_x_2(x)
       x=x-d
       if abs(d)<1e-6:
           break
    return x


f_x=lambda x: x**3-5*(x**2)+10*x-80
f_x_2=lambda x:3*(x**2)-10*x+10
f_x_3=lambda x:6*x-10

x_0=1.0
print('{:.9f}'.format(newton_new(x_0)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-26%20104340.png?raw=true)




### 06640: 倒排索引

data structures, http://cs101.openjudge.cn/practice/06640/



思路：

用字典方便键值查询，所有使用setdefault把词编号记录好，如果单词出现在某个编号里就将该编号加入欲输出中


代码：

```python
N=int(input())
c={}
a=0
for _ in range(N):
    s=input().split()
    c.setdefault(a+1,s[1:])
    a=a+1
M=int(input())

for _ in range(M):
    ans = []
    b=input()
    for i in range(N):
        if b in c[i+1]:
            ans.append(i+1)
    print(*ans if ans else"NOT FOUND")

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-25%20230422.png?raw=true)




### 04093: 倒排索引查询

data structures, http://cs101.openjudge.cn/practice/04093/



思路：

这题因为用交集、并集做比较简单，所有就用set，must_have记录一定会出现的文档编号（第一个 1：直接赋值 ；后续 1：取交集 ），must_nothave记录一定不出现的文档编号（把不要的并起来），两者减去即得答案（差集）

代码：

```python
N=int(input())
find=[]
for i in range(N):
    s=input().split()
    find.append(set(map(int,s[1:])))

M=int(input())
for _ in range(M):
    a=list(map(int,input().split()))
    must_have=set()
    must_not_have=set()
    first=True
    for i in range(N):
        if a[i]==1:
            if first:
                must_have=find[i].copy()
                first=False
            else:
                must_have&=find[i]
        elif a[i]==-1:
            must_not_have|=find[i]
    valid_docs=sorted(must_have - must_not_have)
    if valid_docs:
        print(' '.join(map(str,valid_docs)))
    else:
        print("NOT FOUND")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-25%20224905.png?raw=true)




### Q6. Neural Network实现鸢尾花卉数据分类

在http://clab.pku.edu.cn 云端虚拟机，用Neural Network实现鸢尾花卉数据分类。

参考链接，https://github.com/GMyhf/2025spring-cs201/blob/main/LLM/iris_neural_network.md





## 2. 学习总结和个人收获

这次作业比较考对python语法的掌控，题目看得懂但就是代码比较费劲，比如矩阵相乘大家都会，但是for循环的设置仍要下点功夫，卷积那题也同样；解方程那题就是运用微分学到的牛顿法迭代，挺有意思

倒排其实还是不知道为什么取名作倒排，倒排索引比较清楚，但是查询那题仍需要些时间理解题目和实现

总结就是这次作业的题考验基础语法和语句，是很好的锻炼，毕竟经常都是：“嗯，应该这样做，但是怎么写？”，这种题目往往简单但又容易出错，实在可惜

个人目前仍跟得上每日选做，比起上个学期状态好了很多（可能是上学期是小白？），要月考了，可以试试水平！

*由于这学期有点忙，虽然觉得大模型很有趣，但恐无精力学习故缺做第六题（而且看起来好难理解/(ㄒoㄒ)/~~*

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>