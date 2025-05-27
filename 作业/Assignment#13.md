# Assignment #D: 图 & 散列表

Updated 2042 GMT+8 May 20, 2025

2025 spring, Complied by <mark>马凱权 元培</mark>




## 1. 题目

### M17975: 用二次探查法建立散列表

http://cs101.openjudge.cn/practice/17975/

<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



思路：

hashtable二次探查法主要是先根据m建立全空table，根据散列函数 %m后在table的位置如果已经是该数字或没有数字就填入该数字，如果有其他数字，就以下列方式


di = 1², -1², 2², -2², 3², ..., +k², -k² (k ≤ m/2)

h+di 再%m 直到找到位置填入

代码：

```python
import sys

def hash(num_list,m):
    table=[None]*m
    positions=[]
    for key in num_list:
        h=key%m
        cur=table[h]
        if cur==None or cur==key:
            table[h]=key
            positions.append(h)
        else:
            sign=1
            i=1
            while True:
                pos=(h+sign*i*i)%m
                if table[pos]==None or table[pos]==key:
                    table[pos]=key
                    positions.append(pos)
                    break
                sign*=-1
                if sign==1:
                    i+=1

    return positions

input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
result=hash(num_list,m)
print(*result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-20%20222802.png?raw=true)



### M01258: Agri-Net

MST, http://cs101.openjudge.cn/practice/01258/

思路：

简单说就是给你一堆农场和它们之间的光缆距离，你要选出一些线，把所有农场连起来，而且总长度要尽量短；所以就从第一个农场开始不断更新之间的dist数组让下一个距离最短


代码：

```python
def check_min(n, farm):
    visited = [False] * n
    dist = [float('inf')] * n
    dist[0] = 0
    total = 0
    for _ in range(n):
        u = -1
        for i in range(n):
            if not visited[i] and (u == -1 or dist[i] < dist[u]):
                u = i
        visited[u] = True
        total += dist[u]
        for v in range(n):
            if not visited[v] and farm[u][v] < dist[v]:
                dist[v] = farm[u][v]
    return total

while True:
    try:
        N = int(input())
        farm = []
        while len(farm) < N:
            farm += [list(map(int, input().split()))]
        ans = check_min(N, farm)
        print(ans)
    except EOFError:
        break

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-21%20000646.png?raw=true)




### M3552.网络传送门旅游

bfs, https://leetcode.cn/problems/grid-teleportation-traversal/

思路：

用bfs方式走迷宫，但是现在有传送门，先把传送门都记下来，遇到传送门，则把传送门位置连通即可

代码：

```python
from collections import defaultdict,deque

matrix = ["..#DDF","#.H.F#","D.#A#.","#BF...","BFD.#A","CEEG.B",".FA.FG","F.E#.E"]

move=[(0,1),(1,0),(-1,0),(0,-1)]

def bfs(matrix):
    if matrix[-1][-1]=='#':
        return -1
    m,n=len(matrix),len(matrix[0])

    pos=defaultdict(list)
    for i,row in enumerate(matrix):
        for j,c in enumerate(row):
            if c.isupper():
                pos[c].append((i,j))

    dis=[[float('inf')]*n for _ in range(m)]

    dis[0][0]=0

    q=deque([(0,0)])

    while q:
        x,y=q.popleft()
        d=dis[x][y]
        if x==m-1 and y==n-1:
            return d
        c=matrix[x][y]
        if c in pos:
            for px,py in pos[c]:
                if d<dis[px][py]:
                    dis[px][py]=d
                    q.appendleft((px,py))
            del pos[c]

        for dx,dy in move:
            nx,ny=x+dx,y+dy
            if 0<=nx<m and 0<=ny<n and matrix[nx][ny]!='#' and d+1<dis[nx][ny]:
                dis[nx][ny]=d+1
                q.append((nx,ny))

    return -1


print(bfs(matrix))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-21%20005955.png?raw=true)



### M787.K站中转内最便宜的航班

Bellman Ford, https://leetcode.cn/problems/cheapest-flights-within-k-stops/

思路：

以动态规划方式更新fly[a][j] = min(fly[a][j], fly[a- 1][i] + cost)，代表a次航班到达j地的最小花费等于a-1次航班从i出发的花费+到i到j花费，然后就以从1到k+2（包括起点终点）的经过地点算得最少花费

代码：

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        
        fly = [[float("inf")] * n for _ in range(k + 2)]
        fly[0][src] = 0  
        
        for a in range(1, k + 2):
            for i, j, cost in flights:
                
                fly[a][j] = min(fly[a][j], fly[a- 1][i] + cost)
        
        
        ans = min(fly[a][dst] for a in range(1, k + 2))
        return -1 if ans == float("inf") else ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-21%20013353.png?raw=true)



### M03424: Candies

Dijkstra, http://cs101.openjudge.cn/practice/03424/

思路：

x_B ≤ x_A + c， 此时从 A → B 建一条边，权值为 c，题目还要求在满足所有“不能比别人多太多”的限制下，让 flymouse 拿得尽可能多

难点在于如何理解最大差值是最短路径，是为了保证所有路径都令人满意，x_A到x_B的上界也就是各边权值和得取最小的，因此使用dijkstra


代码：

```python

import heapq

def dijikstra(n,m):

    candi=[float('inf')]*(n+1)
    candi[1]=0

    queue=[False]*(n+1)
    heap=[(0,1)]

    while heap:
        d,u=heapq.heappop(heap)
        if d>candi[u]:
            continue

        for v,w in graph[u]:
            new_d=candi[u]+w
            if new_d<candi[v]:
                candi[v]=new_d
                heapq.heappush(heap,(candi[v],v))
    return candi[n]

N,M=map(int,input().split())
graph=[[]for _ in range(N+1)]
for _ in range(M):
    a,b,c=map(int,input().split())
    graph[a].append((b,c))
print(dijikstra(N,M))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-21%20101443.png?raw=true)




### M22508:最小奖金方案

topological order, http://cs101.openjudge.cn/practice/22508/

思路：

用拓扑排序，首先建造邻接表，a战胜b，则b输给了a，a入度+1，最后先取入度为0，最后一名，它的胜者全部+1分（1元），然后再通过删除入度找出倒数第二以此类推

代码：

```python
from collections import defaultdict,deque

n,m=map(int,input().split())

pk=defaultdict(list)
vic=[0]*n
for _ in range(m):
    a,b=map(int,input().split())
    pk[b].append(a)
    vic[a]+=1

total=[100]*n
queue=deque()
for i in range(n):
    if vic[i]==0:
        queue.append(i)

while queue:
    u=queue.popleft()

    for v in pk[u]:
        vic[v]-=1
        total[v]=max(total[v],total[u]+1)

        if vic[v]==0:
            queue.append(v)

print(sum(total))
'''
1 0
2 0
3 0
4 1 2 3

0 - 100
1 2 3 101*3=303

4 -102
100+303+102=505
'''
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-21%20105719.png?raw=true)




## 2. 学习总结和收获

这次作业很多题看似是模板题（bfs、dijkstra、topo等）但是难点在于要搞懂题目要求去写，因此耗费很长时间debug，知道要以某种方式完成，可是一直过不了测试数据，其实就是自己对题目理解不够，这的确很难办，希望考试自己头脑能清醒点，否则就是距离ac咫尺而不达

还有差不多两周就要上机考了，说实话准备不够充分，希望保底能AC3，目标是AC4/5，本身逻辑能力没那么好，只能多刷点题记下来格式

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>