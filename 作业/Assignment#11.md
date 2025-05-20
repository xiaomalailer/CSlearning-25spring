# Assignment #B: 图为主

Updated 2223 GMT+8 Apr 29, 2025

2025 spring, Complied by <mark>马凱权 元培</mark>





## 1. 题目

### E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

思路：

很典型的bfs题目，把time换成step会更好理解，就是找最短路径，把bfs模板套下去就行

代码：

```python
from collections import deque


def bfs(maze,start,end,R,C):
    move=[(-1,0),(1,0),(0,-1),(0,1)]
    queue=deque()
    queue.append([start,0])
    visited=set()
    visited.add(start)

    while queue:
        (x,y),time=queue.popleft()
        if (x,y) == end:
            return time
        for dx,dy in move:
            nx,ny=x+dx,y+dy
            if 0<=nx<R and 0<=ny<C and (nx,ny) not in visited and maze[nx][ny]!='#':
                visited.add((nx,ny))
                queue.append([(nx,ny),time+1])
    return 'oop!'

T=int(input())
for _ in range(T):
    R,C=map(int,input().split())
    maze=[]
    start=None
    end=None
    for i in range(R):
        row=input()
        maze.append(list(row))
        if 'S' in row:
            start=(i,row.index('S'))
        if 'E' in row:
            end=(i,row.index('E'))
    ans=bfs(maze,start,end,R,C)
    print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-30%20095942.png?raw=true)




### M3532.针对图的路径存在性查询I

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

思路：

首先假设都连通，（所有都在0层），然后如果相邻二者不满足<=maxDiff条件，那么就得上一层，代表不联通，最后只需要判断queries中查找的两个数字是否在同一层即可

当然显而易见也有第二种做法，假设一开始都不联通，如果相邻二者满足条件，那么就联通（降一层）

代码：

第一种
```python
class Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:

        flag=[0]*n
        for i in range(1,n):
            flag[i]=flag[i-1]
            
            if nums[i]-nums[i-1]>maxDiff:
                flag[i]+=1 #隔开
        
        ans=[flag[u]==flag[v] for u,v in queries]
        return(ans)
```

第二种
```python
n=4
nums = [2,5,6,8]
maxDiff = 2
queries = [[0,1],[0,2],[1,3],[2,3]]

flag=list(range(n))
def find(x):
    if flag[x]!=x:
        flag[x]=find(flag[x])
    return flag[x]

def union(x,y):
    flag[find(y)]=flag[find(x)] #连通

for i,j in enumerate(nums):
    if i>0 and j-nums[i-1]<=maxDiff:
        union(i,i-1)

ans=[find(x)==find(y) for x,y in queries]
```

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-30%20103107.png?raw=true)



### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：

经典二分查找，b的范围是1-10e9，先取中间值代入运算看优秀率是否有60%以上，有则还可以再缩小b(题目要求最小b），无则调高b

代码：

```python
def find_min_b(scores):
    left=1
    right=10**9
    result=0
    while left<=right:
        mid=(left+right)//2
        a=mid/10**9
        excellent=0
        for x in scores:
            adjust_score=a*x+1.1**(a*x)
            if adjust_score>=85:
                excellent+=1
        if excellent/len(scores)>=0.6:
            result=mid
            right=mid-1
        else:
            left=mid+1
    return result


scores=[float(x) for x in input().split()]
print(find_min_b(scores))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-01%20110830.png?raw=true)



### Msy382: 有向图判环 

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：

先是以有向图的记录方式记录每个顶点的下个顶点，接着以dfs深度搜索形式一旦搜索回到同一点就算成环，以visited记录每一点状态，有已经过（1），正在经过（0）【出发点】，未经过（-1），若下个顶点未经过就继续搜索，一旦出现下一顶点为已经搜索且在搜索路上就可成环

以测试数据为例：

1 0

0 3

3 2

2 1

从0出发，->3 -> 2 -> 1 ->0 （0是这条路径其中一点，可成环）

代码：

```python

def dfs(u,degrees,visited):
    visited[u] = 0
    for v in degrees[u]:
        if visited[v] == -1:
            if dfs(v,degrees,visited):
                return True
        elif visited[v] == 0:
            return True
    visited[u] = 1
    return False

n,m=map(int,input().split())
degrees=[[] for _ in range(n)]
for _ in range(m):
    u,v=map(int,input().split())
    degrees[u].append(v)
visited=[-1]*n
flag=True
for i in range(n):
    if visited[i]==-1:
        if dfs(i,degrees,visited):
            flag=False
            print('Yes')
            exit()

if flag==True:
    print('No')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-01%20115131.png?raw=true)



### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：

难点在于dijkstra，需要用到heapq来确保每次选择当前距离最小的节点，

按照小北explore的说法：Dijkstra算法基于贪心策略，每次选择当前距离起点最近的节点，并通过该节点更新其邻居节点的最短距离。通过不断重复这一过程，最终得到从起点到所有节点的最短路径。

用defaultdict先构建好图，再处理每组目标路径


出发节点：Uenokouen

可走到：Ginza (35), Shinjukugyoen (85)

距离更新：
Ginza = 35，Shinjukugyoen = 85

当前节点：Ginza（35）
可走到：Sensouji (80), Uenokouen (35)

Sensouji = 35 + 80 = 115

当前节点：Shinjukugyoen（85）
可走到：Sensouji (40) → 新距离 = 85 + 40 = 125（比115大，不更新）

当前节点：Sensouji（115）
可走到：Meijishinguu (60) → 新距离 = 115 + 60 = 175

当前节点：Meijishinguu（175）
可走到：Yoyogikouen (35) → 新距离 = 175 + 35 = 210

最后再反推完整个最短路径输出

代码：

```python
from collections import defaultdict
import heapq

def dijkstra(degrees, start, end):
    distances = {node: float('inf') for node in degrees}
    prev = {node: None for node in degrees}
    distances[start] = 0
    heap = [(0, start)]

    while heap:
        cur_distance, cur_node = heapq.heappop(heap)
        if cur_distance > distances[cur_node]:
            continue
        if cur_node == end:
            break
        for neighbor, weight in degrees[cur_node].items():
            distance = cur_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                prev[neighbor] = cur_node
                heapq.heappush(heap, (distance, neighbor))


    path = []
    current = end

    while prev[current] is not None:
        path.insert(0, current)
        current = prev[current]
    path.insert(0, start)
    return path, distances[end]

P=int(input())
places=[input().strip() for _ in range(P)]

Q=int(input())
degrees=defaultdict(dict)
for _ in range(Q):
    a,b,degree=input().split()
    degree=int(degree)
    degrees[a][b]=degree
    degrees[b][a]=degree
#print(degrees)
R=int(input())
target=[input().split() for _ in range(R)]

for start, end in target:
    if start == end:
        print(start)
        continue
    path, total_distance = dijkstra(degrees, start, end)

    output = path[0]
    for i in range(1, len(path)):
        output += f"->({degrees[path[i - 1]][path[i]]})->{path[i]}"
    print(output)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-01%20144628.png?raw=true)



### T28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：

从当前格子出发

如果已走满全部格子 → 成功

找出所有合法的跳法

每个跳法：

试试看

如果成功 , 返回成功

如果失败 , 撤掉这步，换下一条路

全部失败 ,那就 回溯

基本上就是dfs，只是说为了快一些，把下一步的下一步可能性选择都找出，选择最少可能的先试（如果没有这一步，亲身体验会TLE）

代码：

```python

k_moves=[(2,1),(1,2),(-1,2),(-1,-2),(-2,1),(-2,-1),(2,-1),(1,-2)]

def knights_tour(n,board,x,y,total):
    if total==n*n:
        return True

    next_moves=[]
    for dx,dy in k_moves:
        nx,ny=x+dx,y+dy
        if 0<=nx<n and 0<=ny<n and board[nx][ny]==-1:
            count = 0
            for ddx, ddy in k_moves:
                tx, ty = nx + ddx, ny + ddy
                if 0 <= tx < n and 0 <= ty < n and board[tx][ty] == -1:
                    count += 1
            next_moves.append((count, nx, ny))
    next_moves.sort()

    for _,nx,ny in next_moves:
        board[nx][ny]=total
        if knights_tour(n,board,nx,ny,total+1):
            return True
        board[nx][ny]=-1
    return False

n=int(input())
sr,sc=map(int,input().split())
board=[[-1 for _ in range(n)] for _ in range(n)]
board[sr][sc]=0
if knights_tour(n,board,sr,sc,1):
    print("success")
else:
    print("fail")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-01%20155738.png?raw=true)




## 2. 学习总结和收获

这周为了最后一科的期中直到51才开始做作业，作业有尝试问小北explore但是感觉只能给答案，解释思路方面可能有点生硬，还是更倾向于使用gpt解答疑惑（因为也不是想要答案更多要思路）

dikjstra算法乍一看不会，但是原来就是用heapq的一种方式？

上面的题目大多都是模板题，就是有一个大致的模板，可以放入cheatsheet作参考

虽然51假期没出去哪里旅游，但是手头上需要忙的作业读物很多，不知道能不能抽出时间，计划开始整理cheatsheet

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>