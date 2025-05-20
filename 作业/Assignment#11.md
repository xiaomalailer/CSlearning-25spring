# Assignment #B: ͼΪ��

Updated 2223 GMT+8 Apr 29, 2025

2025 spring, Complied by <mark>��PȨ Ԫ��</mark>





## 1. ��Ŀ

### E07218:�׸�������ٯ�Ļ���

bfs, http://cs101.openjudge.cn/practice/07218/

˼·��

�ܵ��͵�bfs��Ŀ����time����step�������⣬���������·������bfsģ������ȥ����

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-30%20095942.png?raw=true)




### M3532.���ͼ��·�������Բ�ѯI

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

˼·��

���ȼ��趼��ͨ�������ж���0�㣩��Ȼ��������ڶ��߲�����<=maxDiff��������ô�͵���һ�㣬������ͨ�����ֻ��Ҫ�ж�queries�в��ҵ����������Ƿ���ͬһ�㼴��

��Ȼ�Զ��׼�Ҳ�еڶ�������������һ��ʼ������ͨ��������ڶ���������������ô����ͨ����һ�㣩

���룺

��һ��
```python
class Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:

        flag=[0]*n
        for i in range(1,n):
            flag[i]=flag[i-1]
            
            if nums[i]-nums[i-1]>maxDiff:
                flag[i]+=1 #����
        
        ans=[flag[u]==flag[v] for u,v in queries]
        return(ans)
```

�ڶ���
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
    flag[find(y)]=flag[find(x)] #��ͨ

for i,j in enumerate(nums):
    if i>0 and j-nums[i-1]<=maxDiff:
        union(i,i-1)

ans=[find(x)==find(y) for x,y in queries]
```

�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-30%20103107.png?raw=true)



### M22528:����ĵ��ַ���

binary search, http://cs101.openjudge.cn/practice/22528/

˼·��

������ֲ��ң�b�ķ�Χ��1-10e9����ȡ�м�ֵ�������㿴�������Ƿ���60%���ϣ����򻹿�������Сb(��ĿҪ����Сb�����������b

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-01%20110830.png?raw=true)



### Msy382: ����ͼ�л� 

dfs, https://sunnywhy.com/sfbj/10/3/382

˼·��

����������ͼ�ļ�¼��ʽ��¼ÿ��������¸����㣬������dfs���������ʽһ�������ص�ͬһ�����ɻ�����visited��¼ÿһ��״̬�����Ѿ�����1�������ھ�����0���������㡿��δ������-1�������¸�����δ�����ͼ���������һ��������һ����Ϊ�Ѿ�������������·�ϾͿɳɻ�

�Բ�������Ϊ����

1 0

0 3

3 2

2 1

��0������->3 -> 2 -> 1 ->0 ��0������·������һ�㣬�ɳɻ���

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-01%20115131.png?raw=true)



### M05443:������ӣ��

Dijkstra, http://cs101.openjudge.cn/practice/05443/

˼·��

�ѵ�����dijkstra����Ҫ�õ�heapq��ȷ��ÿ��ѡ��ǰ������С�Ľڵ㣬

����С��explore��˵����Dijkstra�㷨����̰�Ĳ��ԣ�ÿ��ѡ��ǰ�����������Ľڵ㣬��ͨ���ýڵ�������ھӽڵ����̾��롣ͨ�������ظ���һ���̣����յõ�����㵽���нڵ�����·����

��defaultdict�ȹ�����ͼ���ٴ���ÿ��Ŀ��·��


�����ڵ㣺Uenokouen

���ߵ���Ginza (35), Shinjukugyoen (85)

������£�
Ginza = 35��Shinjukugyoen = 85

��ǰ�ڵ㣺Ginza��35��
���ߵ���Sensouji (80), Uenokouen (35)

Sensouji = 35 + 80 = 115

��ǰ�ڵ㣺Shinjukugyoen��85��
���ߵ���Sensouji (40) �� �¾��� = 85 + 40 = 125����115�󣬲����£�

��ǰ�ڵ㣺Sensouji��115��
���ߵ���Meijishinguu (60) �� �¾��� = 115 + 60 = 175

��ǰ�ڵ㣺Meijishinguu��175��
���ߵ���Yoyogikouen (35) �� �¾��� = 175 + 35 = 210

����ٷ������������·�����

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-01%20144628.png?raw=true)



### T28050: ��ʿ����

dfs, http://cs101.openjudge.cn/practice/28050/

˼·��

�ӵ�ǰ���ӳ���

���������ȫ������ �� �ɹ�

�ҳ����кϷ�������

ÿ��������

���Կ�

����ɹ� , ���سɹ�

���ʧ�� , �����ⲽ������һ��·

ȫ��ʧ�� ,�Ǿ� ����

�����Ͼ���dfs��ֻ��˵Ϊ�˿�һЩ������һ������һ��������ѡ���ҳ���ѡ�����ٿ��ܵ����ԣ����û����һ�������������TLE��

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-01%20155738.png?raw=true)




## 2. ѧϰ�ܽ���ջ�

����Ϊ�����һ�Ƶ�����ֱ��51�ſ�ʼ����ҵ����ҵ�г�����С��explore���Ǹо�ֻ�ܸ��𰸣�����˼·��������е���Ӳ�����Ǹ�������ʹ��gpt����ɻ���ΪҲ������Ҫ�𰸸���Ҫ˼·��

dikjstra�㷨էһ�����ᣬ����ԭ��������heapq��һ�ַ�ʽ��

�������Ŀ��඼��ģ���⣬������һ�����µ�ģ�壬���Է���cheatsheet���ο�

��Ȼ51����û��ȥ�������Σ�������ͷ����Ҫæ����ҵ����ܶ࣬��֪���ܲ��ܳ��ʱ�䣬�ƻ���ʼ����cheatsheet

<mark>���������ҵ��Ŀ��Լ򵥣��з�Ѱ�Ҷ������ϰ��Ŀ���硰����2025springÿ��ѡ������LeetCode��Codeforces����ȵ���վ�ϵ���Ŀ��</mark>