# Assignment #C: 202505114 Mock Exam

Updated 1518 GMT+8 May 14, 2025

2025 spring, Complied by <mark>马凱权 元培</mark>



> **说明：**
>
> 1. **⽉考**：AC?<mark>（5）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>



## 1. 题目

### E06364: 牛的选举

http://cs101.openjudge.cn/practice/06364/

思路：

AC,代码是结束后改成如下的，一开始是用dict完成，属于是原子弹炸蚂蚁了，其实就是sort的使用

代码：

```python
n, k = map(int, input().split())  
cows = []  
for i in range(n):
    a, b = map(int, input().split())  
    cows.append((a, b, i + 1))  


cows.sort(key=lambda x: x[0], reverse=True)

second_round_cows = cows[:k]

second_round_cows.sort(key=lambda x: x[1], reverse=True)

print(second_round_cows[0][2])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-14%20170735.png?raw=true)



### M04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

思路：

按题目的提示即可完成，就是能入栈就入栈不能就出栈，出的元素到达n个就计数，然后再试看其他方式

比如 （1，2，3）→出（3，2，1） 其实可以改成dfs(stack[1:],push,pop+[stack[0]]) 变成顺序出的

然后 （1，2） 出 2 进3 出3 出 1 （2，3，1）以此类推

代码：

```python
cnt=0
def dfs(stack,push,pop):
    global cnt
    if len(pop)==n:
        cnt+=1
        return
    if push<n:
        dfs(stack+[push+1],push+1,pop)
    if stack:
        dfs(stack[:-1],push,pop+[stack[-1]])

n=int(input())

dfs([],0,[])
print(cnt)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-14%20171041.png?raw=true)



### M05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

思路：

AC,按照题目描述，创造9个队列（使用deque，不一定需要），然后就依照给的牌号分队并输出，同理花色的分队也是如此，只是花色的分队需要在数字分队之上处理，因为数字分队后是小到大，那么花色分队同样花色的才能是小到大的数字排，最后花色分队的结果就是最终排序结果

代码：

```python
from collections import deque
n=int(input())

s=input().split()
queue_nums=[deque() for _ in range(9)]


for nums in s:
    num=int(nums[1])-1
    queue_nums[num].append(nums)

first_order=[]
for i in range(9):
    queue=queue_nums[i]
    first_order.extend(queue)
    print(f"Queue{i+1}:{' '.join(queue_nums[i])}")

#print(queue_nums)
flower_order={'A':0,'B':1,'C':2,'D':3}
queue_flower=[deque() for _ in range(4)]
for nums in first_order:
    flower=nums[0]
    flower_idx=flower_order[flower]
    queue_flower[flower_idx].append(nums)
#print(queue_flower)

final_result=[]
for i in range(4):
    flower=['A','B','C','D'][i]
    queue=queue_flower[i]
    print(f"Queue{flower}:{' '.join(queue)}")
    final_result.extend(queue)

print(' '.join(final_result))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-14%20194958.png?raw=true)



### M04084: 拓扑排序

http://cs101.openjudge.cn/practice/04084/

思路：

AC,但是上网查了拓扑排序是什么，其实就是入度为0中最小的开始“删除”，所以需要用到heapq来维持最小栈，利用有向邻接表方式记录关系，也要记录每个点的入度，然后就是入度为0的入最小栈并开始“删除”

代码：

```python
mport heapq

v,a=map(int,input().split())
degrees=[[] for _ in range(0,v+1)]
in_degree=[0]*(v+1)
for _ in range(a):
    U,V=map(int,input().split())
    degrees[U].append(V)
    in_degree[V]+=1
#print(degrees)
stack=[]
for i in range(1,v+1):
    if in_degree[i]==0:
        heapq.heappush(stack,i)
result=[]
while stack:
    u=heapq.heappop(stack)
    result.append(str(u))
    for v in degrees[u]:
        in_degree[v]-=1
        if in_degree[v]==0:
            heapq.heappush(stack,v)
ans=[]
for i in result:
    ans.append('v'+str(i))
print(' '.join(ans))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-14%20195357.png?raw=true)



### M07735:道路

Dijkstra, http://cs101.openjudge.cn/practice/07735/

思路：

NAC，这题有点麻烦，在于需要考虑不同费用情况下的多个可能路径，就是可能先以远距去到某城市再更快地达到目的地，因此需要注意这点，改成在距离方面的比较是某花费下达到的距离，也就是要记录多条路径可以到达某城市；另外这题需要用到最小栈使得去下个城市的长度是最小的

代码：

```python
import heapq

def dijkstra(graph, n, k):

    dis = [[float('inf')] * (k + 1) for _ in range(n + 1)]
    dis[1][0] = 0

    heap = [(0, 0, 1)]

    while heap:
        distance, cost, city = heapq.heappop(heap)

        if city == n:
            return distance

        if distance > dis[city][cost]:
            continue

        for next_city, length, toll in graph[city]:
            new_distance = distance + length
            new_cost = cost + toll

            if new_cost <= k and new_distance < dis[next_city][new_cost]:
                dis[next_city][new_cost] = new_distance
                heapq.heappush(heap, (new_distance, new_cost, next_city))

    return -1


K=int(input())
N=int(input())
R=int(input())

graph=[[] for _ in range(N+1)]
for _ in range(R):
    s,d,l,t=map(int,input().split())
    graph[s].append((d,l,t))

ans=dijkstra(graph,N,K)
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-15%20010850.png?raw=true)



### T24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

思路：

AC，首先设定好树，并且分配号左右（2*i+1<n:左；2*i+2<n:右），利用dfs形式深度探索，对于每个节点判断取该节点的值还是左右子节点的值最佳，一层一层往上回溯

代码：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def find(root):
    def dfs(node):
        if not node:
            return(0,0)
        left = dfs(node.left)
        right = dfs(node.right)
        take_node_val=node.val+left[1]+right[1]
        not_take_node_val=max(left[0],left[1])+max(right[0],right[1])
        return (take_node_val, not_take_node_val)
    res=dfs(root)
    return max(res)

n=int(input())
values=list(map(int,input().split()))
nodes=[TreeNode(val) for val in values]
for i in range(n):
    if 2*i+1<n:
        nodes[i].left=nodes[2*i+1]
    if 2*i+2<n:
        nodes[i].right=nodes[2*i+2]
root=nodes[0]
ans=find(root)
print(ans)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-05-14%20195816.png?raw=true)



## 2. 学习总结和收获

这次月考题目意外的没那么难，本以为这次是要上机考前最后一次考试，会比较难，但完成下来倒感觉还好，而且我晚了十分钟开始，在剩下20+分钟时剩下最后两题，由于第五题题目长且一时之间想不到，就做了第六题，第六题说实话借鉴了笔记的节点class的设立，比较难的是拓扑排序，单看题目描述根本没办法理解，第一题一开始花了点时间搞了字典，之后发现这简直太麻烦了

最近开始复习期末，会开始着眼于整理cheatsheet，估计会速刷每日选做，尽量搜集各种题目题解，进行分类然后记录象征性的题目，至少能一开始把树写好啊之类的，发现自己对于简单的题也是有时会卡着，比如队列的设立等等

愿机考顺利~

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>