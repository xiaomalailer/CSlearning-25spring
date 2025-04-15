# Assignment #7: 20250402 Mock Exam

Updated 1624 GMT+8 Apr 2, 2025

2025 spring, Complied by <mark>马凱权 元培</mark>



> **说明：**
>
> 1. **⽉考**：AC?<mark> 4 </mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>




## 1. 题目

### E05344:最后的最后

http://cs101.openjudge.cn/practice/05344/



思路：

AC,同约瑟夫问题，利用队列形式，把k前的人出队了再入队，第k个人则出队加入死亡名单


代码：

```python
n,k=map(int,input().split())
people=[i for i in range(1,n+1)]
queue=[]
for num in people:
    queue.append(num)
kill=[]
while len(queue)>1:
    for i in range(k-1):
        queue.append(queue.pop(0))
    kill.append(queue.pop(0))
print(*kill)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-02%20173539.png?raw=true)




### M02774: 木材加工

binary search, http://cs101.openjudge.cn/practice/02774/



思路：

AC，这种数字游戏得找一个在区间的值，一看就得用二分法，最小能分到的段1cm，最大就是最大木材长度，找个中间值，计算各个木材分成该长度后的段数之和大于要求还是小于要求，大于或等于要求代表该长度足够分出至少K段，可以再往更长的长度找，小于则如果按该长度分不够，需要减小长度


代码：

```python
N,K=map(int,input().split())
logs=[]
for i in range(N):
    logs.append(int(input()))
left=1
right=max(logs)
ans=0
while left<=right:
    mid=(left+right)//2
    cnt=0
    for a in logs:
        cnt+=a//mid
    if cnt>=K:
        ans=mid
        left=mid+1
    else:
        right=mid-1
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-02%20173616.png?raw=true)




### M07161:森林的带度数层次序列存储

tree, http://cs101.openjudge.cn/practice/07161/



思路：

AC，难度大，但在纸上推算能自己想出要用deque的popleft功能，遂按照这个思路，为方便理解，以给定的测试数据为推演：

对于这棵树：C 3 E 3 F 0 G 0 K 0 H 0 J 0

先整理好成[(C,3),(E,3),(F,0),(G,0),(K,0),(H,0),(J,0)]

开始重建，考虑到复杂，故制造重建函数

先把该片段变成deque类型，

把最高节点 C,3popleft出，

利用词典 tree记录每个degree不等于0的节点的子节点

利用node_queue记录现移动节点

首先对于C，3；代表有3个子节点，以queue.popleft记录进需要遍历的节点node_queue中(E,3),(F,0),(G,0)

再把这三个属于C的子节点加入C的“值”，C：[...]

然后遍历子节点，从(E,3)开始，按同理会有 E:['K','H','J']，再遍历KHJ时因为没有度数就不会加入字典序

总之最后会返回一个字典：[{'C': ['E', 'F', 'G'], 'E': ['K', 'H', 'J']}]

因为输出要“逆序”：以递归方式，C→E→K+H+J →K+H+J+E → K+H+J+E+F+G+C





代码：

```python
from collections import deque

def rebuild(tree_sequence):
    if not tree_sequence:
        return {}
    queue=deque(tree_sequence)
    root_val,root_degree=queue.popleft()
    tree={}
    tree[root_val]=[]
    node_queue=deque([(root_val,root_degree)])

    while queue and node_queue:
        cur_node,degree=node_queue.popleft()
        roots=[]
        for _ in range(degree):
            if not queue:
                break
            cur_root,cur_root_degree=queue.popleft()
            roots.append(cur_root)
            node_queue.append((cur_root,cur_root_degree))
        tree[cur_node]=roots
    return tree

def reorder_tree(tree,node,ans):
    for root in tree.get(node,[]):
        reorder_tree(tree,root,ans)
    ans.append(node)

n=int(input())
forest=[]
top_roots=[]
for _ in range(n):
    tree_i=input().split()
    tree=[(tree_i[i],int(tree_i[i+1])) for i  in range(0,len(tree_i),2)]
    new_tree=rebuild(tree)
    forest.append(new_tree)
    top_roots.append(tree[0][0])
print(forest)
ans=[]
for i in range(n):
    reorder_tree(forest[i],top_roots[i],ans)
print(' '.join(ans))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-02%20173519.png?raw=true)



### M18156:寻找离目标数最近的两数之和

two pointers, http://cs101.openjudge.cn/practice/18156/



思路：

AC，不难，题目还好心给了双指针提示，但有陷阱，在于如果没仔细审题，会忽略“如果存在多个解，则输出数值较小的那个”，也就是需要min（ans），因此需要判断现答案与目标绝对差值，和之前答案中何者差值小，如果差值同，就得看那个数字更小

代码：

```python
T=int(input())
S=list(map(int,input().split()))
S.sort()
left=0
right=len(S)-1
ans=S[left]+S[right]
while left<right:
    sum2=S[left]+S[right]
    if sum2==T:
        print(sum2)
        exit()
    if abs(sum2-T)<abs(ans-T):
        ans=sum2
    elif abs(sum2-T)==abs(ans-T):
        ans=min(ans,sum2)
    if sum2>T:
        right-=1
    elif sum2<T:
        left+=1
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-02%20173638.png?raw=true)



### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/



思路：

WA，前面题目花太久，事实上这题不难，而且一看就有思路，但是错在没有留意“不包括自己”这句话，还傻傻地一直提交

题目有说一直重复检验每个数是不是质数不可行，很容易联想到：那就一次过把所有质数找出来记录下来就行，那就是埃氏筛，去年T-primes的解法，事实上我不太记得怎么写，只能去找去年的记录（下次放进cheatsheet！），有了一定范围的质数就好办了，检查是不是质数+个位数是否为1


代码：

```python
import math

def sieve_of_eratosthenes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return is_prime


MAX_LIMIT = 10 ** 6
is_prime = sieve_of_eratosthenes(MAX_LIMIT)

T = int(input())
for case_num in range(1, T + 1):
    n = int(input())
    ans=[]
    for i in range(2,n):
        if i % 10==1 and is_prime[i]:
            ans.append(i)

    print(f"Case{case_num}:")
    if ans:
        print(" ".join(map(str,ans)))
    else:
        print("NULL")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-02%20173507.png?raw=true)




### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/



思路：

利用字典，ac中记录ac的题号，submissions中记载提交次数，再重新创造一个list，记录每个队伍ac数、交的次数和名字（此次序也是比较次序）因为ac数是要越多越好，但sort时不好直接逆序排列，那么就乘个负号排，之后再改正输出，输出注意要前12个

代码：

```python
M=int(input())

teams={}

for _ in range(M):

    name,question,result=input().split(',')
    if name not in teams:
        teams[name]={"ac":set(),'submissions':0}
    teams[name]['submissions']+=1
    if result=='yes' and question not in teams[name]['ac']:
        teams[name]['ac'].add(question)

team_list=[]
for name in teams:
    ac_cnt=len(teams[name]['ac'])
    submissions=teams[name]['submissions']
    team_list.append((-1*ac_cnt,submissions,name))
team_list.sort()

for rank,(ac_cnt,submissions,name) in enumerate(team_list[:12]):
    ac=-ac_cnt
    print(f"{rank+1} {name} {ac} {submissions}")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-03%20003144.png?raw=true)



## 2. 学习总结和收获

这次月考还是AC4，究其原因花太多时间在第三题，不断修正修改才能ac，其他题都还好，如果有时间第六题也不难，第五题没完成是因为漏看，这次整体难度不会太高，但感觉自己还有待进步

虽然每日选做停滞不前，但是幸好自己做做月考和作业感觉还在把握之中？也许是错觉，但的确没什么时间给数算，春天来了，期中到了，也有许多额外的活动要参加/(ㄒoㄒ)/~~

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>