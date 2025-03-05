# Assignment #3: 惊蛰 Mock Exam

Updated 1641 GMT+8 Mar 5, 2025

2025 spring, Complied by <mark>同学的姓名、院系</mark>



> **说明：**
>
> 1. **惊蛰⽉考**：<mark>AC4</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>



## 1. 题目

### E04015: 邮箱验证

strings, http://cs101.openjudge.cn/practice/04015



思路：

基本上按照题目指示做if else判断，比较容易出错的是忘记考虑“.@”的情况

代码：

```python
while True:
    try:
        s=input()
    except EOFError:
        break

    if s.count('@')!=1:
        print('NO')
        continue
    if s[0] in ['@','.'] or s[-1] in ['@','.']:
        print('NO')
        continue
    if s.find('@.')!=-1 or s.find('.@')!=-1:
        print('NO')
        continue
    a=s.find('@')
    b=s.find('.',a+1)
    print('YES'if b!=-1 else 'NO')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-05%20205818.png?raw=true)




### M02039: 反反复复

implementation, http://cs101.openjudge.cn/practice/02039/



思路：

按照几个字一行进行分割，偶数（从0开始则为奇数）是倒转的则需要转回来，再讲行列转置即可

代码：

```python
n = int(input())
s = input()
s_g = [s[i:i + n] for i in range(0, len(s), n)]


for i in range(len(s_g)):
    if i % 2 != 0:
        s_g[i] = s_g[i][::-1]

ans = [''] * n
for group in s_g:
    for j in range(len(group)):
        ans[j] += group[j]

print(''.join(ans))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-05%20205829.png?raw=true)




### M02092: Grandpa is Famous

implementation, http://cs101.openjudge.cn/practice/02092/



思路：

题目有点拗口，但其实给出的数据是每行代表一次排名，只要出现在一次排名的编号就可以加一分，按照这个逻辑就可以为每个编号计分，再按照分数排序，排序后找出第二高分的输出即可

代码：

```python
while True:
    N,M=map(int,input().split())
    if N==0 and M==0:
        break
    ranking=[list(map(int,input().split())) for _ in range(N)]
    score={}
    for row in ranking:
        for num in row:
            score[num]=score.get(num,0)+1
    score=sorted(score.items(),key=lambda x:x[1],reverse=True)
    second_score=[num for num,sco in score if sco==score[1][1]]
    second_score.sort()
    print(*second_score)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-05%20205840.png?raw=true)




### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/



思路：

类似于radar那题，垃圾能被以其为中心的爆炸范围内放置的炸弹处理，因此算这些范围内能处理的垃圾数并且可以叠加，再算最大能处理的垃圾数及其爆炸位点数量

代码：

```python
d=int(input())
n=int(input())
mos=[[0]*1025 for _ in range(1025)]
for _ in range(n):
    a,b,c=map(int,input().split())
    for i in range(max(a-d,0),min(1025,a+d+1)):
        for j in range(max(b-d,0),min(1025,b+d+1)):
            mos[i][j]+=c
ans=0
max_c=0
for i in range(1025):
    for j in range(1025):
        if mos[i][j]>max_c:
            max_c=mos[i][j]
            ans=1
        elif mos[i][j]==max_c:
            ans+=1
print(ans,max_c)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-05%20205801.png?raw=true)




### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/



思路：

月考时一直WA，应该是移动顺序问题，要改成字典序大小去移动，看了题解但觉得题解很奇怪，感觉像是把p、q反了似的，而且定A1为起始位点，这点我暂时无法论证对错，因此我尝试修改成每个位点都试，直到找到解

代码：

```python
move = [(-1, -2), (1, -2), (-2, -1), (2, -1), (-2, 1), (2, 1), (-1, 2), (1, 2)]
flag=False
def dfs(x, y, path,step):
    global visited,p,q,flag
    if flag:
        return
    if step == p * q:
        flag = True
        print(path)
        return
    for dx, dy in move:
        nx, ny = x + dx, y + dy
        if 0 <= nx < p and 0 <= ny < q and not visited[nx][ny]:
            visited[nx][ny] = True
            dfs(nx,ny,path+chr(ny+ord('A'))+str(nx+1),step+1)
            visited[nx][ny] = False



n = int(input())
for m in range(1, n + 1):
    p, q = map(int, input().split())

    visited = [[False] * (q) for _ in range(p)]
    flag = False
    print(f"Scenario #{m}:")
    for i in range(p):
        for j in range(q):
            if flag:
                break
            visited[i][j] = True
            path=chr(j+ord('A'))+str(i+1)
            dfs(i, j, path, 1)
            visited[i][j] = False

    if not flag:
        print("impossible")
    print()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-05%20205747.png?raw=true)



### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/



思路：

此处参考了题解，是一个一个序列处理，首先对第一个和第二个序列排序，第一个序列中前n个小的数与第二个序列最小的数组合成新的小序列，如果第二个序列中还有未用过的元素，可以继续扩展（弹出一个又加入一个，保持着一定大小），再对接下来的序列重复，可避免每个都进行运算，只需迭代得出小序列就行

代码：

```python
import heapq

T=int(input())

for _ in range(T):
    m,n=map(int,input().split())

    current_s=sorted(map(int,input().split()))
    for _ in range(m-1):
        next_s=sorted(map(int,input().split()))
        min_heap=[(current_s[i]+next_s[0],i,0) for i in range(n)]
        heapq.heapify(min_heap)
        result=[]
        for _ in range(n):
            c_sum,i,j=heapq.heappop(min_heap)
            result.append(c_sum)
            if j+1<len(next_s):
                heapq.heappush(min_heap,(current_s[i]+next_s[j+1],i,j+1))
        current_s=result
    print(*current_s)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-05%20231128.png?raw=true)




## 2. 学习总结和收获

这次月考有点摆，比较悠闲地做，前面两道都不难，尤其第一道一看到就觉得很熟悉，第三道英文题倒是在理解题意花了些时间，第四道垃圾炸弹虽然做过但是还是有难度，所幸能类比最近完成的radarinstallation，觉得很相似；第五道题倒是试了很多次都WA，最后发现很可能是字典序问题，最后一道难度大，看了题解，是heapq方式，对这个不太熟悉，原来最小栈是那么用

最近是每天练习两道每日选做，都赶的上进度，也许是还没涉及到新的算法如树、图，暂时觉得还能应付，但是寒假的每日选做倒是没完成

下次月考可以试看上机房，自己的电脑还是太好用了

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>