# Assignment #6: 回溯、树、双向链表和哈希表

Updated 1526 GMT+8 Mar 22, 2025

2025 spring, Complied by <mark>马凱权 元培</mark>



## 1. 题目

### LC46.全排列

backtracking, https://leetcode.cn/problems/permutations/

思路：

利用递归（又似dfs）的方式完成全排列，如果正在排列的数组未达足够长度，则递归放置下个数字，且因为不能重复，所以需要引入used的判断

具体例子：[1,2,3]

[]+1+2+3 =[1,2,3] 返回到[1] → [1]+3+2=[1,3,2] 接着[2,1,3],[2,3,1]。。。



代码：

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(nums,n,length,path,used,ans):
            if length==n:
                ans.append(path)
                return
            for i in range(n):
                if not used[i]:
                    used[i]=True
                    backtrack(nums,n,length+1,path+[nums[i]],used,ans)
                    used[i]=False

        n=len(nums)
        used=[False for i in range(n)]
        ans=[]
        backtrack(nums,n,0,[],used,ans)
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-24%20173055.png?raw=true)




### LC79: 单词搜索

backtracking, https://leetcode.cn/problems/word-search/

思路：

用dfs（backtrack），就好像探索迷宫，从字开头开始找起（上下左右）

代码：

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        move=[(1,0),(0,1),(-1,0),(0,-1)]
        def dfs(x,y,length):
            if length==len(word):
                return True
            temp=board[x][y]
            board[x][y]='0'
            for dx,dy in move:
                nx=x+dx
                ny=y+dy
                if 0<=nx<len(board) and 0<=ny<len(board[0]) and board[nx][ny]==word[length]:
                    if dfs(nx,ny,length+1):
                        return True
            board[x][y]=temp
            return False
        

        for m in range(len(board)):
            for n in range(len(board[0])):
                if board[m][n]==word[0] and dfs(m,n,1):
                    return True
                    exit()

        return False
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

!{Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-24%20220638.png?raw=true)



### LC94.二叉树的中序遍历

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

思路：

所谓中序遍历，就是往树的左边一直往下遍历直到遇到“最左”【即无左根】再将其加入答案，再搜索其右根（当成新树）；之后再返回上一根，重复。

可以看题解的动画展示，会很清楚

![SET](https://assets.leetcode-cn.com/solution-static/94/9.png)

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans=[]
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            ans.append(root.val)
            dfs(root.right)
        
        dfs(root)
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-24%20222154.png?raw=true)



### LC102.二叉树的层序遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

思路：

bfs,也就是用了deque的形式，有点像队列（或者就是），首先把第一行入队，再出队加进答案，再把下一行入队、出队，并进行遍历以把接下来的一行全部入队、出队以此类推

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue=deque()
        queue.append(root)
        ans=[]
        while queue:
            tmp = []
            for _ in range(len(queue)):
                node = queue.popleft()
                tmp.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            ans.append(tmp)

        return ans
        
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-24%20224816.png?raw=true)



### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

思路：

dfs，但为了让时间复杂度小，可以先找出所有能找出的回文字符串，比如每个字就是一个，规则就是头尾相同且里边也是回文（s[i+1][j-1])，比如abba是回文的条件就是头尾相同且中间bb回文，然后再通过dfs把回文字符串加入答案

例：s="aab"

先找出回文：

palindro[0][0] = True:  'a'

palindro[1][1] = True: 'a'

palindro[2][2] = True: "b"

palindro[0][1] = True: "aa"

再dfs：

dfs(0): 选 "a" → dfs(1)

dfs(1): 选 "a" → dfs(2)

dfs(2): 选 "b" → dfs(3)  → 记录 ["a", "a", "b"]

回溯：
dfs(1): 选 "aa" → dfs(2)

dfs(2): 选 "b" → dfs(3) → 记录 ["aa", "b"]


代码：

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        ans=[]
        out=[]
        def dfs(length):
            if length==len(s):
                out.append(ans[:])
                return

            for j in range(length,len(s)):
                if palindro[length][j]:
                    ans.append(s[length:j+1])
                    dfs(j+1)
                    ans.pop()


        palindro=[[True]*len(s) for _ in range(len(s))]

        for i in range(len(s)-1,-1,-1):
            for j in range(i+1,len(s)):
                palindro[i][j]= (s[i]==s[j]) and palindro[i+1][j-1]
        dfs(0)
        return(out)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-24%20232534.png?raw=true)




### LC146.LRU缓存

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

思路：

依照题解的提示完成，题解中用书本的抽放来对比十分清楚，dummy是一个同时对书头书尾作处理的节点，next是书头prev是书尾，因此要放到书头或删除书尾都需要dummy帮忙

![](https://pic.leetcode.cn/1696039105-PSyHej-146-3-c.png)

代码：

```python
class ListNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dummy = ListNode() 
        self.dummy.prev = self.dummy
        self.dummy.next = self.dummy
        self.key_to_node = dict()

    def get(self, key: int) -> int:
        if key not in self.key_to_node:
            return -1
        node = self.key_to_node[key]
        self.remove(node)
        self.push_front(node)
        return node.value

    def remove(self, x: ListNode) -> None:
        x.prev.next = x.next
        x.next.prev = x.prev

    def push_front(self, x: ListNode) -> None:
        x.prev = self.dummy
        x.next = self.dummy.next
        self.dummy.next.prev = x
        self.dummy.next = x

    def put(self, key: int, value: int) -> None:
        node = self.key_to_node.get(key)  
        if node:  
            node.value = value
            self.remove(node)
            self.push_front(node)
        else:
            node = ListNode(key, value)  
            self.key_to_node[key] = node
            self.push_front(node)
            if len(self.key_to_node) > self.capacity:  
                back_node = self.dummy.prev
                del self.key_to_node[back_node.key]
                self.remove(back_node)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-25%20113923.png?raw=true)



## 2. 学习总结和收获

这次题目几题dfsbfs题目类似也不会太难，有一定公式可循。二叉树的题只是形式看起来不同但是内容处理还是bfsdfs或递归的形式，理解和完成起来不难，中序那题倒是要花点时间理解题目，有几题看了点提示完成，最后一题倒是蛮难的，对照上次的url来写好像也不太行，直到看到抽书放书才晓得大概怎么做，但还是看了题解。

树、链表（listnode）类的题目都有点难，难在格式上不知道怎么写和链表之间节点关系怎么处理，而且用到的函数还不止题目提供的那些，上机考试要考则可能难倒我，不过做了几次感觉还是有一定公式和条理

最近越发忙碌，选做也好久没做，这次作业题好像好早就发也是等到周二才看，4月要到了，期中考要来了，希望能平衡好，不落后进度。

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>