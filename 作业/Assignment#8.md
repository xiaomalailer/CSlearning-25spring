# Assignment #8: 树为主

Updated 1434 GMT+8 Apr 8, 2025

2025 spring, Complied by <mark>马凱权 元培</mark>





## 1. 题目

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：

是二分法，但其实一想，更是dfs，有两种思路，一种是左右往中间延申(中间根节点左右为叶子节点），中间是第一个节点，左边是其左边的根（从远处开始），右边同理，因此可以以dfs形式往下深进行左右划分；另一种形式答案只是变成由中间（最靠近左右为非叶子节点）往远处延申

代码：

```python
#第一种答案【从左右远处往中延申】
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def binary_sort(left, right):
            if left > right:
                return None
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = binary_sort(left, mid - 1)
            root.right = binary_sort(mid + 1, right)
            return root

        return binary_sort(0, len(nums) - 1)

##第二种【根从中间向左右延申】
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def binary_sort(left, right):
            if left == right:
                return None
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = binary_sort(left, mid)
            root.right = binary_sort(mid + 1, right)
            return root

        return binary_sort(0, len(nums))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20160918.png?raw=true)



### M27928:遍历树

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：

参考了王铭健学长的答案，但是是用dict{}而没有用defaultdict，原理相同，现把根与子节点关系处理好，然后重点是要把整棵树根节点找出来，方法就是用集合减法找出唯一不成为子节点的根节点，不过难想到的是如何排序输出，王学长的方法很好，就是将根节点和其子节点排序，较小的如果是子节点，如果度数不为0，则向下搜索，如果是根节点就直接输出，

对第二个测试数据：

{10: [3, 1], 7: [], 9: [2], 2: [10], 3: [7], 1: []} 

9 #根节点

2 #sorted=>[2,9] , sorted[2,10] print"2"

1 #sorted[1,3,10] print"1" 

3 #sorted[3,7] print"3"

7 #print"7"

10 #print"10"

9 #print"9"

代码：

```python
n = int(input())
tree = {}
children_set = set()
parent_list = []

for _ in range(n):
    parts = list(map(int, input().split()))
    node = parts[0]
    parent_list.append(node)
    if len(parts) > 1:
        children = parts[1:]
        tree[node] = children
        children_set.update(children)
    else:
        tree[node] = []

#print(tree)
root = (set(parent_list) - children_set).pop()
#print(root)
def traverse(node):
    parent_children=sorted(tree[node]+[node])
    for x in parent_children:
        if x==node:
            print(node)
        else:
            traverse(x)

traverse(root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20174009.png?raw=true)



### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：

仍然是dfs形式向下进行遍历，多带一个num记录总值，因为是二叉树，分左右进行运算即可，叶节点标志：if not root.left and not root.right:

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(root, num):
            if not root:
                return 0
            total = num * 10 + root.val
            if not root.left and not root.right:
                return total
            return dfs(root.left, total) + dfs(root.right, total)

        return dfs(root, 0)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20180323.png?raw=true)



### 22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/24729/

思路：

给定前序和中序表达，先重建树，再转换为后续表达，重建树情况如下：首先确定前序第一个字母为整棵树的首个根节点，，那么就以他为中心重建左边和右边的子树，左边的树，根据中序表达式中首根节点左边位置确定，又可分为前序和中序的左树（中序中左边数到首根节点（n个字母）为左树中序表达，前序中首节点后n个数为左树前序表达）；右树同样如此，按照递归/dfs就可重建；再换去后续表达即可


代码：

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.left=None
        self.right=None

def rebuild(preorder,inorder):
    if not preorder or not inorder:
        return None
    root=Node(preorder[0])
    root_idx_inorder=inorder.index(preorder[0])
    root.left=rebuild(preorder[1:1+root_idx_inorder],inorder[:root_idx_inorder])
    root.right=rebuild(preorder[1+root_idx_inorder:],inorder[root_idx_inorder+1:])
    return root

def postorder(root):
    if root is None:
        return ''
    return postorder(root.left) + postorder(root.right) + root.val

while True:
    try:
        preorder=input()
        inorder=input()
        root=rebuild(preorder,inorder)
        print(postorder(root))
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20215652.png?raw=true)



### T24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：

首先是要对树进行一个重建，利用类和stack的方法，建起每个节点，同时根据括号匹配，遇到左括号，代表遇到一个子节点处，接下来的字母就是子节点，遇到右括号代表遇到叶节点，需要弹出目前正在处理的节点；前序后序排列都以递归形式；前序是一直往深处加，比较清楚，后序是找到叶节点后再往回加，再往下个节点深入到叶节点再往回加

后序：(A->B->E->NONE) E → B →  F → G → C  → I → H → D  → A

代码：

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.children = []

def parse_tree(s):
    stack = []
    root = None
    node = None

    for char in s:
        if char.isalpha():
            node = Node(char)
            if stack:
                stack[-1].children.append(node)
            else:
                root = node
        elif char == '(':
            if node:
                stack.append(node)
                node = None
        elif char == ')':
            if stack:
                node = stack.pop()
        elif char == ',':
            continue

    return root

def preorder(node):
    if not node:
        return ''
    res = node.val
    for child in node.children:
        res += preorder(child)
    return res

def postorder(node):
    if not node:
        return ''
    res = ''
    for child in node.children:
        res += postorder(child)
    res += node.val
    return res

s = input().strip()

root = parse_tree(s)

print(preorder(root))
print(postorder(root))



```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20212237.png?raw=true)




### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：

利用pairwise进行两两配对，enumerate加上index，配对好进行加和后利用heapq进行最小堆，那么pop出来就是当下选中的一组最小和，

对于此和，分别判断加和后对于递减数量的影响，然后分别判断与之前的一个数和之后的一个数（下下个数）的递减影响，对于每个数它们的之前个数left和后个数right，利用left=[-1,0,1,...],right=[1,2,3...]记录，进行完判断后，除非加和的数是在边界（左/右），否则可以和左/右再一次加和再放入最小堆判断

但是因为只是更改数组中某个位置的数，需要删除“nums[i+1]”，并让它在后续计算中不再被访问到，即它左边数的下个数变成它右边的，它右边的左边变成它左边，听起来很拗口，但其实就是链表的一种实现


代码：

```python
import heapq
from itertools import pairwise

#nums=[5,2,3,1]
#nums=list(map(int,input().split()))
n=len(nums)
h=[]
dec=0
for i,(x,y) in enumerate(pairwise(nums)):
    if x>y:
        dec+=1
    h.append((x+y,i))
heapq.heapify(h)
#print(h)

left=list(range(-1,n))
right=list(range(1,n+1))
cnt=0

while dec:
    cnt+=1
    while right[h[0][1]]>=n or h[0][0]!=nums[h[0][1]]+nums[right[h[0][1]]]:
        heapq.heappop(h)
    s,i=heapq.heappop(h)

    next_nums_idx=right[i]

    if nums[i]>nums[next_nums_idx]:
        dec-=1

    before_nums_idx=left[i]
    if before_nums_idx>=0:
        if nums[before_nums_idx]>nums[i]:
            dec-=1
        if nums[before_nums_idx]>s:
            dec+=1
        heapq.heappush(h,(nums[before_nums_idx]+s,before_nums_idx))

    after_nums_idx=right[next_nums_idx]
    if after_nums_idx<n:
        if nums[after_nums_idx]<nums[next_nums_idx]:
            dec-=1
        if nums[after_nums_idx]<s:
            dec+=1
        heapq.heappush(h,(nums[after_nums_idx]+s,i))

    nums[i]=s
    right[left[next_nums_idx]]=right[next_nums_idx]
    left[right[next_nums_idx]] = left[next_nums_idx]
    right[next_nums_idx] = n
print(cnt)


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20234900.png?raw=true)



## 2. 学习总结和收获

这次题目看似难但是做完or参考答案后觉得还好，存粹是树的写法太吓人，容易在各种root、node、tree之间迷失，加上一些格式的写法的确需要练习，所以看起来难了点，树的题目有时也有点“复杂”，把事情说得复杂；树大部分题目（或者几乎都是）需要用到递归/dfs思路进行遍历，这一点比较需要在纸上进行模拟

整体而言，一道题一道题练习下来倒会逐渐对树上手，导致我前面几题花的时间反而更多，后面就还好

这周是期中周，算是很忙，忙着复习，不过今天刚好在教室复习，就过来实体课一边上一边完成数算作业，才惊觉错过了很多笔试重点


<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>