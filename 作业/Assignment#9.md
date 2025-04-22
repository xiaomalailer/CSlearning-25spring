# Assignment #: Huffman & Fenwick

Updated 1034 GMT+8 Apr 15, 2025

2025 spring, Complied by <mark>马凱权 元培</mark>





## 1. 题目

### LC222.完全二叉树的节点个数

dfs, https://leetcode.cn/problems/count-complete-tree-nodes/

思路：

dfs思路往下收索遍历即可

看了解答有一个复杂度较低的方法蛮厉害的，就是分成左子树满右子树不满or右子树满左子树满情况，其中左子树满，右子树不满对应：左子树高度 == 右子树高度；左子树不满，右子树满对应：左子树高度 > 右子树高度

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        else :
            return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```

方法二参考解答：
```
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        # 满二叉树结点个数： 2^h - 1
        def get_height(node):
            if node is None:
                return 0
            
            left = get_height(node.left)
            right = get_height(node.right)
            
            return max(left, right) + 1
        
        if root is None:
            return 0
        
        left_height = get_height(root.left)
        right_height = get_height(root.right)

        if left_height == right_height: # 左子树是满的
            # 满二叉树 2^h - 1 然后加root
            return 2 ** left_height + self.countNodes(root.right)
        else:  # 右子树是满的
            return 2 ** right_height + self.countNodes(root.left)

```


代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-15%20163254.png?raw=true)



### LC103.二叉树的锯齿形层序遍历

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

思路：

bfs层层遍历，如果是偶数（根节点从0开始）层，则该层所以节点正常append（后进），如果是奇数，则需要都先进

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans=[]
        if root is None:
            return ans
        queue=deque([root])
        ans=[]
        while queue:
            t_queue=deque([])
            for _ in range(len(queue)):
                node=queue.popleft()
                if len(ans)%2==0:
                    t_queue.append(node.val)
                else:
                    t_queue.appendleft(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            ans.append(list(t_queue))
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-15%20170613.png?raw=true)




### M03720:文本二叉树（原第三题）

tree, http://cs101.openjudge.cn/practice/03720/

思路：

首先得重建树，关键代码我认为是while stack and stack[-1][1] != cur_level - 1:stack.pop() ，就是确定当前节点的层数-1是父节点，stack负责储存父节点，层数计算方法是当前行长-1（由i个-和1个字母构成），遇到*也照算节点，在进行前序后序中序时遇到带 *的需要跳过，前中后序排列代码易懂，不做赘述

代码：

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def build_tree(lines):
    if not lines:
        return None
    tree = []
    for line in lines:
        node_alpha = line.lstrip('-')
        level = len(line) - len(node_alpha)
        val = node_alpha.strip()
        tree.append((level, val))
    root = TreeNode(tree[0][1])
    stack = [(root, tree[0][0])]
    for i in range(1, len(tree)):
        cur_level, cur_val = tree[i]
        node = TreeNode(cur_val)
        while stack and stack[-1][1] != cur_level - 1:
            stack.pop()
        if stack:
            parent, _ = stack[-1]
            if parent.left is None:
                parent.left = node
            else:
                parent.right = node
        if node is not None:
            stack.append((node, cur_level))

    return root


def preorder(root):
    if root is None:
        return
    if root.val != '*':
        print(root.val, end='')
    preorder(root.left)
    preorder(root.right)

def postorder(root):
    if root is None:
        return
    postorder(root.left)
    postorder(root.right)
    if root.val != '*':
        print(root.val, end='')

def inorder(root):
    if root is None:
        return
    inorder(root.left)
    if root.val!='*':
        print(root.val, end='')

    inorder(root.right)
n = int(input())
for _ in range(n):
    lines = []
    while True:
        line = input().strip()
        if line == '0':
            break
        lines.append(line)
    #print(lines)
    root = build_tree(lines)
    preorder(root)
    print()
    postorder(root)
    print()
    inorder(root)
    print()

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-15%20181018.png?raw=true)



### M04080:Huffman编码树

greedy, http://cs101.openjudge.cn/practice/04080/

思路：

本质上就是两两最小的配对的总和，比如 1 1 3 5， 就是 1+1 =2 ，2再和 3 成 5，5再和5 =10 ，最小外部路径长度总和就是2+5+10

可视化些就是 [1,1,3,5] → [2,3,5] → [5,5] → [10]

建树来看也可以 

![huff](https://images2015.cnblogs.com/blog/610439/201612/610439-20161214230137323-1092491743.png)

代码：

```python
import heapq


def min_external_path_length(n, weights):
    if n == 1:
        return weights[0]

    heap = weights.copy()
    heapq.heapify(heap)
    total = 0

    while len(heap) > 1:
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        combined = first + second
        total += combined
        heapq.heappush(heap, combined)

    return total


n = int(input())
weights = list(map(int, input().split()))
print(min_external_path_length(n, weights))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-16%20083053.png?raw=true)

### M05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/

思路：

首先要明白创建树的规则，对于某个节点

1）左子树中的所有节点的值都小于该节点的值

2）右子树中的所有节点的值都大于该节点的值

所以造树，就是对于一个节点值，通过上述规则的比对进行插入

比如 51 45 59 86

对于 45 比51小 ：放到51 左子树

对于 59 比51大：放到51 右子树

对于 86 比51大：放到51 右子树，在51右子树中，比59大，放到59右子树

进行层递的输出，利用deque完成bfs的层次遍历

代码：

```python
from collections import deque


class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None

def insert(root,val):
    if root is None:
        return TreeNode(val)
    if val<root.val:
        root.left=insert(root.left,val)
    elif val>root.val:
        root.right=insert(root.right,val)
    return root

def order_to_print(root):
    if root is None:
        return []
    queue=deque([root])
    result=[]
    while queue:
        node=queue.popleft()
        result.append(str(node.val))
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

nums=list(map(int,input().split()))
once_nums=[]
seen=set()
for num in nums:
    if num not in seen:
        seen.add(num)
        once_nums.append(num)

root=None
for num in once_nums:
    root=insert(root,num)

print(' '.join(order_to_print(root)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-16%20083546.png?raw=true)

### M04078: 实现堆结构

手搓实现，http://cs101.openjudge.cn/practice/04078/

类似的题目是 晴问9.7: 向下调整构建大顶堆，https://sunnywhy.com/sfbj/9/7

思路：

堆本质上是完全二叉树，实现主要靠heappush和heappop，对于heappush，就是从最小叶节点往上比对父节点，只要能交换就一直向上交换；对于heappop，首先最小堆处理后根节点必最小，然后把最后一个数移到根节点，再往下进行比对保证最小堆，即如果左子节点存在且比当前节点小或右子节点存在比当前节点小，那么就要交换


代码：

```python

def heapify_up(heap,idx):
    parent=(idx-1)//2
    while idx>0 and heap[idx]<heap[parent]:
        heap[idx],heap[parent]=heap[parent],heap[idx]
        idx=parent
        parent=(idx-1)//2

def heapify_down(heap,idx):
    left=2*idx+1
    right=2*idx+2
    smallest=idx
    if left<len(heap) and heap[left]<heap[smallest]:
        smallest=left
    if right<len(heap) and heap[right]<heap[smallest]:
        smallest=right
    if smallest!=idx:
        heap[idx],heap[smallest]=heap[smallest],heap[idx]
        heapify_down(heap,smallest)

def heappush(heap,item):
    heap.append(item)
    heapify_up(heap,len(heap)-1)

def heappop(heap):
    if not heap:
        return None
    min_val=heap[0]
    heap[0]=heap[-1]
    heap.pop()
    heapify_down(heap,0)
    return min_val

n=int(input())
heap=[]
for _ in range(n):
    mission=input().split()
    type=int(mission[0])
    if type==1:
        s=int(mission[1])
        heappush(heap,s)
    elif type==2:
        print(heappop(heap))

```

直接实现
```python
import heapq

n=int(input())
heap=[]
for _ in range(n):
    mission=input().split()
    type=int(mission[0])
    if type==1:
        s=int(mission[1])
        heapq.heappush(heap,s)
    elif type==2:
        print(heapq.heappop(heap))

```

sy题解
```python
def adjust_heap(heap, low, high):
    i = low
    j = 2 * i + 1  
    while j <= high:
        if j + 1 <= high and heap[j + 1] > heap[j]:
            j += 1  
        if heap[j] > heap[i]:
            heap[i], heap[j] = heap[j], heap[i]
            i = j
            j = 2 * i + 1
        else:
            break

def heap_make(heap, n):
    for i in range(n // 2 - 1, -1, -1):
        adjust_heap(heap, i, n - 1)

n = int(input())
tree = list(map(int, input().split()))
heap_make(tree, n)
print(' '.join(map(str, tree)))
```


代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q51](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-16%20083723.png?raw=true)

![Q52](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-15%20234549.png?raw=true)


### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/

思路：

首先了解哈夫曼建树规则：

节点比较：先比权值，权值相同比字符集最小字符

合并时小节点作为左子节点

左边为0，右边为1

例：

      {c,g,d}:22
      /      \
    {c}:10    {g,d}:12
              /    \
          {g}:4    {d}:8

在这个例子中，c=0，g=10，d=11

所以 100 =gc ，110=dc 反之相同

所以在这里可以预先生成解码和编码表，在需要时可以快速编码解码，大致是这个思路，我让ai帮忙注解代码



代码：

```python
import heapq  
class HuffmanNode:
    def __init__(self, chars, weight, left=None, right=None):
        self.chars = chars    
        self.weight = weight  
        self.left = left     
        self.right = right    

    def __lt__(self, other):
        """定义节点比较规则，用于堆排序：
        1. 首先比较权重，权重小的节点更小
        2. 权重相同时，比较字符集中最小的字符"""
        if self.weight != other.weight:
            return self.weight < other.weight
        return min(self.chars) < min(other.chars)

def build_huffman(char_weight):
    """构建哈夫曼树：
    1. 初始化最小堆
    2. 不断合并权重最小的两个节点
    3. 返回最终的根节点"""
    heap = []
    # 初始化堆，为每个字符创建叶子节点
    for char, weight in char_weight.items():
        heapq.heappush(heap, HuffmanNode({char}, weight))  # 注意：使用集合存储单个字符
    
    # 合并节点直到只剩一个根节点
    while len(heap) > 1:
        left = heapq.heappop(heap)   # 取出权重最小的节点
        right = heapq.heappop(heap)  # 取出权重第二小的节点
        
        # 合并两个节点：字符集合并，权重相加
        merged_c = left.chars.union(right.chars)
        merged_w = left.weight + right.weight
        # 创建新节点，较小的节点作为左子节点
        merged_node = HuffmanNode(merged_c, merged_w, left, right)
        heapq.heappush(heap, merged_node)  # 将新节点放回堆中
    
    return heapq.heappop(heap)  # 返回最终的根节点

def build_code(root, path='', codebook=None):
    """递归构建编码字典（字符 -> 编码）：
    - 左路径添加'0'，右路径添加'1'
    - 到达叶子节点时记录字符的编码"""
    if codebook is None:
        codebook = {}
    
    # 叶子节点：存储字符到编码的映射
    if root.left is None and root.right is None:
        for char in root.chars:
            codebook[char] = path
    else:
        # 递归处理左右子树
        build_code(root.left, path + '0', codebook)
        build_code(root.right, path + '1', codebook)
    
    return codebook

def build_decode(root, path='', decodebook=None):
    """递归构建解码字典（编码 -> 字符）：
    - 结构与build_code类似，但记录的是编码到字符的映射"""
    if decodebook is None:
        decodebook = {}
    
    # 叶子节点：存储编码到字符的映射
    if root.left is None and root.right is None:
        for char in root.chars:
            decodebook[path] = char
    else:
        # 递归处理左右子树
        build_decode(root.left, path + '0', decodebook)
        build_decode(root.right, path + '1', decodebook)
    
    return decodebook

def encode_string(s, codebook):
    """编码字符串：将每个字符替换为对应的哈夫曼编码"""
    return ''.join([codebook[char] for char in s])

def decode_string(s, decodebook):
    """解码字符串：
    1. 逐个读取bit构建当前编码
    2. 当编码匹配解码表时输出对应字符并重置"""
    ans = []
    cur_code = ''
    for bit in s:
        cur_code += bit
        if cur_code in decodebook:
            ans.append(decodebook[cur_code])
            cur_code = ''  # 重置当前编码
    return ''.join(ans)


n = int(input())  
char_weight = {}  
for _ in range(n):
    char, weight = input().split()
    char_weight[char] = int(weight)

# 构建哈夫曼树和编码解码表
root = build_huffman(char_weight)
codebook = build_code(root)
decodebook = build_decode(root)

# 处理查询直到EOF
while True:
    try:
        query = input()
        
        # 判断输入是编码串还是解码串
        if all(c in ('0', '1') for c in query):
            # 解码01串
            print(decode_string(query, decodebook))
        else:
            # 编码字符
            print(encode_string(query, codebook))
            
    except EOFError:  # 捕获文件结束符
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-16%20090902.png?raw=true)

## 2. 学习总结和收获

这次6题的作业好像做了8题？ 提早开始做作业，不料完成第三题后刷新发现作业题目有更多，算多做了一题，再加上也去做了sy那题，后半题目都是15日晚做的，当时要提交还发现机房维修，只能查看题解中的题目，早上再提交

这次作业的代码长度都很长，大部分不会难，只是太长不好debug，又或者会有所怯步，对于huffman树一开始不是很了解，上网查了才学会，后半题目有一些都是要思考好久才理解题意，比如二叉树层序遍历初看是完全不了解（对于我）测试数据，在纸上写完了对照数据才了解（希望测试数据给短点。。）

期中还有一门，希望51有时间好好复习数算至今的内容

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>