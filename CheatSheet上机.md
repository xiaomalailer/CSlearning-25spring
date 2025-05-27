## 一、有关语法

### 输入
```n = int(input())```:输入 n为数字

```list1 = list(map(int, input().split()))``` ：输入一行形成列表 其中```input().split()```将整行输入按空格分割为多个部分 

```M, N = map(int, input().split())```一次读取两个数（经典必会）

```arr.append(tuple(map(int, input().split())))``` 形成list中的tuple

### 输出
```print("{a:._f}"))```：输出a保留_位小数

```print(*arr)```：列表中的所有元素展开为独立的参数，方便输出

```print(,end='')```： 就是输出后不要换下一行(也可用于输出后添加东西在后边）

```print(f"{}")```：可用于输出一些符号代表数值同时又不会有空格

```print(, sep='\n')```：输出每个元素占一行


### 函数（简单的不收录）

import math -> ```math.ceil()``` :向上取整 ;``` math.log() ```:可取logn

```abs()```：绝对值

```ord()```： 是转变成可比较数（但不同于int)

```chr()```： 转变成字符

 ```a= float("inf")```：inf无穷，需用float(也可以加负号变成负无穷）```float('-inf')```

 ```.replace(old,new,count)```：将字符串中的某些字符替换（前为欲替换字符，后为替换后）
 
 ```count(substring,start,end)``` 返回字符串中指定子串出现的次数
 
 ```rfind(subsring,start,end) ```返回指定子串在字串中最后一次出现的位置，无则返回-1

 ```startswith(prefix,start,end) ; endswith(suffix,start,end)``` 检查字符串是否以指定前缀开始/以指定后缀结束

 ```
try:
    while True:
        
except EOFError:
    pass/break  # This will stop the loop when no more input is given
```

```''.join（）```：以''链接（）内的元素(使用字符串将序列中的元素连接起来生成新字符串）

```all()```：all() 返回 True，当且仅当可迭代对象中的所有元素都为 True。如果有一个元素为 False，则 all() 返回 False

```[::-1]```意思是翻转

```bin()``` 是转换为二进制，但是开头会带有多余的 '0b' 

```global ```：使全局变量

```from collections import deque -> deque()```：双端队列，可以在两端高效地进行插入和删除操作的数据结构<br> 
**左侧操作：**<br>
```appendleft(x)```: 在左侧插入元素 x。<br>
```popleft()```: 从左侧弹出并返回一个元素。<br>
**右侧操作：**<br>
```append(x)```: 在右侧插入元素 x（类似于列表的 append）。<br>
```pop()```: 从右侧弹出并返回一个元素。

```from itertools import permutations``` ``` permutations(n)``` :全排列数字

```enumerate(perm)```将数组转化为（索引，元素）

``` a==a[::-1] ```判断回文方法之一

### 数组list用
```.index()```：找出某个值（数组内）的索引值

```.append()```：后面添加元素

```.sort()```：排序数组内元素 ；```.sort(reverse=True)```：逆序排列 ；```sorted(arr_1, key=lambda x: )```，将arr_1按后边的指示排序（也可以是处理后排序）；```sorted(,key=lambda x:( , ))```用来面对当数值/字符无法按此处理时用‘，’后的方法排序

```.rstrip()```： 表示移除字符串末尾（右侧）的指定字符（默认为空格和换行符）

```arr=[(arr1[i], i) for i in range(j)] ```数组内创造tuple（带index）

```arr = [[0] * bc for _ in range(ar)]``` 是用来初始化一个二维列表（矩阵）的，它的作用是创建一个大小为 ar 行、bc 列的矩阵，并将所有元素初始化为 0(也可以没有元素）。

```arr[-1]```：arr中最后一个元素

```arr.find(' ' ,0) ```返回索引，找不到返回-1，后面数字可以指定从第几个索引以后开始找

```list_n.remove(i)```：去除i

```arr1=arr2.copy()```将arr2内容copy给arr1


### 字典
写法：```dict1={'a':0,'b':1...}```; 调用方式 ```dict1['a']=0```

```dict.setdefault(key, default=None)```# 如果 key 存在于字典中则不更改；如果不存在，则将 key 加入字典，并赋值为 default


## 二、算法或经典方法

### 使用埃氏筛法生成所有小于等于 10^6 的质数（见于Tprimes）
```
def sieve_of_eratosthenes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False  # 0 和 1 不是质数
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return is_prime
```

### 矩阵相乘
```
for i in range(n):
    for j in range(n):
        for k in range(n):
            result[i][j] += arr_1[i][k] * arr_2[k][j]
```

### 二分查找
#### 08210: 河中跳房子
```
def binary_stone(s):
    m = 0          # 移除的岩石数量
    s_now = 0      # 当前岩石的位置
    for i in range(1, N + 2):  # 遍历所有岩石（包括起点和终点）
        if stone[i] - s_now < s:  # 如果当前岩石与上一个岩石的距离小于 s
            m += 1               # 需要移除这个岩石
        else:
            s_now = stone[i]     # 否则更新当前位置
    if M < m:  # 如果移除的岩石数量超过了 M
        return True  # 不可行
    return False  # 可行

L, N, M = map(int, input().split())  # 输入 L, N, M
stone = [0]  # 起点位置
for _ in range(N):
    stone.append(int(input()))  # 输入岩石位置
stone.append(L)  # 终点位置

left = 0  # 二分查找的左边界
right = L   # 二分查找的右边界
ans = 0  # 最终结果

while left < right:  # 二分查找
    mid = (left + right) // 2  # 中间值
    if binary_stone(mid):  # 判断 mid 是否可行
        right = mid  # 如果不可行，说明 mid 太大，缩小范围
    else:
        left = mid + 1  # 如果可行，说明 mid 是一个潜在解，继续尝试更大的值
        ans = mid  # 更新答案
print(ans) 
```

### 合法出栈检查
```
def valid_stack(s):
    stack=[]
    cur=0
    for num in s:
        if num>cur:
            for j in range(cur+1,num+1):
                stack.append(j)
            cur=num
        if stack.pop()!=num:
            return False
    return True
```

### 回文检查
```
def check_is_palindrome(s):
    start = 0
    end = len(s) - 1
    while start < end:
        if s[start] != s[end]:
            return False
        else:
            start += 1
            end -= 1
    return True
```


### 表达式

#### 后序表达式 “24588:后序表达式求值”
```
def calculate(op):
    # 从栈中弹出两个操作数（注意顺序，后弹出的为左操作数a，先弹出的是右操作数b）
    b = stack.pop()
    a = stack.pop()

    # 根据操作符进行运算并将结果压回栈中
    if op == '+':
        stack.append(b + a)  # 注意是 b + a，因为加法交换律顺序无影响
    elif op == '-':
        stack.append(a - b)  # 减法顺序重要，a 是先弹出的，表示左操作数
    elif op == '*':
        stack.append(b * a)
    elif op == '/':
        stack.append(a / b)  # 除法顺序也重要，a / b 是正确顺序

n = int(input())
for _ in range(n):
    s = input().strip()      # 读取并去除首尾空格
    t = s.split()            # 按空格分割为一个个 token（操作数或运算符）
    priority = {'+': 0, '-': 0, '*': 1, '/': 1}  # 运算符优先级，这里其实没用到
    stack = []               # 初始化一个空栈，用于计算逆波兰式

    # 遍历 token 列表
    for token in t:
        # 判断是否为数字（包含小数）
        if token.replace('.', '', 1).isdigit():
            stack.append(float(token))  # 转换为浮点数后压入栈中
        elif token in '+-*/':
            calculate(token)  # 如果是运算符，调用 calculate 函数进行运算

    # 输出栈顶元素，保留两位小数
    print(f"{stack[0]:.2f}")
```

#### 24591:中序表达式转后序表达式
```
def infix_to_postfix(exp):
    # 操作符优先级字典，数字越大的优先级越高
    pre = { '+': 1, "-": 1, '*': 2, '/': 2 }

    stack = []      # 用栈存储操作符和括号
    postfix = []    # 存储最终的后缀表达式
    number = ''     # 临时存储数字

    # 遍历表达式中的每个字符
    for char in exp:
        # 如果字符是数字或小数点，继续构建数字
        if char.isnumeric() or char == '.':
            number += char  # 逐位构建数字

        else:
            # 如果有构建好的数字，将其添加到后缀表达式中
            if number:
                num = float(number)  # 将构建的数字转为浮动类型
                postfix.append(int(num) if num.is_integer() else num)
                number = ''  # 清空临时数字变量，准备处理下一个数字

            # 处理操作符（+、-、*、/）
            if char in '+-*/':
    # 判断栈顶的操作符优先级，若栈顶操作符优先级高于或等于当前操作符，则弹出栈顶操作符
                while stack and stack[-1] in '+-*/' and pre[char] <= pre[stack[-1]]:
                    postfix.append(stack.pop())  # 弹出栈顶操作符，并添加到后缀表达式中
                stack.append(char)  # 将当前操作符压入栈

            # 处理左括号 '('
            elif char == '(':
                stack.append(char)  # 将左括号压入栈中，表示子表达式的开始

            # 处理右括号 ')'
            elif char == ')':
                # 弹出栈顶的操作符，直到遇到左括号 '('
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())  # 将操作符弹出并添加到后缀表达式中
                stack.pop()  # 弹出左括号 '('，但不加入后缀表达式

    # 处理表达式结束后，若还有数字，添加到后缀表达式中
    if number:
        num = float(number)
        postfix.append(int(num) if num.is_integer() else num)

    # 弹出栈中的剩余操作符，并添加到后缀表达式中
    while stack:
        postfix.append(stack.pop())

    # 返回后缀表达式，元素用空格分隔
    return ' '.join(str(x) for x in postfix)

n = int(input())  
for _ in range(n):
    exp = input()  # 输入一个中缀表达式
    print(infix_to_postfix(exp))  # 输出转换后的后缀表达式
```

### 归并排序 : 计算一个数组中的 “逆序对个数”
```
def merge_sort(run):
    # 基本情况：当序列长度为 1 或更短时，无需排序，也没有逆序对
    if len(run) <= 1:
        return run, 0

    # 分成左右两部分递归处理
    mid = len(run) // 2
    left, left_cnt = merge_sort(run[:mid])
    right, right_cnt = merge_sort(run[mid:])

    # 合并两个已排序部分，并统计合并过程中产生的逆序对数量
    run, merge_cnt = merge(left, right)

    # 返回合并后的排序结果和总逆序对数
    return run, left_cnt + right_cnt + merge_cnt


def merge(left, right):
    merged = []                 # 合并后的结果
    left_idx, right_idx = 0, 0  # 指针
    cnt = 0                     # 当前合并过程中产生的逆序对数

    while left_idx < len(left) and right_idx < len(right):
        # 注意这里是“降序排序”（大到小）
        if left[left_idx] >= right[right_idx]:
            # 若左边值大于等于右边，直接添加，不构成逆序对
            merged.append(left[left_idx])
            left_idx += 1
        else:
            # 若左边值小于右边，说明左边剩下的所有值都 > 当前右边值
            merged.append(right[right_idx])
            right_idx += 1
            cnt += len(left) - left_idx  # 增加逆序对数量

    # 添加剩余未处理的部分
    merged += left[left_idx:] + right[right_idx:]

    return merged, cnt

n = int(input())  # 输入元素个数
run = [int(input()) for _ in range(n)]  # 输入数组
run, ans = merge_sort(run)  # 执行归并排序和逆序对统计
print(ans)  # 输出逆序对总数
```

### 全排列
```
def permute(self, nums: List[int]) -> List[List[int]]:
    def backtrack(nums, n, length, path, used, ans):
        # 终止条件：当 path 的长度等于 nums 的长度，说明得到一个完整排列
        if length == n:
            ans.append(path)  # 收集当前排列
            return

        for i in range(n):
            if not used[i]:
                # 选择当前数字 nums[i]
                used[i] = True
                # 递归进入下一层，路径加上 nums[i]
                backtrack(nums, n, length + 1, path + [nums[i]], used, ans)
                # 回溯，撤销选择
                used[i] = False

    n = len(nums)                   # 数组长度
    used = [False for _ in range(n)]  # 用于标记某个元素是否已被使用
    ans = []                        # 存储所有排列结果
    backtrack(nums, n, 0, [], used, ans)  # 启动回溯
    return ans
```

### 约瑟夫问题
```
while True:
    n, p, m = map(int, input().split())
    if n == 0 and p == 0 and m == 0:
        break
    kids = [i for i in range(1, n + 1)]

    # 将第 p 个人调到队首（模拟从 p 开始数数）
    for _ in range(p - 1):
        out = kids.pop(0)  # 将队首移除
        kids.append(out)   # 加到队尾，相当于队列轮换

    cnt = 0           # 报数计数器
    ans = []          # 存储出列顺序的结果

    # 模拟游戏过程，直到只剩最后一个人
    while len(kids) > 1:
        out = kids.pop(0)  # 队首出队，准备判断是否出列
        cnt += 1           # 报数 +1

        if cnt == m:
            ans.append(out)  # 若数到 m，则此人出列，加入结果列表
            cnt = 0          # 计数器重置
            continue         # 不将此人加入队尾，直接进入下一轮

        kids.append(out)     # 若未数到 m，则此人排到队尾继续游戏

    ans.append(kids.pop(0))
    print(','.join(map(str, ans)))

#简单版
 queue = []
   for name in name_list:
        queue.append(name)

    while len(queue) > 1:
        for i in range(num):
            queue.append(queue.pop(0))
        queue.pop(0)										
    return queue.pop(0)
```

### 队列
#### 小组队列
```
from collections import deque 

t = int(input())  # 读取小组数量 t
groups = {}  # 存储小组ID与小组成员队列的映射
member_to_group = {}  # 存储成员编号与其所属小组ID的映射

for _ in range(t):
    members = list(map(int, input().split()))  # 读取小组成员（用空格分开）
    group_id = members[0]  # 以第一个成员的编号作为小组ID
    groups[group_id] = deque()  # 初始化该小组的成员队列
    for member in members:
        member_to_group[member] = group_id  # 将每个成员与其小组ID关联起来

queue = deque()  # 主队列，存储当前排队的小组ID
queue_set = set()  # 队列集，确保每个小组只在队列中出现一次

while True:
    command = input().split()  # 读取操作命令，按空格分割
    if command[0] == 'STOP':  # 如果命令是"STOP"，结束循环
        break
    elif command[0] == 'ENQUEUE':  # 如果命令是"ENQUEUE"，执行入队操作
        x = int(command[1])  # 获取要入队的成员编号
        group = member_to_group.get(x, None)  # 查找该成员所属的小组ID，如果没有则为None
        if group is None:  # 如果该成员没有所属小组（即散客）
            group = x  # 散客的ID本身作为小组ID
            groups[group] = deque([x])  # 创建一个新的队列，只有该散客
            member_to_group[x] = group  # 将该散客与其小组ID关联
        else:
            groups[group].append(x)  # 将该成员加入到其所属小组的队列中
        if group not in queue_set:  # 如果该小组还没有排队
            queue.append(group)  # 将该小组ID加入主队列
            queue_set.add(group)  # 将该小组ID添加到队列集，避免重复加入
    elif command[0] == 'DEQUEUE':  # 如果命令是"DEQUEUE"，执行出队操作
        if queue:  # 如果主队列不为空
            group = queue[0]  # 获取主队列中排在最前面的小组ID
            x = groups[group].popleft()  # 从该小组队列中移除并输出第一个成员
            print(x)  # 输出出队的成员编号
            if not groups[group]:  # 如果该小组队列为空
                queue.popleft()  # 从主队列中移除该小组
                queue_set.remove(group)  # 从队列集移除该小组
```

### dfs
#### 马走日
```
move=[(2,-1),(2,1),(-1,2),(1,2),(-1,-2),(1,-2),(-2,-1),(-2,1)]
path_sum=0
def dfs(n,m,x,y,path):
    global path_sum

    path.append((x,y))
    if len(path)==n*m:
        path_sum+=1
        path.pop()
        return
    for dy,dx in move:
        nx=x+dx
        ny=y+dy
        if 0<=nx<n and 0<=ny<m and (nx,ny) not in path:
            dfs(n,m,nx,ny,path)
    path.pop()
T=int(input())
for _ in range(T):
    n,m,x,y=map(int,input().split())
    path_sum=0
    dfs(n,m,x,y,[])
    print(path_sum)
```

### bfs
#### 滑雪
```
rows, cols = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(rows)]

# 将所有点按高度从小到大排序
points = sorted([(matrix[i][j], i, j) for i in range(rows) for j in range(cols)])

# 每个点的L值初始化为1
dp = [[1] * cols for _ in range(rows)]

# 定义方向数组，用于遍历上下左右
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 记录最长递增路径长度
longest_path = 1

# 从低到高，前面的不会对后面造成影响！
for height, x, y in points:
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and matrix[nx][ny] < height:
            dp[x][y] = max(dp[x][y], dp[nx][ny] + 1)
    longest_path = max(longest_path, dp[x][y])

print(longest_path)
```

####
```
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

### 链表
#### 相交 + 反转链表
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        a,b=headA,headB
        while a!=b:
            a=a.next if a else headB
            b=b.next if b else headA
        return a

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        start,end=head,None
        while start:
            tmp=start.next
            start.next=end
            end=start
            start=tmp
        return end
```
#### 合并两个有序链表
```
def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        ans = ListNode(0)  
        prev = ans
        
        while list1 and list2:
            if list1.val < list2.val:
                prev.next = list1  
                list1 = list1.next
            else:
                prev.next = list2  
                list2 = list2.next
            prev = prev.next  
        

        if list1:
            prev.next = list1
        elif list2:
            prev.next = list2

        return ans.next
```
#### 回文链表
```
def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or head.next is None:
            return True
        fast,slow=head,head
        
        while fast.next and fast.next.next:
            fast=fast.next.next
            slow=slow.next
        prev=None
        cur=slow.next
        while cur:
            tmp=cur.next
            cur.next=prev
            prev=cur
            cur=tmp
        flag=True
        l1=head
        l2=prev
        while l2:  
            if l1.val != l2.val:
                return False
            l1 = l1.next
            l2 = l2.next
        
        return True
```

### 树
#### 二叉树
```
#中序遍历
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []  # 用于存储遍历结果（中序）

        def dfs(root):
            if not root:
                return  # 空节点，返回

            dfs(root.left)        # 1. 递归访问左子树
            ans.append(root.val)  # 2. 访问当前节点
            dfs(root.right)       # 3. 递归访问右子树

        dfs(root)  # 启动 DFS 递归
        return ans
#层序遍历
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

#将有序数组转换为二叉树
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

# 求根节点到叶节点数字之和
def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(root, num):
            if not root:
                return 0
            total = num * 10 + root.val
            if not root.left and not root.right:
                return total
            return dfs(root.left, total) + dfs(root.right, total)

        return dfs(root, 0)
·   #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    print("前序遍历:", preorder_recursive(root))  # [1, 2, 4, 5, 3]
    print("中序遍历:", inorder_recursive(root))   # [4, 2, 5, 1, 3]
    print("后序遍历:", postorder_recursive(root))  # [4, 5, 2, 3, 1]

def preorder_recursive(root):
    """前序遍历（根-左-右）"""
    if not root:
        return []
    return [root.val] + preorder_recursive(root.left) + preorder_recursive(root.right)

def inorder_recursive(root):
    """中序遍历（左-根-右）"""
    if not root:
        return []
    return inorder_recursive(root.left) + [root.val] + inorder_recursive(root.right)

def postorder_recursive(root):
    """后序遍历（左-右-根）"""
    if not root:
        return []
    return postorder_recursive(root.left) + postorder_recursive(root.right) + [root.val]

# 完全二叉树的节点个数
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

#### 遍历树
```
n = int(input())

# 初始化树结构，使用字典存储，键为节点，值为子节点列表
tree = {}
# 使用集合存储所有子节点，便于后续查找根节点
children_set = set()
# 存储所有出现过的节点（包括父节点和子节点）
parent_list = []

# 处理每个节点的输入
for _ in range(n):
    # 读取每行输入并转换为整数列表
    parts = list(map(int, input().split()))
    node = parts[0]  # 当前节点
    parent_list.append(node)  # 将当前节点加入父节点列表
    
    # 如果当前节点有子节点（即输入长度大于1）
    if len(parts) > 1:
        children = parts[1:]  # 获取子节点列表
        tree[node] = children  # 将子节点列表存入树结构
        children_set.update(children)  # 将子节点加入子节点集合
    else:
        tree[node] = []  # 如果没有子节点，存入空列表

# 找出根节点（在父节点列表中但不在子节点集合中的节点）
root = (set(parent_list) - children_set).pop()

def traverse(node):
    # 将当前节点的子节点和当前节点本身排序（为了特定顺序遍历）
    parent_children = sorted(tree[node] + [node])
    
    # 遍历排序后的节点列表
    for x in parent_children:
        if x == node:
            # 如果是当前节点本身，直接输出
            print(node)
        else:
            # 否则递归遍历子节点
            traverse(x)
traverse(root)# 从根节点开始遍历
```

#### 22158:根据二叉树前中序序列建树
```
# 定义二叉树节点类
class Node:
    def __init__(self, val):
        self.val = val      # 节点值
        self.left = None    # 左子节点
        self.right = None   # 右子节点

# 根据前序遍历和中序遍历序列重建二叉树
def rebuild(preorder, inorder):
    # 如果序列为空，返回None
    if not preorder or not inorder:
        return None
    
    # 前序遍历的第一个元素是根节点
    root = Node(preorder[0])
    
    # 在中序遍历中找到根节点的位置
    root_idx_inorder = inorder.index(preorder[0])
    
    # 递归构建左子树：
    # 左子树的前序遍历 = 原始前序遍历[1:1+左子树长度]
    # 左子树的中序遍历 = 原始中序遍历[:根节点位置]
    root.left = rebuild(preorder[1:1+root_idx_inorder], inorder[:root_idx_inorder])
    
    # 递归构建右子树：
    # 右子树的前序遍历 = 原始前序遍历[1+左子树长度:]
    # 右子树的中序遍历 = 原始中序遍历[根节点位置+1:]
    root.right = rebuild(preorder[1+root_idx_inorder:], inorder[root_idx_inorder+1:])
    
    return root

# 后序遍历二叉树
def postorder(root):
    # 如果节点为空，返回空字符串
    if root is None:
        return ''
    
    # 后序遍历顺序：左子树 -> 右子树 -> 根节点
    return postorder(root.left) + postorder(root.right) + root.val

while True:
    try:
        # 读取前序和中序遍历序列
        preorder = input()
        inorder = input()
        
        root = rebuild(preorder, inorder)
        print(postorder(root))
    except EOFError:
        break
```
#### 霍夫曼编码树
```
import heapq  # 导入堆队列算法模块

def min_external_path_length(n, weights):
    """计算最小外部路径长度
    
    参数:
        n: 叶子节点数量
        weights: 各叶子节点的权重列表
        
    返回:
        最小外部路径长度值
    """
    # 特殊情况处理：只有一个叶子节点时，路径长度就是其自身权重
    if n == 1:
        return weights[0]

    # 创建最小堆（优先队列）
    heap = weights.copy()
    heapq.heapify(heap)  # 将列表转换为堆结构
    total = 0  # 初始化总路径长度

    # 核心算法：贪心策略合并最小权重的两个节点
    while len(heap) > 1:
        # 取出当前最小的两个权重
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        
        # 合并这两个节点
        combined = first + second
        # 累加到总路径长度（相当于内部节点被计算的次数）
        total += combined
        # 将合并后的新节点放回堆中
        heapq.heappush(heap, combined)
    return total

n = int(input())  
weights = list(map(int, input().split()))
print(min_external_path_length(n, weights))
```

#### M07161:森林的带度数层次序列存储
```
from collections import deque

def rebuild(tree_sequence):
    """
    根据给定的树序列重建树结构（使用字典表示）
    
    参数:
        tree_sequence: 树的序列化表示，格式为[(节点值, 子节点数), ...]
        
    返回:
        以字典形式表示的树结构，键为节点值，值为子节点列表
    """
    if not tree_sequence:
        return {}
    
    # 使用队列存储输入的树序列
    queue = deque(tree_sequence)
    # 取出根节点和它的子节点数
    root_val, root_degree = queue.popleft()
    # 初始化树结构
    tree = {}
    tree[root_val] = []
    # 使用队列存储待处理的节点及其子节点数
    node_queue = deque([(root_val, root_degree)])

    while queue and node_queue:
        # 取出当前节点及其子节点数
        cur_node, degree = node_queue.popleft()
        roots = []
        
        # 根据子节点数添加子节点
        for _ in range(degree):
            if not queue:
                break
            # 取出子节点及其子节点数
            cur_root, cur_root_degree = queue.popleft()
            roots.append(cur_root)
            # 将子节点加入待处理队列
            node_queue.append((cur_root, cur_root_degree))
        
        # 将子节点列表加入树结构
        tree[cur_node] = roots
    
    return tree

def reorder_tree(tree, node, ans):
    """
    对树进行后序遍历
    
    参数:
        tree: 字典表示的树结构
        node: 当前遍历的节点
        ans: 存储遍历结果的列表
    """
    # 递归遍历所有子节点
    for root in tree.get(node, []):
        reorder_tree(tree, root, ans)
    # 后序位置添加当前节点
    ans.append(node)

n = int(input())  # 输入树的数量
forest = []       # 存储所有树的列表
top_roots = []    # 存储所有树的根节点

for _ in range(n):
    # 读取每棵树的序列化表示
    tree_i = input().split()
    # 将输入转换为(节点值, 子节点数)的元组列表
    tree = [(tree_i[i], int(tree_i[i+1])) for i in range(0, len(tree_i), 2)]
    # 重建当前树
    new_tree = rebuild(tree)
    forest.append(new_tree)
    # 保存当前树的根节点
    top_roots.append(tree[0][0])

print(forest)

ans = []
# 对每棵树进行后序遍历
for i in range(n):
    reorder_tree(forest[i], top_roots[i], ans)

# 打印后序遍历结果
print(' '.join(ans))
```
#### T24637:宝藏二叉树
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def find(root):
    def dfs(node):
        if not node:
            return (0, 0)  # 返回元组：(选这个节点的最大和, 不选这个节点的最大和)

        # 递归处理左子树和右子树
        left = dfs(node.left)
        right = dfs(node.right)

        # 选当前节点：则不能选子节点
        take_node_val = node.val + left[1] + right[1]

        # 不选当前节点：可选可不选子节点（取子节点两种情况的最大值）
        not_take_node_val = max(left[0], left[1]) + max(right[0], right[1])

        return (take_node_val, not_take_node_val)

    res = dfs(root)
    return max(res)  # 返回最终最大值，选或不选根节点中较大的

n = int(input())  # 节点数
```

### 图
```
# 无向度
degrees=[0]*n
for _ in range(m):
    u,v=map(int,input().split())
    degrees[u]+=1
    degrees[v]+=1

# 有向度
degrees=[[0]*2 for i in range(n)]
for _ in range(m):
    u,v=map(int,input().split())
    degrees[u][1]+=1
    degrees[v][0]+=1

# 无向图矩阵
degrees=[[0]*n for i in range(n)]
for _ in range(m):
    u,v=map(int,input().split())
    degrees[u][v]+=1
    degrees[v][u]+=1

# 有向图矩阵
degrees=[[0]*n for i in range(n)]
for _ in range(m):
    u,v=map(int,input().split())
    degrees[u][v]=1

# 无向邻接
degrees=[[] for _ in range(n)]
for _ in range(m):
    u,v=map(int,input().split())
    degrees[u].append(v)
    degrees[v].append(u)

# 有向邻接
degrees=[[] for _ in range(n)]
for _ in range(m):
    u,v=map(int,input().split())
    degrees[u].append(v)

# 有向图判环
def dfs(u, degrees, visited):
    visited[u] = 0  # 标记当前节点为"正在访问"
   
    # 遍历所有邻接节点
    for v in degrees[u]:
        if visited[v] == -1:  # 如果邻接节点未访问
            if dfs(v, degrees, visited):  # 递归访问
                return True
        elif visited[v] == 0:  # 如果邻接节点正在被访问（在递归栈中）
            return True  # 发现环
    
    visited[u] = 1  # 标记当前节点为"已访问完成"
    return False

n, m = map(int, input().split())  # n=节点数，m=边数
degrees = [[] for _ in range(n)]  # 初始化邻接表
    
for _ in range(m):
  u, v = map(int, input().split())  # 读取有向边u→v
  degrees[u].append(v)

  visited = [-1] * n  # -1表示未访问
  flag = True  # 标记是否发现环
    
# 对每个未访问的节点进行DFS
  for i in range(n):
    if visited[i] == -1:
       if dfs(i, degrees, visited):  # 如果发现环
            flag = False
            print('Yes')  # 输出存在环
            exit()  # 立即退出
if flag == True:
   print('No')  # 输出无环
```


### 拓扑排序
```
import heapq
v, a = map(int, input().split())  # v=顶点数，a=边数
degrees = [[] for _ in range(v + 1)]  # 邻接表
in_degree = [0] * (v + 1)  # 入度统计

for _ in range(a):
    U, V = map(int, input().split())  # 边 U→V
    degrees[U].append(V)
    in_degree[V] += 1  # 更新入度
stack = []  # 优先队列（最小堆）
for i in range(1, v + 1):
    if in_degree[i] == 0:  # 入度为0的顶点入队
        heapq.heappush(stack, i)
result = []
while stack:
    u = heapq.heappop(stack)  # 取出当前最小编号的顶点
    result.append(str(u))  # 加入结果
    for v in degrees[u]:  # 遍历 u 的所有邻接点
        in_degree[v] -= 1  # 减少入度
        if in_degree[v] == 0:  # 如果入度降为0，加入队列
            heapq.heappush(stack, v)
ans = []
for i in result:
    ans.append('v' + str(i))  # 添加 'v' 前缀
print(' '.join(ans))  # 输出结果
```

### 堆
#### 手搓
```
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

#### M05443:兔子与樱花——dijikstra
```
from collections import defaultdict
import heapq

def dijkstra(degrees, start, end):
    distances = {node: float('inf') for node in degrees}  # 初始化距离
    prev = {node: None for node in degrees}  # 前驱节点
    distances[start] = 0
    heap = [(0, start)]  # 最小堆：(距离, 节点)

    while heap:
        cur_distance, cur_node = heapq.heappop(heap)
        if cur_distance > distances[cur_node]:  # 已找到更短路径，跳过
            continue
        if cur_node == end:  # 找到终点，提前退出
            break
        for neighbor, weight in degrees[cur_node].items():
            distance = cur_distance + weight
            if distance < distances[neighbor]:  # 松弛操作
                distances[neighbor] = distance
                prev[neighbor] = cur_node
                heapq.heappush(heap, (distance, neighbor))

    # 回溯路径
    path = []
    current = end
    while prev[current] is not None:
        path.insert(0, current)
        current = prev[current]
    path.insert(0, start)
    return path, distances[end]

P = int(input())  # 地点数量
places = [input().strip() for _ in range(P)]  # 地点列表

Q = int(input())  # 路径数量
degrees = defaultdict(dict)  # 邻接表（无向图）
for _ in range(Q):
    a, b, degree = input().split()
    degree = int(degree)
    degrees[a][b] = degree  # 存储权重
    degrees[b][a] = degree  # 无向图需双向存储

R = int(input())  # 查询数量
targets = [input().split() for _ in range(R)]  # 查询列表

for start, end in targets:
    if start == end:  # 起点=终点，直接输出
        print(start)
        continue
    path, total_distance = dijkstra(degrees, start, end)  # 计算最短路径
    # 格式化输出
    output = path[0]
    for i in range(1, len(path)):
        output += f"->({degrees[path[i-1]][path[i]]})->{path[i]}"
    print(output)
```

#### 走山路
```
import heapq

def find_min_cost_path(m, n, mat, queries):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向
    results = []

    for x1, y1, x2, y2 in queries:
        if mat[x1][y1] == '#' or mat[x2][y2] == '#':
            results.append("NO")
            continue
        heap = [(0, x1, y1)]  # (体力消耗, 行, 列)
        dist = {(x1, y1): 0}  # 记录从起点到每个坐标的最小体力消耗
        found = False
        while heap:
            cost, x, y = heapq.heappop(heap)
            # 如果到达终点，记录结果并退出循环
            if (x, y) == (x2, y2):
                results.append(cost)
                found = True
                break
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and mat[nx][ny] != '#':
                    new_cost = cost + abs(int(mat[nx][ny]) - int(mat[x][y]))
                    # 如果新路径的体力消耗小于已知的最小体力消耗，则更新
                    if (nx, ny) not in dist or new_cost < dist[(nx, ny)]:
                        dist[(nx, ny)] = new_cost
                        heapq.heappush(heap, (new_cost, nx, ny))

        # 如果未找到终点，记录 "NO"
        if not found:
            results.append("NO")

    return results

m, n, p = map(int, input().split())
mat = [input().split() for _ in range(m)]
queries = [tuple(map(int, input().split())) for _ in range(p)]

answers = find_min_cost_path(m, n, mat, queries)
print("\n".join(map(str, answers)))
```

### 双指针
#### M18156:寻找离目标数最近的两数之和
```
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

### 其他
#### 倒排索引
```
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
#### 04093:倒排索引查询
```
N = int(input())  # 输入关键词种类数量
find = []

# 读取每个关键词组的文档编号集合
for i in range(N):
    s = input().split()
    # s[0] 是该关键词的文档数量，s[1:] 是对应的文档编号
    find.append(set(map(int, s[1:])))

M = int(input())  # 输入查询次数

# 处理每次查询
for _ in range(M):
    a = list(map(int, input().split()))  # 用户输入的查询条件，长度为 N
    must_have = set()        # 需要出现的文档（初始为空集合）
    must_not_have = set()    # 不能出现的文档
    first = True             # 用于初始化交集的起点

    for i in range(N):  # 遍历所有关键词的查询要求
        if a[i] == 1:
            if first:
                must_have = find[i].copy()  # 第一个必须包含的集合直接赋值
                first = False
            else:
                must_have &= find[i]        # 后续必须包含的集合取交集
        elif a[i] == -1:
            must_not_have |= find[i]        # 不允许包含的集合取并集

    # 从必须包含的集合中去除不允许包含的集合
    valid_docs = sorted(must_have - must_not_have)

    # 输出结果
    if valid_docs:
        print(' '.join(map(str, valid_docs)))
    else:
        print("NOT FOUND")
```

#### 27256:当前队列中位数
```
from bisect import bisect_left

a, b, cnt, now = [], [], 0, 0
# a：用于存储排序后的值，元素为 [值, 插入编号]，方便处理重复元素
# b：记录插入的原始顺序，用于删除时按顺序找到元素
# cnt：记录当前插入元素的编号
# now：表示删除到第几个元素（模拟队列头）

for _ in range(int(input())):
    opt = input().split()
    
    if opt[0] == 'query':
        l = len(a)  # 当前元素个数
        if l & 1:
            # 若个数为奇数，取中间那个数
            print(a[l >> 1][0])
        else:
            # 个数为偶数，取中间两个数平均
            ans = (a[l >> 1][0] + a[(l - 1) >> 1][0]) / 2
            # 如果结果是整数就输出整数，否则输出浮点数
            print(ans if int(ans) != ans else int(ans))
    
    if opt[0] == 'add':
        v = int(opt[1])
        # 插入 [值, 插入编号]，用 bisect 保证 a 始终是升序排列
        a.insert(bisect_left(a, [v, 0]), [v, cnt])
        b.append(v)  # 记录原始顺序
        cnt += 1
    
    if opt[0] == 'del':
        v = b[now]   # 找到当前最早插入的值
        now += 1     # 队列头右移
        # 从 a 中删除对应的 [值, 插入编号]，因为可能有重复值，用编号区分
        a.pop(bisect_left(a, [v, 0]))
```

#### 合并区间
```
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    # 先将所有区间按起始位置升序排序
    intervals.sort(key=lambda x: x[0])
    
    # 初始化第一个区间的起始和结束位置
    start = intervals[0][0]
    end = intervals[0][1]
    
    ans = []  # 存储合并后的区间结果
    
    # 遍历后续所有区间
    for i in range(1, len(intervals)):
        # 如果当前区间的起点在前一个区间的终点之内，说明有重叠
        if intervals[i][0] <= end:
            # 更新当前合并区间的终点为两个终点的较大值
            end = max(end, intervals[i][1])
        else:
            # 如果没有重叠，把之前的合并结果加入答案
            ans.append([start, end])
            # 重新开始新的合并区间
            start = intervals[i][0]
            end = intervals[i][1]
    
    # 最后一个区间也要加入结果
    ans.append([start, end])
    return ans
```

#### 北大夺冠——字典
```
M = int(input())

# 初始化团队字典，用于存储每个团队的信息
teams = {}

# 处理每个团队的提交信息
for _ in range(M):
    # 读取每行输入，格式为：团队名称,题目编号,提交结果
    name, question, result = input().split(',')
    
    # 如果团队不存在，初始化其信息
    if name not in teams:
        teams[name] = {"ac": set(), 'submissions': 0}
   # ac存储通过的题目集合，submissions存储总提交次数
    
    teams[name]['submissions'] += 1
    
    # 如果提交结果为'yes'且题目未被记录过，添加到ac集合中
    if result == 'yes' and question not in teams[name]['ac']:
        teams[name]['ac'].add(question)

# 将团队信息转换为列表，用于排序
team_list = []
for name in teams:
    ac_cnt = len(teams[name]['ac'])  # 计算通过的题目数量
    submissions = teams[name]['submissions']  # 获取总提交次数
    # 将团队信息以元组形式存入列表，其中ac_cnt取负值以便排序时降序排列
    team_list.append((-1 * ac_cnt, submissions, name))

# 对团队列表进行排序（首先按通过的题目数量降序，其次按提交次数升序）
team_list.sort()

# 输出前12名的团队信息
for rank, (ac_cnt, submissions, name) in enumerate(team_list[:12]):
    ac = -ac_cnt  # 将ac_cnt恢复为正值
    # 格式化输出：排名、团队名称、通过题目数量、总提交次数
    print(f"{rank + 1} {name} {ac} {submissions}")
```

#### 牛选举 list、tuple
```
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

#### poker
```
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

flower_order={'A':0,'B':1,'C':2,'D':3}
queue_flower=[deque() for _ in range(4)]
for nums in first_order:
    flower=nums[0]
    flower_idx=flower_order[flower]
    queue_flower[flower_idx].append(nums)

final_result=[]
for i in range(4):
    flower=['A','B','C','D'][i]
    queue=queue_flower[i]
    print(f"Queue{flower}:{' '.join(queue)}")
    final_result.extend(queue)

print(' '.join(final_result))
```
