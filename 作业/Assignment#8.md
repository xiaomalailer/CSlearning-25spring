# Assignment #8: ��Ϊ��

Updated 1434 GMT+8 Apr 8, 2025

2025 spring, Complied by <mark>��PȨ Ԫ��</mark>





## 1. ��Ŀ

### LC108.����������ת��Ϊ������

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

˼·��

�Ƕ��ַ�������ʵһ�룬����dfs��������˼·��һ�����������м�����(�м���ڵ�����ΪҶ�ӽڵ㣩���м��ǵ�һ���ڵ㣬���������ߵĸ�����Զ����ʼ�����ұ�ͬ����˿�����dfs��ʽ������������һ��֣���һ����ʽ��ֻ�Ǳ�����м䣨�������Ϊ��Ҷ�ӽڵ㣩��Զ������

���룺

```python
#��һ�ִ𰸡�������Զ���������꡿
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

##�ڶ��֡������м����������꡿
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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20160918.png?raw=true)



### M27928:������

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

˼·��

�ο���������ѧ���Ĵ𰸣���������dict{}��û����defaultdict��ԭ����ͬ���ְѸ����ӽڵ��ϵ����ã�Ȼ���ص���Ҫ�����������ڵ��ҳ��������������ü��ϼ����ҳ�Ψһ����Ϊ�ӽڵ�ĸ��ڵ㣬�������뵽������������������ѧ���ķ����ܺã����ǽ����ڵ�����ӽڵ����򣬽�С��������ӽڵ㣬���������Ϊ0������������������Ǹ��ڵ��ֱ�������

�Եڶ����������ݣ�

{10: [3, 1], 7: [], 9: [2], 2: [10], 3: [7], 1: []} 

9 #���ڵ�

2 #sorted=>[2,9] , sorted[2,10] print"2"

1 #sorted[1,3,10] print"1" 

3 #sorted[3,7] print"3"

7 #print"7"

10 #print"10"

9 #print"9"

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20174009.png?raw=true)



### LC129.����ڵ㵽Ҷ�ڵ�����֮��

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

˼·��

��Ȼ��dfs��ʽ���½��б��������һ��num��¼��ֵ����Ϊ�Ƕ������������ҽ������㼴�ɣ�Ҷ�ڵ��־��if not root.left and not root.right:

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20180323.png?raw=true)



### 22158:���ݶ�����ǰ�������н���

tree, http://cs101.openjudge.cn/practice/24729/

˼·��

����ǰ�����������ؽ�������ת��Ϊ�������ؽ���������£�����ȷ��ǰ���һ����ĸΪ���������׸����ڵ㣬����ô������Ϊ�����ؽ���ߺ��ұߵ���������ߵ���������������ʽ���׸��ڵ����λ��ȷ�����ֿɷ�Ϊǰ����������������������������׸��ڵ㣨n����ĸ��Ϊ���������ǰ�����׽ڵ��n����Ϊ����ǰ���������ͬ����ˣ����յݹ�/dfs�Ϳ��ؽ����ٻ�ȥ������Ｔ��


���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20215652.png?raw=true)



### T24729:����Ƕ����

dfs, stack, http://cs101.openjudge.cn/practice/24729/

˼·��

������Ҫ��������һ���ؽ����������stack�ķ���������ÿ���ڵ㣬ͬʱ��������ƥ�䣬���������ţ���������һ���ӽڵ㴦������������ĸ�����ӽڵ㣬���������Ŵ�������Ҷ�ڵ㣬��Ҫ����Ŀǰ���ڴ���Ľڵ㣻ǰ��������ж��Եݹ���ʽ��ǰ����һֱ����ӣ��Ƚ�������������ҵ�Ҷ�ڵ�������ؼӣ������¸��ڵ����뵽Ҷ�ڵ������ؼ�

����(A->B->E->NONE) E �� B ��  F �� G �� C  �� I �� H �� D  �� A

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20212237.png?raw=true)




### LC3510.�Ƴ���С����ʹ��������II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

˼·��

����pairwise����������ԣ�enumerate����index����Ժý��мӺͺ�����heapq������С�ѣ���ôpop�������ǵ���ѡ�е�һ����С�ͣ�

���ڴ˺ͣ��ֱ��жϼӺͺ���ڵݼ�������Ӱ�죬Ȼ��ֱ��ж���֮ǰ��һ������֮���һ���������¸������ĵݼ�Ӱ�죬����ÿ�������ǵ�֮ǰ����left�ͺ����right������left=[-1,0,1,...],right=[1,2,3...]��¼���������жϺ󣬳��ǼӺ͵������ڱ߽磨��/�ң���������Ժ���/����һ�μӺ��ٷ�����С���ж�

������Ϊֻ�Ǹ���������ĳ��λ�õ�������Ҫɾ����nums[i+1]�����������ں��������в��ٱ����ʵ���������������¸���������ұߵģ����ұߵ���߱������ߣ����������ֿڣ�����ʵ���������һ��ʵ��


���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20234900.png?raw=true)



## 2. ѧϰ�ܽ���ջ�

�����Ŀ�����ѵ�������or�ο��𰸺���û��ã����������д��̫���ˣ������ڸ���root��node��tree֮����ʧ������һЩ��ʽ��д����ȷ��Ҫ��ϰ�����Կ��������˵㣬������Ŀ��ʱҲ�е㡰���ӡ���������˵�ø��ӣ����󲿷���Ŀ�����߼������ǣ���Ҫ�õ��ݹ�/dfs˼·���б�������һ��Ƚ���Ҫ��ֽ�Ͻ���ģ��

������ԣ�һ����һ������ϰ���������𽥶������֣�������ǰ�漸�⻨��ʱ�䷴�����࣬����ͻ���

�����������ܣ����Ǻ�æ��æ�Ÿ�ϰ����������պ��ڽ��Ҹ�ϰ���͹���ʵ���һ����һ�����������ҵ���ž�������˺ܶ�����ص�


<mark>���������ҵ��Ŀ��Լ򵥣��з�Ѱ�Ҷ������ϰ��Ŀ���硰����2025springÿ��ѡ������LeetCode��Codeforces����ȵ���վ�ϵ���Ŀ��</mark>