# Assignment #: Huffman & Fenwick

Updated 1034 GMT+8 Apr 15, 2025

2025 spring, Complied by <mark>��PȨ Ԫ��</mark>





## 1. ��Ŀ

### LC222.��ȫ�������Ľڵ����

dfs, https://leetcode.cn/problems/count-complete-tree-nodes/

˼·��

dfs˼·����������������

���˽����һ�����ӶȽϵ͵ķ����������ģ����Ƿֳ�������������������or�����������������������������������������������Ӧ���������߶� == �������߶ȣ�����������������������Ӧ���������߶� > �������߶�

���룺

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

�������ο����
```
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        # ���������������� 2^h - 1
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

        if left_height == right_height: # ������������
            # �������� 2^h - 1 Ȼ���root
            return 2 ** left_height + self.countNodes(root.right)
        else:  # ������������
            return 2 ** right_height + self.countNodes(root.left)

```


�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-15%20163254.png?raw=true)



### LC103.�������ľ���β������

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

˼·��

bfs�������������ż�������ڵ��0��ʼ���㣬��ò����Խڵ�����append������������������������Ҫ���Ƚ�

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-15%20170613.png?raw=true)




### M03720:�ı���������ԭ�����⣩

tree, http://cs101.openjudge.cn/practice/03720/

˼·��

���ȵ��ؽ������ؼ���������Ϊ��while stack and stack[-1][1] != cur_level - 1:stack.pop() ������ȷ����ǰ�ڵ�Ĳ���-1�Ǹ��ڵ㣬stack���𴢴游�ڵ㣬�������㷽���ǵ�ǰ�г�-1����i��-��1����ĸ���ɣ�������*Ҳ����ڵ㣬�ڽ���ǰ���������ʱ������ *����Ҫ������ǰ�к������д����׶�������׸��

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-15%20181018.png?raw=true)



### M04080:Huffman������

greedy, http://cs101.openjudge.cn/practice/04080/

˼·��

�����Ͼ���������С����Ե��ܺͣ����� 1 1 3 5�� ���� 1+1 =2 ��2�ٺ� 3 �� 5��5�ٺ�5 =10 ����С�ⲿ·�������ܺ;���2+5+10

���ӻ�Щ���� [1,1,3,5] �� [2,3,5] �� [5,5] �� [10]

��������Ҳ���� 

![huff](https://images2015.cnblogs.com/blog/610439/201612/610439-20161214230137323-1092491743.png)

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-16%20083053.png?raw=true)

### M05455: �����������Ĳ�α���

http://cs101.openjudge.cn/practice/05455/

˼·��

����Ҫ���״������Ĺ��򣬶���ĳ���ڵ�

1���������е����нڵ��ֵ��С�ڸýڵ��ֵ

2���������е����нڵ��ֵ�����ڸýڵ��ֵ

�������������Ƕ���һ���ڵ�ֵ��ͨ����������ıȶԽ��в���

���� 51 45 59 86

���� 45 ��51С ���ŵ�51 ������

���� 59 ��51�󣺷ŵ�51 ������

���� 86 ��51�󣺷ŵ�51 ����������51�������У���59�󣬷ŵ�59������

���в�ݵ����������deque���bfs�Ĳ�α���

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-16%20083546.png?raw=true)

### M04078: ʵ�ֶѽṹ

�ִ�ʵ�֣�http://cs101.openjudge.cn/practice/04078/

���Ƶ���Ŀ�� ����9.7: ���µ��������󶥶ѣ�https://sunnywhy.com/sfbj/9/7

˼·��

�ѱ���������ȫ��������ʵ����Ҫ��heappush��heappop������heappush�����Ǵ���СҶ�ڵ����ϱȶԸ��ڵ㣬ֻҪ�ܽ�����һֱ���Ͻ���������heappop��������С�Ѵ������ڵ����С��Ȼ������һ�����Ƶ����ڵ㣬�����½��бȶԱ�֤��С�ѣ���������ӽڵ�����ұȵ�ǰ�ڵ�С�����ӽڵ���ڱȵ�ǰ�ڵ�С����ô��Ҫ����


���룺

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

ֱ��ʵ��
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

sy���
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


�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q51](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-16%20083723.png?raw=true)

![Q52](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-15%20234549.png?raw=true)


### T22161: ������������

greedy, http://cs101.openjudge.cn/practice/22161/

˼·��

�����˽��������������

�ڵ�Ƚϣ��ȱ�Ȩֵ��Ȩֵ��ͬ���ַ�����С�ַ�

�ϲ�ʱС�ڵ���Ϊ���ӽڵ�

���Ϊ0���ұ�Ϊ1

����

      {c,g,d}:22
      /      \
    {c}:10    {g,d}:12
              /    \
          {g}:4    {d}:8

����������У�c=0��g=10��d=11

���� 100 =gc ��110=dc ��֮��ͬ

�������������Ԥ�����ɽ���ͱ��������Ҫʱ���Կ��ٱ�����룬���������˼·������ai��æע�����



���룺

```python
import heapq  
class HuffmanNode:
    def __init__(self, chars, weight, left=None, right=None):
        self.chars = chars    
        self.weight = weight  
        self.left = left     
        self.right = right    

    def __lt__(self, other):
        """����ڵ�ȽϹ������ڶ�����
        1. ���ȱȽ�Ȩ�أ�Ȩ��С�Ľڵ��С
        2. Ȩ����ͬʱ���Ƚ��ַ�������С���ַ�"""
        if self.weight != other.weight:
            return self.weight < other.weight
        return min(self.chars) < min(other.chars)

def build_huffman(char_weight):
    """��������������
    1. ��ʼ����С��
    2. ���Ϻϲ�Ȩ����С�������ڵ�
    3. �������յĸ��ڵ�"""
    heap = []
    # ��ʼ���ѣ�Ϊÿ���ַ�����Ҷ�ӽڵ�
    for char, weight in char_weight.items():
        heapq.heappush(heap, HuffmanNode({char}, weight))  # ע�⣺ʹ�ü��ϴ洢�����ַ�
    
    # �ϲ��ڵ�ֱ��ֻʣһ�����ڵ�
    while len(heap) > 1:
        left = heapq.heappop(heap)   # ȡ��Ȩ����С�Ľڵ�
        right = heapq.heappop(heap)  # ȡ��Ȩ�صڶ�С�Ľڵ�
        
        # �ϲ������ڵ㣺�ַ����ϲ���Ȩ�����
        merged_c = left.chars.union(right.chars)
        merged_w = left.weight + right.weight
        # �����½ڵ㣬��С�Ľڵ���Ϊ���ӽڵ�
        merged_node = HuffmanNode(merged_c, merged_w, left, right)
        heapq.heappush(heap, merged_node)  # ���½ڵ�Żض���
    
    return heapq.heappop(heap)  # �������յĸ��ڵ�

def build_code(root, path='', codebook=None):
    """�ݹ鹹�������ֵ䣨�ַ� -> ���룩��
    - ��·�����'0'����·�����'1'
    - ����Ҷ�ӽڵ�ʱ��¼�ַ��ı���"""
    if codebook is None:
        codebook = {}
    
    # Ҷ�ӽڵ㣺�洢�ַ��������ӳ��
    if root.left is None and root.right is None:
        for char in root.chars:
            codebook[char] = path
    else:
        # �ݹ鴦����������
        build_code(root.left, path + '0', codebook)
        build_code(root.right, path + '1', codebook)
    
    return codebook

def build_decode(root, path='', decodebook=None):
    """�ݹ鹹�������ֵ䣨���� -> �ַ�����
    - �ṹ��build_code���ƣ�����¼���Ǳ��뵽�ַ���ӳ��"""
    if decodebook is None:
        decodebook = {}
    
    # Ҷ�ӽڵ㣺�洢���뵽�ַ���ӳ��
    if root.left is None and root.right is None:
        for char in root.chars:
            decodebook[path] = char
    else:
        # �ݹ鴦����������
        build_decode(root.left, path + '0', decodebook)
        build_decode(root.right, path + '1', decodebook)
    
    return decodebook

def encode_string(s, codebook):
    """�����ַ�������ÿ���ַ��滻Ϊ��Ӧ�Ĺ���������"""
    return ''.join([codebook[char] for char in s])

def decode_string(s, decodebook):
    """�����ַ�����
    1. �����ȡbit������ǰ����
    2. ������ƥ������ʱ�����Ӧ�ַ�������"""
    ans = []
    cur_code = ''
    for bit in s:
        cur_code += bit
        if cur_code in decodebook:
            ans.append(decodebook[cur_code])
            cur_code = ''  # ���õ�ǰ����
    return ''.join(ans)


n = int(input())  
char_weight = {}  
for _ in range(n):
    char, weight = input().split()
    char_weight[char] = int(weight)

# �������������ͱ�������
root = build_huffman(char_weight)
codebook = build_code(root)
decodebook = build_decode(root)

# �����ѯֱ��EOF
while True:
    try:
        query = input()
        
        # �ж������Ǳ��봮���ǽ��봮
        if all(c in ('0', '1') for c in query):
            # ����01��
            print(decode_string(query, decodebook))
        else:
            # �����ַ�
            print(encode_string(query, codebook))
            
    except EOFError:  # �����ļ�������
        break
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-16%20090902.png?raw=true)

## 2. ѧϰ�ܽ���ջ�

���6�����ҵ��������8�⣿ ���翪ʼ����ҵ��������ɵ������ˢ�·�����ҵ��Ŀ�и��࣬�������һ�⣬�ټ���Ҳȥ����sy���⣬�����Ŀ����15�������ģ���ʱҪ�ύ�����ֻ���ά�ޣ�ֻ�ܲ鿴����е���Ŀ���������ύ

�����ҵ�Ĵ��볤�ȶ��ܳ����󲿷ֲ����ѣ�ֻ��̫������debug���ֻ��߻������Ӳ�������huffman��һ��ʼ���Ǻ��˽⣬�������˲�ѧ�ᣬ�����Ŀ��һЩ����Ҫ˼���þò�������⣬������������������������ȫ���˽⣨�����ң��������ݣ���ֽ��д���˶������ݲ��˽⣨ϣ���������ݸ��̵㡣����

���л���һ�ţ�ϣ��51��ʱ��úø�ϰ�������������

<mark>���������ҵ��Ŀ��Լ򵥣��з�Ѱ�Ҷ������ϰ��Ŀ���硰����2025springÿ��ѡ������LeetCode��Codeforces����ȵ���վ�ϵ���Ŀ��</mark>