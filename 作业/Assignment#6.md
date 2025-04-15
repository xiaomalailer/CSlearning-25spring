# Assignment #6: ���ݡ�����˫������͹�ϣ��

Updated 1526 GMT+8 Mar 22, 2025

2025 spring, Complied by <mark>��PȨ Ԫ��</mark>



## 1. ��Ŀ

### LC46.ȫ����

backtracking, https://leetcode.cn/problems/permutations/

˼·��

���õݹ飨����dfs���ķ�ʽ���ȫ���У�����������е�����δ���㹻���ȣ���ݹ�����¸����֣�����Ϊ�����ظ���������Ҫ����used���ж�

�������ӣ�[1,2,3]

[]+1+2+3 =[1,2,3] ���ص�[1] �� [1]+3+2=[1,3,2] ����[2,1,3],[2,3,1]������



���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-24%20173055.png?raw=true)




### LC79: ��������

backtracking, https://leetcode.cn/problems/word-search/

˼·��

��dfs��backtrack�����ͺ���̽���Թ������ֿ�ͷ��ʼ�����������ң�

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

!{Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-24%20220638.png?raw=true)



### LC94.���������������

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

˼·��

��ν����������������������һֱ���±���ֱ�����������󡱡�����������ٽ������𰸣����������Ҹ���������������֮���ٷ�����һ�����ظ���

���Կ����Ķ���չʾ��������

![SET](https://assets.leetcode-cn.com/solution-static/94/9.png)

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-24%20222154.png?raw=true)



### LC102.�������Ĳ������

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

˼·��

bfs,Ҳ��������deque����ʽ���е�����У����߾��ǣ������Ȱѵ�һ����ӣ��ٳ��Ӽӽ��𰸣��ٰ���һ����ӡ����ӣ������б����԰ѽ�������һ��ȫ����ӡ������Դ�����

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-24%20224816.png?raw=true)



### LC131.�ָ���Ĵ�

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

˼·��

dfs����Ϊ����ʱ�临�Ӷ�С���������ҳ��������ҳ��Ļ����ַ���������ÿ���־���һ�����������ͷβ��ͬ�����Ҳ�ǻ��ģ�s[i+1][j-1])������abba�ǻ��ĵ���������ͷβ��ͬ���м�bb���ģ�Ȼ����ͨ��dfs�ѻ����ַ��������

����s="aab"

���ҳ����ģ�

palindro[0][0] = True:  'a'

palindro[1][1] = True: 'a'

palindro[2][2] = True: "b"

palindro[0][1] = True: "aa"

��dfs��

dfs(0): ѡ "a" �� dfs(1)

dfs(1): ѡ "a" �� dfs(2)

dfs(2): ѡ "b" �� dfs(3)  �� ��¼ ["a", "a", "b"]

���ݣ�
dfs(1): ѡ "aa" �� dfs(2)

dfs(2): ѡ "b" �� dfs(3) �� ��¼ ["aa", "b"]


���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-24%20232534.png?raw=true)




### LC146.LRU����

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

˼·��

����������ʾ��ɣ���������鱾�ĳ�����Ա�ʮ�������dummy��һ��ͬʱ����ͷ��β������Ľڵ㣬next����ͷprev����β�����Ҫ�ŵ���ͷ��ɾ����β����Ҫdummy��æ

![](https://pic.leetcode.cn/1696039105-PSyHej-146-3-c.png)

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-25%20113923.png?raw=true)



## 2. ѧϰ�ܽ���ջ�

�����Ŀ����dfsbfs��Ŀ����Ҳ����̫�ѣ���һ����ʽ��ѭ������������ֻ����ʽ��������ͬ�������ݴ�����bfsdfs��ݹ����ʽ����������������ѣ��������⵹��Ҫ����ʱ�������Ŀ���м��⿴�˵���ʾ��ɣ����һ�⵹�����ѵģ������ϴε�url��д����Ҳ��̫�У�ֱ�����������������ô����ô���������ǿ�����⡣

��������listnode�������Ŀ���е��ѣ����ڸ�ʽ�ϲ�֪����ôд������֮��ڵ��ϵ��ô���������õ��ĺ�������ֹ��Ŀ�ṩ����Щ���ϻ�����Ҫ��������ѵ��ң��������˼��θо�������һ����ʽ������

���Խ��æµ��ѡ��Ҳ�þ�û���������ҵ��������ͷ�Ҳ�ǵȵ��ܶ��ſ���4��Ҫ���ˣ����п�Ҫ���ˣ�ϣ����ƽ��ã��������ȡ�

<mark>���������ҵ��Ŀ��Լ򵥣��з�Ѱ�Ҷ������ϰ��Ŀ���硰����2025springÿ��ѡ������LeetCode��Codeforces����ȵ���վ�ϵ���Ŀ��</mark>