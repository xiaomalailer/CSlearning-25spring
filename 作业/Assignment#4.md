# Assignment #4: λ������ջ�������Ѻ�NN

Updated 1203 GMT+8 Mar 10, 2025

2025 spring, Complied by <mark>��PȨ Ԫ��</mark>





## 1. ��Ŀ

### 136.ֻ����һ�ε�����

bit manipulation, https://leetcode.cn/problems/single-number/



<mark>����λ������ʵ�֣�����ֻʹ�ó�������ռ䡣</mark>

*λ�����У������˳���޹أ����㽻���ɺͽ���ɣ���ͬ����0��0��a����a*

���룺

```python
from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> int:

        x=0
        for num in nums:
            x^=num
        return x


s=Solution()
nums=list(map(int,input().split()))
print(s.singleNumber(nums))
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-11%20163230.png?raw=true)




### 20140:���ջ�ѧ����

stack, http://cs101.openjudge.cn/practice/20140/



˼·��

����������֤�ķ�������⵽��]�������ţ�����ȡ�������ĸ�����ּӽ�ȥ��������[��ֹͣ��Ȼ����ĸ���Ƽӽ�ȥջ���Խ����¸�����ʱ��һ����



���룺

```python
s=input()
stack=[]
tmp=[]
ans=''
for i in range(len(s)):
    stack.append(s[i])
    if s[i]==']':
        stack.pop()
        while stack[-1]!='[':
            tmp.append(stack.pop())
        stack.pop()
        ans=''
        while tmp[-1].isdigit():
            ans+=str(tmp.pop())
        tmp=tmp*(int(ans))
        while tmp:
            stack.append(tmp.pop())
print(''.join(stack))

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-11%20173114.png?raw=true)



### 160.�ཻ����

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/



˼·��

˫ָ�뷽����A,Bͬʱ��������֮���������ֱ���ཻ���ǵ�һ���ཻλ�㣬ԭ����A�������������a�ٴ�B�ߵ��ཻ��C����b-c�ܹ�a+b-c;��B����������A��һ���ģ�b+a-c

���룺

```python
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
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-11%20221945.png?raw=true)



### 206.��ת����

linked list, https://leetcode.cn/problems/reverse-linked-list/



˼·��

˫ָ�루��ʵ�����ˣ�������ָ�뿪ͷ��β����ת����ԭ�еĿ�ͷ����ȥ��β������ͷ��ɽ�β���¸���ͷ�ͱ��ԭ���ĵڶ���

1->2->3->none

�� 1->none 2->3->none

�� 2->1->none 3->none

�� 3->2->1->none 

���룺

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        start,end=head,None
        while start:
            tmp=start.next
            start.next=end
            end=start
            start=tmp
        return end
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-11%20221945.png?raw=true)



### 3478.ѡ��������K��Ԫ��

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/



˼·��

�������� nums1����֤����ʱǰ��� nums1[j] ����С�ڵ�ǰ nums1[i]���öѣ�heapq��ά�� k ����� nums2[j]����֤ sum_k ������ k �����ĺ�(����ѵĴ�С���� k���Ƴ���С��Ԫ��),���� nums1���𲽸��� s ����¼��

˵ʵ��������ⶼ�����׵����֣�Ȼ��͸������������ߵ�����������

���������������Լ��������

nums1=[2,3,5,1];nums2=[3,4,2,10],k=2

��һ�������nums1�� nums1[0]=1��û�бȴ˸�С��nums[j]������ԭλΪ3������ans[3]=0����λ����nums2��Ϊ10��������С�Ѻ͵�ǰ��

�ڶ���nums1[1]=2,ֻ����һ����֮С������+10��ans[0]=10(ԭnums1��2��λ�ö�Ӧnums2λ��ֵ�����ܺ��ټӶ�Ӧ��nums2��ֵ3=13��Ҳ�����¸�nums1[i]>nums1[j]ʱ�ܵõ���ֵ

������ nums1[2]=3,���Զ�Ӧδ֪ans[1]=13,�ܺ�����13+4=17��������Ϊk=2����Ҫ���ֻ���޳�С��ֵ��ͨ��heapq����ʱΪ[3,4,10]��С��ֵ�޳���17-3��=14

���ԣ�nums1[3]=5 ��Ӧans[2]=14

�������[10,13,14,0]

���룺

```python
class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        a = sorted((x, y, i) for i, (x, y) in enumerate(zip(nums1, nums2)))
        ans=[0]*len(nums1)
        min_heap=[]
        sum_k=0 s #�ǵ�ǰ k ����� nums2[j] ֮��
        i=0
        while i<len(nums1):
            start=i
            x=a[start][0]
            while i<len(nums1) and a[i][0]==x: #�ҵ�������ͬ�� nums1[i]����Ϊ a �Ѿ����������
                ans[a[i][2]]=sum_k #a[i][2] �� nums1[i] ��ԭ���������������Ҫ�� sum_k ��ֵ�����λ�õ� answer
                i+=1
            for j in range(start,i):
                y=a[j][1]
                sum_k+=y
                heapq.heappush(min_heap,y)
                if len(min_heap)>k:
                    sum_k-=heapq.heappop(min_heap)
        return ans

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-11%20234031.png?raw=true)




### Q6.�������ӻ�neural network

https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises

**Your task:** configure a neural network that can separate the orange dots from the blue dots in the diagram, achieving a loss of less than 0.2 on both the training and test data.

**Instructions:**

In the interactive widget:

1. Modify the neural network hyperparameters by experimenting with some of the following config settings:
   - Add or remove hidden layers by clicking the **+** and **-** buttons to the left of the **HIDDEN LAYERS** heading in the network diagram.
   - Add or remove neurons from a hidden layer by clicking the **+** and **-** buttons above a hidden-layer column.
   - Change the learning rate by choosing a new value from the **Learning rate** drop-down above the diagram.
   - Change the activation function by choosing a new value from the **Activation** drop-down above the diagram.
2. Click the Play button above the diagram to train the neural network model using the specified parameters.
3. Observe the visualization of the model fitting the data as training progresses, as well as the **Test loss** and **Training loss** values in the **Output** section.
4. If the model does not achieve loss below 0.2 on the test and training data, click reset, and repeat steps 1�C3 with a different set of configuration settings. Repeat this process until you achieve the preferred results.

��������Լ��������<mark>��ͼ</mark>����˵��ѧϰ���ĸ����ԭ��
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-12%20015502.png?raw=true)


���ǲ�̫���ף�������ѡ�������ⷽ��û���о���ѧϰ�������ǵ㿪������������������ǿ��Կ���һЩ�仯������ҵ���õĲ�������������������ͼ��������ǰ�Ӵ��Ļ���ѧϰ��ͼʾӦ����ͬһ�����������Ȥ���л����о�

## 2. ѧϰ�ܽ���ջ�

�����ҵ�еڶ���һֱ��С���⣬���˺�Щʱ�����������ٿ�ÿ��ѡ���нӴ������Բ�İ�������һ���˼·���ѣ�����ʱ����������ɣ��ö�������

����е�æ����ʼ������ÿ��ѡ������tough���⻹�����Զ������

<mark>���������ҵ��Ŀ��Լ򵥣��з�Ѱ�Ҷ������ϰ��Ŀ���硰����2025springÿ��ѡ������LeetCode��Codeforces����ȵ���վ�ϵ���Ŀ��</mark>