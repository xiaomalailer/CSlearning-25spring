# Assignment #4: 位操作、栈、链表、堆和NN

Updated 1203 GMT+8 Mar 10, 2025

2025 spring, Complied by <mark>马凱权 元培</mark>





## 1. 题目

### 136.只出现一次的数字

bit manipulation, https://leetcode.cn/problems/single-number/



<mark>请用位操作来实现，并且只使用常量额外空间。</mark>

*位操作中，异或与顺序无关，满足交换律和结合律，相同异或得0，0与a异或得a*

代码：

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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-11%20163230.png?raw=true)




### 20140:今日化学论文

stack, http://cs101.openjudge.cn/practice/20140/



思路：

利用括号验证的方法，检测到“]”闭括号，就拿取最靠近的字母与数字加进去（碰到“[”停止，然后字母复制加进去栈，以进行下个复制时能一起复制



代码：

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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-11%20173114.png?raw=true)



### 160.相交链表

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/



思路：

双指针方法，A,B同时遍历，且之后交替遍历，直到相交就是第一个相交位点，原因是A遍历完假设走了a再从B走到相交处C花了b-c总共a+b-c;而B遍历完再走A是一样的，b+a-c

代码：

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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-11%20221945.png?raw=true)



### 206.反转链表

linked list, https://leetcode.cn/problems/reverse-linked-list/



思路：

双指针（其实到三了），两个指针开头结尾，倒转就是原有的开头连接去结尾，而开头变成结尾，下个开头就变成原本的第二个

1->2->3->none

① 1->none 2->3->none

② 2->1->none 3->none

③ 3->2->1->none 

代码：

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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-11%20221945.png?raw=true)



### 3478.选出和最大的K个元素

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/



思路：

首先排序 nums1，保证遍历时前面的 nums1[j] 总是小于当前 nums1[i]；用堆（heapq）维护 k 个最大 nums2[j]，保证 sum_k 是最大的 k 个数的和(如果堆的大小超过 k，移除最小的元素),遍历 nums1，逐步更新 s 并记录答案

说实话看了题解都不明白的那种，然后就跟着例子慢慢走倒是清楚了许多

举例：以下是我自己想的数据

nums1=[2,3,5,1];nums2=[3,4,2,10],k=2

第一步排序号nums1后 nums1[0]=1，没有比此更小的nums[j]，因其原位为3，所以ans[3]=0，此位置在nums2中为10，加入最小堆和当前和

第二：nums1[1]=2,只有上一个比之小，所以+10，ans[0]=10(原nums1中2的位置对应nums2位置值），总和再加对应在nums2的值3=13，也就是下个nums1[i]>nums1[j]时能得到的值

第三： nums1[2]=3,所以对应未知ans[1]=13,总和增至13+4=17，但是因为k=2，又要最大，只能剔除小的值，通过heapq（此时为[3,4,10]把小的值剔除（17-3）=14

所以：nums1[3]=5 对应ans[2]=14

结果就是[10,13,14,0]

代码：

```python
class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        a = sorted((x, y, i) for i, (x, y) in enumerate(zip(nums1, nums2)))
        ans=[0]*len(nums1)
        min_heap=[]
        sum_k=0 s #是当前 k 个最大 nums2[j] 之和
        i=0
        while i<len(nums1):
            start=i
            x=a[start][0]
            while i<len(nums1) and a[i][0]==x: #找到所有相同的 nums1[i]（因为 a 已经排序过）。
                ans[a[i][2]]=sum_k #a[i][2] 是 nums1[i] 在原数组的索引，我们要把 sum_k 赋值给这个位置的 answer
                i+=1
            for j in range(start,i):
                y=a[j][1]
                sum_k+=y
                heapq.heappush(min_heap,y)
                if len(min_heap)>k:
                    sum_k-=heapq.heappop(min_heap)
        return ans

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-11%20234031.png?raw=true)




### Q6.交互可视化neural network

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
4. If the model does not achieve loss below 0.2 on the test and training data, click reset, and repeat steps 1–3 with a different set of configuration settings. Repeat this process until you achieve the preferred results.

给出满足约束条件的<mark>截图</mark>，并说明学习到的概念和原理。
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-12%20015502.png?raw=true)


还是不太明白，本来当选做，对这方面没有研究和学习，但还是点开看看，随意调参数倒是可以看到一些变化，最后找到最好的差不多就是这样，不过这种图形与我以前接触的机器学习的图示应该是同一个概念，倒是有趣，有机会研究

## 2. 学习总结和收获

这次作业中第二题一直有小问题，花了好些时间解决；链表寒假靠每日选做有接触，所以不陌生，最后一题的思路好难，考试时很难想出来吧，得多练才能

最近有点忙，开始跟不上每日选做，且tough的题还是难以独自完成

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>