# Assignment #5: 链表、栈、队列和归并排序

Updated 1348 GMT+8 Mar 17, 2025

2025 spring, Complied by <mark>马凱权 元培</mark>





## 1. 题目

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：

准版一个新链表作为答案，比较两个链表当下值谁小，新链表指向谁，以此继续，直到其中一个链表到头，另一个链表剩余的就都放进去答案链表

代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%9)5%E6%88%AA%E5%9B%BE%202025-03-18%20150804.png?raw=true




### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>

这题要注意的细节有点多，首先是保证[]和[1]算回文，然后就是确定快慢指针的速度，以前用过的先让fast=head.next.next在这里有问题，比如有一个数据[1,0,3,4,0,1]，slow会停在4，而后面反转=1，0,所以采用fast走两步slow走一步，而判定fast.next 和fast.next.next即可，找到中间对半分后尾部链表的开头，将其反转，并与前半部链表比对即可


代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-18%20155351.png?raw=true）



### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双链表实现。</mark>

双链表实现：创建一个链表，有双指针，指向过去与未来

visit部分：创建一个临时链表，临时链表贮存新的url，让现在的链表的next（尾部）指向这个临时链表（新的url）【断掉原本的尾部】，临时链表的头部=现在的链表，从而临时列表等于原有链表的头+新url

back和forward都比较好动，使用双指针prev和next就可以达到



代码：

双链表做法

```python
class ListNode:
    def __init__(self,url:str):
        self.url=url
        self.prev=None
        self.next=None


class BrowserHistory:

    def __init__(self, homepage: str):
        self.cur=ListNode(homepage)

    def visit(self, url: str) -> None:
        new_node=ListNode(url)
        self.cur.next=new_node
        new_node.prev=self.cur
        self.cur=new_node

    def back(self, steps: int) -> str:
        while steps>0 and self.cur.prev is not None:
            self.cur=self.cur.prev
            steps-=1
        return self.cur.url

    def forward(self, steps: int) -> str:
        while steps>0 and self.cur.next is not None:
            self.cur=self.cur.next
            steps-=1
        return self.cur.url

```


双栈做法
```python
class BrowserHistory:

    def __init__(self, homepage: str):
        self.cur=[]
        self.forwad=[]
        self.visit(homepage)

    def visit(self, url: str) -> None:
        self.cur.append(url)
        self.forwad.clear()

    def back(self, steps: int) -> str:
        while steps and len(self.cur)>1:
            self.forwad.append(self.cur.pop())
            steps-=1
        return self.cur[-1]

    def forward(self, steps: int) -> str:
        while steps and self.forwad:
            self.cur.append(self.forwad.pop())
            steps-=1
        return self.cur[-1]


```


list做法
```python
class BrowserHistory:

    def __init__(self, homepage: str):
        self.history=[homepage]
        self.cur=0

    def visit(self, url: str) -> None:
        self.cur+=1
        del self.history[self.cur:]
        self.history.append(url)

    def back(self, steps: int) -> str:
        self.cur=max(0,self.cur-steps)
        return self.history[self.cur]


    def forward(self, steps: int) -> str:
        self.cur=min(len(self.history)-1,self.cur+steps)
        return self.history[self.cur]



# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-18%20163948.png?raw=true)




### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：

由于数字可能含有小数点或超过个位数，需要先把这个可能性处理，即只要是数字下个还是数字或者小数点，就合并起来（看作字符）

如果不是数字，是+-*/，先丢去一个栈里，如果原栈顶的符号是+-乘/，且这个符号优先级比将要存如栈的高或相等，那就先把原栈顶的符号加到答案序列中

例子： 1*2+3 ，序列=[1,2],在遇到“+”时，原栈顶有了优先级比较高的符号，那就先把原栈顶的符号加入序列【1，2，×】

接着要处理的是括号问题，括号里的数字要单独处理，所以遇到左括号入栈后一旦遇到右括号，左括号之后的都要加入答案序列

代码：

```python

def infix_to_postfix(infix):
    num=''
    postfix=[]
    stack_cal=[]
    cal={'+':1,'-':1,'*':2,'/':2}
    for char in infix:
        if char.isnumeric() or char=='.':
            num+=char
        else:
            if num:
                num=float(num)
                postfix.append(int(num) if num.is_integer() else num)
                num=''
            if char in '+-*/':
                while stack_cal and stack_cal[-1] in '+-*/' and cal[stack_cal[-1]]>=cal[char]:
                    postfix.append(stack_cal.pop())
                stack_cal.append(char)
            elif char=='(':
                stack_cal.append(char)
            elif char==')':
                while stack_cal and stack_cal[-1]!='(':
                    postfix.append(stack_cal.pop())
                stack_cal.pop()
    if num:
        num = float(num)
        postfix.append(int(num) if num.is_integer() else num)

    while stack_cal:
        postfix.append(stack_cal.pop())

    return ' '.join(str(x) for x in postfix)




n=int(input())
for _ in range(n):
    exp=input()
    print(infix_to_postfix(exp))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-18%20171105.png?raw=true)




### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>

队列实现：先编码1-n，然后从第p号开始，所以把1到p-1号加到队列尾端，然后开始报数，如果报数非m，那就出队再入队，如果是m，那就直接出队加到答案中

代码：

```python
while True:
    n,p,m=map(int,input().split())
    if n==0 and p==0 and m==0:
        break
    kids=[i for i in range(1,n+1)]
    for _ in range(p-1):
        out=kids.pop(0)
        kids.append(out)

    cnt=0
    ans=[]
    while len(kids)>1:
        out=kids.pop(0)
        cnt+=1
        if cnt==m:
            ans.append(out)
            cnt=0
            continue
        kids.append(out)
    ans.append(kids.pop(0))
    print(','.join(map(str,ans)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-18%20173254.png?raw=true)



### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：

"逆序对"：二分法，以便两对两对比较和排序（归并排序），方便找出所有“逆序对”，最后会出现一个逆序序列，可按照如下例子参照代码即清楚明了

例子：1 5 5 7 6 

left= 1,5 （5能超1） 返回 （5，1）,1

right= 5 | 7 ,6 

对于 7，6 (6不难超1)，返回（7，6），0

5 ： （7，6）：5<7 , 5<6 返回(7,6,5),2

(5,1) :(7,6,5) : 5<7,5<6,1<7,1<6,1<5: 返回（7，6，5，5，1），5

ans=1+2+5=8


代码：

```python

def merge_sort(run):
    if len(run) <= 1:
        return run,0
    mid=len(run)//2
    left,left_cnt=merge_sort(run[:mid])
    right,right_cnt=merge_sort(run[mid:])
    run,merge_cnt=merge(left,right)
    return run,left_cnt+right_cnt+merge_cnt

def merge(left,right):
    merged=[]
    left_idx,right_idx=0,0
    cnt=0
    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] >= right[right_idx]:
            merged.append(left[left_idx])
            left_idx+=1
        else:
            merged.append(right[right_idx])
            right_idx+=1
            cnt+=len(left)-left_idx
    merged+=left[left_idx:]+right[right_idx:]
    return merged,cnt

n=int(input())
run=list(int(input()) for _ in range(n))
run,ans=merge_sort(run)
print(ans)
   
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-18%20193355.png?raw=true)



## 2. 学习总结和收获

这次作业第1，2，4都好像做过（寒假选做？），但是第二题规定用快慢指针，就会比较麻烦，第三题要求双链表也是，但是链表本身不难，就是连接关系，容易在纸上推出，只是正规的写法难度比较高，

第四题是栈？有点括号匹配的味道，第五题约瑟夫2跟约瑟夫1没差太多，考察队列的使用，让我想起上学期一道监狱的题，第六题难度高，看了答案很久也需要一点时间推导和理解，是一种没学过的排序，二分法的概念常见但都比较难，得多练习，方便后面面对“树”

这次题目有点升级了，难度高了点，但特别新颖的算法倒没有，链表是假期接触到的，寒假的那些题也挺有意思，可以再复习复习

每日选做彻底跟不上，也还有接近一个月要期中，能赶多少多少吧

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>