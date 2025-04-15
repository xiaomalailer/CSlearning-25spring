# Assignment #5: ����ջ�����к͹鲢����

Updated 1348 GMT+8 Mar 17, 2025

2025 spring, Complied by <mark>��PȨ Ԫ��</mark>





## 1. ��Ŀ

### LC21.�ϲ�������������

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

˼·��

׼��һ����������Ϊ�𰸣��Ƚ�����������ֵ˭С��������ָ��˭���Դ˼�����ֱ������һ������ͷ����һ������ʣ��ľͶ��Ž�ȥ������

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%9)5%E6%88%AA%E5%9B%BE%202025-03-18%20150804.png?raw=true




### LC234.��������

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>���ÿ���ָ��ʵ�֡�</mark>

����Ҫע���ϸ���е�࣬�����Ǳ�֤[]��[1]����ģ�Ȼ�����ȷ������ָ����ٶȣ���ǰ�ù�������fast=head.next.next�����������⣬������һ������[1,0,3,4,0,1]��slow��ͣ��4�������淴ת=1��0,���Բ���fast������slow��һ�������ж�fast.next ��fast.next.next���ɣ��ҵ��м�԰�ֺ�β������Ŀ�ͷ�����䷴ת������ǰ�벿����ȶԼ���


���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-18%20155351.png?raw=true��



### LC1472.����������ʷ��¼

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>����˫����ʵ�֡�</mark>

˫����ʵ�֣�����һ��������˫ָ�룬ָ���ȥ��δ��

visit���֣�����һ����ʱ������ʱ���������µ�url�������ڵ������next��β����ָ�������ʱ�����µ�url�����ϵ�ԭ����β��������ʱ�����ͷ��=���ڵ������Ӷ���ʱ�б����ԭ�������ͷ+��url

back��forward���ȽϺö���ʹ��˫ָ��prev��next�Ϳ��Դﵽ



���룺

˫��������

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


˫ջ����
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


list����
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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-18%20163948.png?raw=true)




### 24591: ������ʽת������ʽ

stack, http://cs101.openjudge.cn/practice/24591/

˼·��

�������ֿ��ܺ���С����򳬹���λ������Ҫ�Ȱ���������Դ�����ֻҪ�������¸��������ֻ���С���㣬�ͺϲ������������ַ���

����������֣���+-*/���ȶ�ȥһ��ջ����ԭջ���ķ�����+-��/��������������ȼ��Ƚ�Ҫ����ջ�ĸ߻���ȣ��Ǿ��Ȱ�ԭջ���ķ��żӵ���������

���ӣ� 1*2+3 ������=[1,2],��������+��ʱ��ԭջ���������ȼ��Ƚϸߵķ��ţ��Ǿ��Ȱ�ԭջ���ķ��ż������С�1��2������

����Ҫ��������������⣬�����������Ҫ������������������������ջ��һ�����������ţ�������֮��Ķ�Ҫ���������

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-18%20171105.png?raw=true)




### 03253: Լɪ������No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>���ö���ʵ�֡�</mark>

����ʵ�֣��ȱ���1-n��Ȼ��ӵ�p�ſ�ʼ�����԰�1��p-1�żӵ�����β�ˣ�Ȼ��ʼ���������������m���Ǿͳ�������ӣ������m���Ǿ�ֱ�ӳ��Ӽӵ�����

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-18%20173254.png?raw=true)



### 20018: ����������ԽҰ��

merge sort, http://cs101.openjudge.cn/practice/20018/

˼·��

"�����"�����ַ����Ա��������ԱȽϺ����򣨹鲢���򣩣������ҳ����С�����ԡ����������һ���������У��ɰ����������Ӳ��մ��뼴�������

���ӣ�1 5 5 7 6 

left= 1,5 ��5�ܳ�1�� ���� ��5��1��,1

right= 5 | 7 ,6 

���� 7��6 (6���ѳ�1)�����أ�7��6����0

5 �� ��7��6����5<7 , 5<6 ����(7,6,5),2

(5,1) :(7,6,5) : 5<7,5<6,1<7,1<6,1<5: ���أ�7��6��5��5��1����5

ans=1+2+5=8


���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>

![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-18%20193355.png?raw=true)



## 2. ѧϰ�ܽ���ջ�

�����ҵ��1��2��4����������������ѡ�����������ǵڶ���涨�ÿ���ָ�룬�ͻ�Ƚ��鷳��������Ҫ��˫����Ҳ�ǣ������������ѣ��������ӹ�ϵ��������ֽ���Ƴ���ֻ�������д���ѶȱȽϸߣ�

��������ջ���е�����ƥ���ζ����������Լɪ��2��Լɪ��1û��̫�࣬������е�ʹ�ã�����������ѧ��һ���������⣬�������Ѷȸߣ����˴𰸺ܾ�Ҳ��Ҫһ��ʱ���Ƶ�����⣬��һ��ûѧ�������򣬶��ַ��ĸ���������Ƚ��ѣ��ö���ϰ�����������ԡ�����

�����Ŀ�е������ˣ��Ѷȸ��˵㣬���ر���ӱ���㷨��û�У������Ǽ��ڽӴ����ģ����ٵ���Щ��Ҳͦ����˼�������ٸ�ϰ��ϰ

ÿ��ѡ�����׸����ϣ�Ҳ���нӽ�һ����Ҫ���У��ܸ϶��ٶ��ٰ�

<mark>���������ҵ��Ŀ��Լ򵥣��з�Ѱ�Ҷ������ϰ��Ŀ���硰����2025springÿ��ѡ������LeetCode��Codeforces����ȵ���վ�ϵ���Ŀ��</mark>