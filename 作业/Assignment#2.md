# Assignment #2: ���ѧϰ�������ģ��

Updated 2204 GMT+8 Feb 25, 2025

2025 spring, Complied by <mark>��PȨ Ԫ��ѧԺ</mark>





## 1. ��Ŀ

### 18161: ��������

matrices, http://cs101.openjudge.cn/practice/18161



˼·��

�������㣬���ж��ܲ�����ɣ���A����B���Ƿ��ͬ���˷��������γɵ��¾���A��B���Ƿ�ͬC�����ٽ������㣻�˷���������*�У���A��һ�г�B��һ�С�A��һ�С�B�ڶ���...�ٵ�A��һ�У����������������A���С�A���� B���У�

���룺

```python
A=[]
B=[]
C=[]
r_a,c_a=map(int,input().split())
for _ in range(r_a):
    A.append(list(map(int,input().split())))

r_b,c_b=map(int,input().split())
for _ in range(r_b):
    B.append(list(map(int,input().split())))

r_c,c_c=map(int,input().split())
for _ in range(r_c):
    C.append(list(map(int,input().split())))

if c_a!=r_b or r_a!=r_c or c_b!=c_c:
    print('Error!')
    exit()
D=[[0]*c_b for _ in range(r_a)]
for i in range(r_a):
    for j in range(c_b):
        for k in range(c_a):
            D[i][j]+=A[i][k]*B[k][j]
        D[i][j]+=C[i][j]
for ans in D:
    print(*ans)
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-26%20092359.png?raw=true)




### 19942: ��ά�����ϵľ������

matrices, http://cs101.openjudge.cn/practice/19942/




˼·��

���ڰ�һ��for����һ���ѿ������Կ��������������㣬���������Ҫ�仯������ԭ������Ҫ�䣬�����ڽ�������i��jʱ��ԭ����ӦҲ��i��j��ʼ���㣬�Ҳ�����p��q


���룺

```python

def juan(x,y):
    ans=0
    for i in range(p):
        for j in range(q):
            ans+=matrix_1[i+x][j+y]*matrix_2[i][j]
    return ans

m,n,p,q=map(int,input().split())
matrix_1=[]
matrix_2=[]
for i in range(m):
    matrix_1.append(list(map(int,input().split())))

for i in range(p):
    matrix_2.append(list(map(int,input().split())))

matrix_3=[[0]*(n+1-q) for _ in range(m+1-p)]

for i in range(m+1-p):
    for j in range(n+1-q):
        matrix_3[i][j]=juan(i,j)
for row in matrix_3:
    print(*row)

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-26%20095841.png?raw=true)




### 04140: �������

ţ�ٵ�������http://cs101.openjudge.cn/practice/04140/

����<mark>ţ�ٵ�����</mark>ʵ�֡�

��Ϊ������ģ�͵�ѵ���������漰�����ݶ��½���������֣���SGD��Adam�ȣ��������Ż�ģ�Ͳ�������С����ʧ���������ַ�������ͨ�������ķ�ʽ�𲽽ӽ����Ž⡣ÿһ�ε��������ڵ�ǰ��ľֲ���Ϣ������������ͼ�ҵ�һ���ȵ�ǰ����ŵ��µ㡣���ţ�ٵ����������������������ݶȵ��Ż��㷨�Ĺ���ԭ���ر�������������õ�����Ϣ���о��ߡ�

> **ţ�ٵ�����**
>
> - **Ŀ��**����Ҫ����Ѱ��һ������ $f(x)$ �ĸ������ҵ����� $f(x)=0$ �� $x$ ֵ��������ͨ���ʵ��任Ŀ�꺯������Ҳ��������Ѱ�Һ����ļ�ֵ��
> - **��������**������̩�ռ�����һ�׺Ͷ�����������Ŀ�꺯������ÿ�ε�����ʹ��Ŀ�꺯�����䵼������Ϣ��������һ���ķ���Ͳ�����
> - **������ʽ**��$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $ ������ֵ���⣬�����ת��Ϊ$ x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)} $������ $f'(x)$ �� $f''(x)$ �ֱ���Ŀ�꺯����һ�׵����Ͷ��׵�����
> - **�ص�**��ţ�ٷ�ͨ�����и���������ٶȣ������Ƕ��ڶ��ο�΢��������������Ҫ����Ŀ�꺯���Ķ��׵�����Hessian�����ڶ�ά����£������ҶԳ�ʼ���ѡ���Ϊ���С�
>
> **�ݶ��½���**
>
> - **Ŀ��**��ֱ������Ѱ�Һ�������Сֵ��Ҳ����ͨ��ȡ��Ѱ�����ֵ���������ڻ���ѧϰ����Ӧ�ù㷺��
> - **��������**����������Ŀ�꺯����һ�׵�����Ϣ�����ݶȣ��������ݶȵķ������ƶ��Դﵽ���ٺ���ֵ��Ŀ�ġ�
> - **������ʽ**��$ x_{n+1} = x_n - \alpha \cdot \nabla f(x_n) $ ���� $\alpha$ ��ѧϰ�ʣ�$\nabla f(x_n)$ ��ʾĿ�꺯���� $x_n$ ����ݶȡ�
> - **�ص�**���ݶ��½�����Ҫ���㸴�ӵĶ��׵���������ڸ�ά�ռ����������ʵ�֡�Ȼ�������������ٶ�ͨ���������ر��ǵ�Ŀ�꺯���ĵȸ��߳��ֳ���Բ����Բ��ʱ������������������������
>
> **��ͬ�벻ͬ**
>
> - **��ͬ��**�����߶��������Ż����⣬��ͼ�ҵ������ļ�Сֵ�㣻����ҪĿ�꺯������һ�׿ɵ���
> - **��ͬ��**��
>   - ţ�ٷ�ʹ���˸���ľֲ���Ϣ�������׵���������������������ٶȸ��죬����ʵ��Ӧ���п��ܻ���������ɱ��ߡ����Դ�����ģ���ݼ������⡣
>   - �ݶ��½����Ϊ�򵥣�����ʵ�֣��ر����ڸ�ά�ռ��У�������ֻʹ����һ�׵�����Ϣ���������ٶȿ��ܽ������������ڽӽ���ֵ��ʱ��
>

�����һ�׵����Ͷ��׵�������ò�Ʋ�̫�У�Ӧ����ԭ������һ�׵������̣���ʼֵ����1��2��3��4��5..���������1e-6������������ֵ����ʵ��x_n��x_n-1�Ĳ�ֵ����С������ֵ�򲻼�����100��Ϊ����������

ţ�ٷ��ĺ���˼���������߱ƽ����ߣ�Ȼ�������ߵ������Ϊ��һ�����ƽ�

���룺

```python

def newton_new(x):
    for i in range(100):
       d=f_x(x)/f_x_2(x)
       x=x-d
       if abs(d)<1e-6:
           break
    return x


f_x=lambda x: x**3-5*(x**2)+10*x-80
f_x_2=lambda x:3*(x**2)-10*x+10
f_x_3=lambda x:6*x-10

x_0=1.0
print('{:.9f}'.format(newton_new(x_0)))
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-26%20104340.png?raw=true)




### 06640: ��������

data structures, http://cs101.openjudge.cn/practice/06640/



˼·��

���ֵ䷽���ֵ��ѯ������ʹ��setdefault�Ѵʱ�ż�¼�ã�������ʳ�����ĳ�������ͽ��ñ�ż����������


���룺

```python
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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-25%20230422.png?raw=true)




### 04093: ����������ѯ

data structures, http://cs101.openjudge.cn/practice/04093/



˼·��

������Ϊ�ý������������Ƚϼ򵥣����о���set��must_have��¼һ������ֵ��ĵ���ţ���һ�� 1��ֱ�Ӹ�ֵ ������ 1��ȡ���� ����must_nothave��¼һ�������ֵ��ĵ���ţ��Ѳ�Ҫ�Ĳ������������߼�ȥ���ô𰸣����

���룺

```python
N=int(input())
find=[]
for i in range(N):
    s=input().split()
    find.append(set(map(int,s[1:])))

M=int(input())
for _ in range(M):
    a=list(map(int,input().split()))
    must_have=set()
    must_not_have=set()
    first=True
    for i in range(N):
        if a[i]==1:
            if first:
                must_have=find[i].copy()
                first=False
            else:
                must_have&=find[i]
        elif a[i]==-1:
            must_not_have|=find[i]
    valid_docs=sorted(must_have - must_not_have)
    if valid_docs:
        print(' '.join(map(str,valid_docs)))
    else:
        print("NOT FOUND")
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-25%20224905.png?raw=true)




### Q6. Neural Networkʵ���β�������ݷ���

��http://clab.pku.edu.cn �ƶ����������Neural Networkʵ���β�������ݷ��ࡣ

�ο����ӣ�https://github.com/GMyhf/2025spring-cs201/blob/main/LLM/iris_neural_network.md





## 2. ѧϰ�ܽ�͸����ջ�

�����ҵ�ȽϿ���python�﷨���ƿأ���Ŀ���ö������Ǵ���ȽϷѾ������������˴�Ҷ��ᣬ����forѭ����������Ҫ�µ㹦�򣬾������Ҳͬ�����ⷽ�������������΢��ѧ����ţ�ٷ�������ͦ����˼

������ʵ���ǲ�֪��Ϊʲôȡ�������ţ����������Ƚ���������ǲ�ѯ��������ҪЩʱ�������Ŀ��ʵ��

�ܽ���������ҵ���⿼������﷨����䣬�ǺܺõĶ������Ͼ��������ǣ����ţ�Ӧ����������������ôд������������Ŀ�����򵥵������׳���ʵ�ڿ�ϧ

����Ŀǰ�Ը�����ÿ��ѡ���������ϸ�ѧ��״̬���˺ܶࣨ��������ѧ����С�ף�����Ҫ�¿��ˣ���������ˮƽ��

*������ѧ���е�æ����Ȼ���ô�ģ�ͺ���Ȥ�������޾���ѧϰ��ȱ�������⣨���ҿ������������/(��o��)/~~*

<mark>���������ҵ��Ŀ��Լ򵥣��з�Ѱ�Ҷ������ϰ��Ŀ���硰����2025springÿ��ѡ������LeetCode��Codeforces����ȵ���վ�ϵ���Ŀ��</mark>