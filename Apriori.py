import itertools

def creatDataset():
    dataset = [[1, 2, 5], [2, 4], [2, 3], [1, 2, 4], [1, 3], [2, 3], [1, 3], [1, 2, 3, 5], [1, 2, 3]]
    return  dataset

def Apriori(min_sup=2):
    dataset=creatDataset()
    C1=getC1(dataset)
    C1.sort()
    L,num=jsC(C1,min_sup,dataset)
    level=1
    while 1:
        C=newC(L,level)
        L,num=jsC(C,min_sup,dataset)
        if len(L)==0:
            break
        level+=1
        print(level,':')
        print(L)
        print(num)


#频繁1项集
def getC1(dataset):
    C=[]
    for i in dataset:
        for j in i:
            if [j] not in C:
                C.append([j])
    return C

#计算项集频率
def jsC(L,ps,dataset):
    C={}              #所有项的频率
    CC=[]             #满足最小支持度阈值的项
    num={}            #满足最小支持度阈值的项与相应的频率
    for i in L:
        for j in dataset:
            if set(i).issubset(j):
                if str(i) in C.keys():
                    C[str(i)]+=1
                else: C[str(i)]=1
    #筛选超过最小阈值频率项集
    for i in C:
        if C[i]>=ps:
            CC.append(eval(i))
            num[i]=C[i]
    return CC,num

#连接步，剪枝步
def newC(L,level):
    LL=[]
    L.sort()
    #连接步     判断前l-1个项是否相同，相同保留前l-1增加最后两位
    for i in range(len(L)):
        for j in range(i+1,len(L)):
            l1=list(L[i])
            l2=list(L[j])
            if l1[:-1]==l2[:-1]:
                l1.append(l2[-1])
                LL.append(l1)

    #剪枝步      判断子集是否包含在k-1项集内
    for l in LL:
        a=list(itertools.combinations(l,level))
        for i in a:
            if list(i) not in L:
                LL.remove(l)
                break
    return LL

if __name__ == '__main__':
    Apriori(min_sup=3)