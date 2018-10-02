import math
import operator

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    features = [['no surfacing',0],['flippers',1]] #0代表离散型
    return dataSet,features

def treeGrowth(dataSet,feature,feature_all):
    #判断是否是一类
    simClass=isSimClass(dataSet)
    if simClass :
        return simClass
    #判断特征列表是否为空,为空则选取D中最多的类作为子叶
    if len(features)==0:
        return moreClass(dataSet)
    #选择分裂属性和属性值
    Fl_feature,feature_index=selection_method(dataSet,feature,feature_all)
    #用分裂属性标记树节点
    treeNode={Fl_feature:{}}
    print('分裂属性：',Fl_feature)                       #调试代码
    print('分裂点：',feature_index)                      #调试代码
    #判断属性是否是离散值,离散值属性移除
    if liSan(Fl_feature,feature)==0:                     #判断是否是离散型属性
        feature_name_index = feature_all.index([Fl_feature, liSan(Fl_feature, feature)])  # 获取属性对应的列索引
        feature.remove([Fl_feature,liSan(Fl_feature,feature)])
        #筛选出每一个子数据组
        for i in feature_index:
            dataset1 = []
            for j in dataSet:
                if j[feature_name_index]==i:
                   dataset1.append(j)
            #判断子数据组是否为空，如果为空去D中多数元祖
            if len(dataset1)==0:
                moreClass(dataSet)
            else:
                treeNode[Fl_feature][i]=treeGrowth(dataset1,feature,feature_all)
    else:
        #如果分裂点为空则直接选取多的类做为标记
        if feature_index[0]==None:
            return moreClass(dataSet)
        else:
            feature_name_index = feature_all.index([Fl_feature, liSan(Fl_feature, feature)])  # 获取属性对应的列索引
        dataset1=[]
        dataset2=[]
        for i in dataSet:
            if i[feature_name_index]<feature_index[0]:
                dataset1.append(i)
            else:
                dataset2.append(i)
                # 判断子数据组是否为空，如果为空去D中多数元祖
        if len(dataset1) == 0:
            moreClass(dataSet)
        else:
            treeNode[Fl_feature]['<'+str(feature_index[0])] = treeGrowth(dataset1, feature,feature_all)
            # 判断子数据组是否为空，如果为空去D中多数元祖
        if len(dataset2) == 0:
            moreClass(dataSet)
        else:
            treeNode[Fl_feature]['>'+str(feature_index[0])] = treeGrowth(dataset2, feature,feature_all)
    return treeNode


#判断D中的元组都在同一类C中
def isSimClass(dataset):
    a=[i[-1] for i in dataset[:]]
    for i in a:
        if i!=a[0]:
            return None
    return a[0]

#计算D中的元组多数类
def moreClass(dataset):
    a=[i[-1] for i in dataset[:]]
    b={}
    for i in a:
        if i in b.keys():
            b[i]+=1
        else:
            b[i]=1
    sorted(b.items(), key=lambda  b:b[1],reverse=True)
    return list(b.keys())[0]

#选取最优分类属性和分类点
def selection_method(dataset,feature,feature_all):
    a={}                                               #存放每一属性的结果
    #遍历整个属性列表
    for i in feature :
        shuxing=[]                                     #属性包含的属性值
        feature_index=feature_all.index(i)
        b=[item[feature_index] for item in dataset[:]]
        #添加属性包含的属性值
        for j in b:
            if j not in shuxing:
                shuxing.append(j)
        shuxing.sort()
        print(i[0] ,'feature_index：',shuxing)                         #调试代码
        #如果属性是连续的
        if i[1]!=0:
            feature_shang={}                  #各属性值对应的信息增益
            l=len(shuxing)
            #分别取中间值做分界点，并选取最优分界点
            if l>1:                                   #判断属性的取值是否只有1个，如果一个则返回空的分裂点
                for k in range(l-1):
                    dataset1 = []        #小于分界点的数据组
                    dataset2 = []        #大于分界点的数据组
                    mid=(shuxing[k]+shuxing[k+1])/2
                    #分组
                    for m in dataset:
                        if m[feature_index]<mid:
                            dataset1.append(m)
                        else:
                            dataset2.append(m)
                    shang=len(dataset1)/len(dataset)*jsShang(dataset1)+len(dataset2)/len(dataset)*jsShang(dataset2)      #熵值计算
                    shang2=(-len(dataset1)/len(dataset))*math.log(len(dataset1)/len(dataset),2)+(-len(dataset2)/len(dataset))*math.log(len(dataset2)/len(dataset),2)           #c4.5与ID3的区别
                    feature_shang[mid]=shang/shang2
            else:
                shang = len(dataset) / len(dataset) * jsShang(dataset)  # 熵值计算
                shang2=(-len(dataset) / len(dataset))*math.log(len(dataset) / len(dataset),2)             #c4.5与ID3的区别
                feature_shang[None] = shang/shang2                                                       #空的分裂点
            feature_shang=sorted(feature_shang.items(), key=lambda b:b[1])                                     #熵值从小到大排列
            print(i[0],'feature_shang:',feature_shang)                                        #调试代码
            a[i[0]]=[[feature_shang[0][0]],feature_shang[0][1]]                                         #选取小的熵值——>可以转化为信息增益最大
        else:
            feature_shang=0
            feature_shang2=0
            for k in shuxing:
                dataset1=[]
                for m in dataset:
                    if m[feature_index]==k:
                        dataset1.append(m)
                feature_shang+=len(dataset1)/len(dataset)*jsShang(dataset1)
                feature_shang2+=(-len(dataset1)/len(dataset))*math.log(len(dataset1)/len(dataset),2)                 #c4.5与ID3的区别
            a[i[0]]=[shuxing,feature_shang/feature_shang2]
            print(i[0], 'feature_shang:', feature_shang)                          #调试代码
    feature_return=None
    feature_min_shang=math.inf
    for i in a:
        if a[i][1]<feature_min_shang:
            feature_return=i
            feature_min_shang=a[i][1]
    return feature_return,a[feature_return][0]                                                 #返回最优分裂属性和分裂点

#计算熵值
def jsShang(dataset):
    a={}
    sum=0
    shang=0
    b=[i[-1] for i in dataset[:]]
    for i in b:
        if i in a.keys():
            a[i]+=1
        else:
            a[i]=1
        sum+=1
    for i in a:
        shang+=((-a[i]/sum)*math.log(a[i]/sum,2))
    return shang

#判断属性是否是离散值
def liSan(feature,feature_list):
    for i in feature_list:
        if feature==i[0]:
            return i[1]


if __name__=='__main__':
    for i in range (100):
        dataSet,features=createDataSet()
        features_all=features.copy()
        print('第',i,'次：',treeGrowth(dataSet,features,features_all))