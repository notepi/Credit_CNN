 # -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 11:04:55 2018

@author: Administrator
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import math
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.model_selection import GridSearchCV    # 0.17 grid_search
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
# 模型评价 混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report,precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from time import time
from scipy.stats import randint as sp_randint

def ageclass(agetemp):
    if agetemp < 16:
        cc=0
        pass
    elif agetemp >65:
        cc=11
        pass
    else:
        cc=int((agetemp-15)/5)+1
        pass
    return cc  
    pass
def incomeclass(temp):
    #    bottom=[1000,1000,1000,10000,10000,10000,10000]
    bottom=[10,10,10,10,10,10,10]
#    income=Data[u'收入']
    incometemp=np.zeros((1, 8))[0]
    #0计算低于3500部分
    if temp > 3500:
        incometemp[0]= math.log(3500,bottom[1])#计算以c为底，b的对数:
        pass
    else:
        incometemp[0] = math.log(temp+1,bottom[1])
        return incometemp
        pass
    
    #1计算3500-5000部分
    temp=temp-3500
    
    if temp > 1500:
        incometemp[1]= math.log(1500,bottom[1])#计算以c为底，b的对数:
        pass
    else:
        incometemp[1] = math.log(temp+1,bottom[1])
        return incometemp
        pass

    #2计算5000-8000部分  
    temp=temp-1500
    if temp > 3000:
        incometemp[1]= math.log(3000,bottom[1])#计算以c为底，b的对数:
        pass
    else:
        incometemp[1] = math.log(temp+1,bottom[1])
        return incometemp
        pass

    #3计算8000-12500部分
    temp=temp-3000
    if temp > 4500:
        incometemp[2]= math.log(4500,bottom[2]) #计算以c为底，b的对数:
        pass
    else:
        incometemp[2] = math.log(temp+1,bottom[2])
        return incometemp
        pass
    
    #4计算12000-38000部分
    temp=temp-4500
    if temp > 26000:
        incometemp[3]= math.log(26000,bottom[3]) #计算以c为底，b的对数:
        pass
    else:
        incometemp[3] =  math.log(temp+1,bottom[3])
        return incometemp
        pass
    
    #5计算38000-5800部分
    temp=temp-26000
    if temp > 20000:
        incometemp[4]= math.log(20000,bottom[4]) #计算以c为底，b的对数:
        pass
    else:
        incometemp[4] =  math.log(temp+1,bottom[4])
        return incometemp
        pass

    #6计算58000-83000部分
    temp=temp-20000

    if temp > 25000:
        incometemp[5]= math.log(25000,bottom[5]) #计算以c为底，b的对数:
        pass
    else:
        incometemp[5] =  math.log(temp+1,bottom[5])
        return incometemp
        pass
 
    
    #7计算>83000部分
    temp=temp-25000

    incometemp[6] =  math.log(temp+1,bottom[6])
    return  incometemp

    pass
if __name__ == "__main__":

    Data = pd.read_excel('../data/data2.xlsx',encoding = "GBK")
    
######################################################################    
    #年龄
    """
    年龄这个选项，做离散化处理：
    15岁以下一个分组，16-20
    65岁以上一个分组, 66以上
    65岁以上一个分组
    """
    age=Data[u'年龄']
    AgeClass=list(map(ageclass,age))
    #进行OneHot编码
    #共计分成12个组
    AgeOneHot=np.zeros((len(Data), 12))
    for i,j in enumerate(AgeClass):
        AgeOneHot[i][j]=1.0
        pass
    agename=[u"15岁以下",u"16-20岁",u"21-25岁",
             u"26-30岁",u"31-35岁",u"36-40岁",
             u"41-45岁",u"46-50岁",u"51-55岁",
             u"56-60岁",u"61-65岁",u"66岁以上"
             ]
    AgeOneHot=pd.DataFrame(AgeOneHot,columns=agename)
######################################################################        
    #收入
    """
    收入按照个人所得税的梯度划分
    <3500           0-3500
    3500-5000       0-1500
    5000-8000       0-3000
    8000-12500      0-4500
    12500-38500     0-26000
    38500-58500     0-20000
    58500-83500     0-25000
    >83000
    统计每部分的金额,然后以10为底，取对数
    """
    bottom=[10,10,10,10,10,10,10]
    income=Data[u'收入']
    IncomeOneHot=np.zeros((len(Data), 8))
    
    for i,j in enumerate(income):
        IncomeOneHot[i]=incomeclass(j)
#        break
        pass
    agename=[u"3500以下",u"3500-5000",u"5000-8000",u"8000-1250",
             u"12500-38500",u"38500-58500",u"58500-83500",u"大于83000"]
    IncomeOneHot=pd.DataFrame(IncomeOneHot,columns=agename)
######################################################################    
#婚姻
    #未婚是0，已婚是1,离婚2
    Marriage=Data[u'婚姻状况']
    state={u"未婚":0,u"已婚":1,u"已婚 ":1,u"离婚":2,u"离异":2}
    Marriage=pd.DataFrame([state[i] for i in Marriage],columns=[u'婚姻状况'])
    
######################################################################    
#受教育程度
    #未婚是0，已婚是1,离婚2
    Education=Data[u'教育程度']
    state={u"小学":0,u"初中":1,u"初中 ":1,u"高中":2,u"中专":2,u"大专":3,
          u"大专 ":3, u"专科":3,u"本科":4,u"大学":4,u"硕士":5,u"博士":6}
    Education=pd.DataFrame([state[i] for i in Education],columns=[u'教育程度'])
######################################################################    
#存款
    #存款取以2为底的对数
    Deposit=Data[u'存款']
    temp=[]
    for i in Deposit:
        if u"万" in str(i):
            i=int(i.split(u"万")[0]+"0000")
            temp.append(i)
            pass
        elif u"无" in str(i) or 0== i:
            temp.append(0)
            pass
        elif i > 0:
            temp.append(i)
            pass
        else:
            print("error")
            break
            pass
#        break
        pass
    DepositCode=[math.log(i+1,2) for i in temp]
    DepositCode=pd.DataFrame(DepositCode,columns=[u'存款'])
######################################################################    
#房产    
    RealEstate=Data[u'房产']
    RealEstateCode=np.zeros((len(Data), 2))
    for i,j in enumerate(RealEstate):
        if j == u"有" or j == u"有 ":
            RealEstateCode[i][0]=1
            pass
        elif j == u"自建":
            RealEstateCode[i][1]=1
            pass
        elif j == u"无" or j == u"无 ":
            pass
        else:
            print("error")
            break
            pass
        pass
    RealEstateCode=pd.DataFrame(RealEstateCode,columns=[u'商品房', u'自建房'])
######################################################################    
#车辆信息    
    Car=Data[u'车辆信息']
    state={u"有":0,u"有 ":0,u"无":1,u"无 ":1,u"无、":1,u" 无":1}
    CarCode=pd.DataFrame([state[i] for i in Car],columns=[u'车辆信息'])
######################################################################    
#网购记录    
    OnlineShopping=Data[u'网购记录']
    OnlineShoppingCode=[math.log(i+1,2) for i in OnlineShopping]
    OnlineShoppingCode=pd.DataFrame(OnlineShoppingCode,columns=[u'网购记录'])
#######################################################################    
#贷款记录    
    Loan=Data[u'贷款记录']
    tag=0
    #房贷车贷 其他贷款
    LoanCode=np.zeros((len(Data), 3))
    for i,j in enumerate(Loan):
        tag=0
        if u"房贷" in str(j):
            LoanCode[i][0]=1
            tag=1
            pass
        if u"车贷" in str(j):
            LoanCode[i][1]=1
            tag=1
            pass
        if u"无" in str(j):
            tag=1
            pass
        else:
            tag=1
            LoanCode[i][2]=1
            pass
        pass
    LoanCode=pd.DataFrame(LoanCode,columns=[u'房贷',u'车贷',u'其他'])
######################################################################    
#违约记录
    BreachContract=Data[u'违约记录']
    temp=[]
    for i,j in enumerate(BreachContract):
        if j == "无" or j == "无 ":
            temp.append(0)
            pass
        else:
            temp.append(1)
        pass
    BreachContractCode=pd.DataFrame(temp,columns=[u'违约记录'])
    
######################################################################    
#公积金
    #有1，无0
    AccumulationFund=Data[u'公积金']
    AccumulationFund=AccumulationFund.fillna(u"有")
    state={u"有":1,u"无":0}
    AccumulationFundCode=pd.DataFrame([state[i] for i in AccumulationFund]
                                        ,columns=[u'公积金'])

######################################################################    
#支付宝年纪
    #有1，无0
    AlipayAge=Data[u'支付宝年龄']
    AlipayAgeCode=AlipayAge
    
######################################################################    
#职业
    #一般 是 教师，医生，金融从业，公务员，职业  个体
    #无，
    #个体户，包括,个体劳务
    #教师
    #公务员
    #职员（初级职员、中级职员、高级职员）
    #医护人员（医生、护士）
    Occupation=Data.iloc[:,-1]
    aa=Occupation.value_counts()
    temp=[]
    for i in Occupation:
        if u"无" in i:
            temp.append(i)
            pass
        elif u"个体户" in i or u"个体劳务" in i or u"小企业主"in i :
            temp.append(u"个体")
            pass
        elif u"教师" in i:
            temp.append(u"教师")
            pass
        elif u"医生" in i or u"护士" in i:
            temp.append(u"医护人员")
            pass
        elif u"职员" in i:
            temp.append(i)
            pass
        elif u"公务员" in i:
            temp.append(i)
            pass
        elif u"金融从业者" in i:
            temp.append(i)
            pass
        else:
            print(i)
#        break
        pass
    
    #无职业，个体，教师，医护，职员，公务员，金融从业者
    OccupationCode=np.zeros((len(Data), 7))
    #进行编码
#    i,j in enumerate(RealEstate):
    for i,j in enumerate(temp):
        if u"无" in j:
            OccupationCode[i][0]=1
            pass
        elif u"个体" in j:
            OccupationCode[i][1]=1
            pass
        elif u"教师" in j:
            OccupationCode[i][2]=1
            pass
        elif u"医护" in j:
            OccupationCode[i][3]=1
            pass
        elif u"职员" in j:
            if u"初级" in j:
                OccupationCode[i][4]=1
            elif u"中级" in j:
                OccupationCode[i][4]=2
            elif u"高级" in j:
                OccupationCode[i][4]=3
                pass
            pass
        elif u"公务员" in j:
            OccupationCode[i][5]=1
            pass
        elif u"金融从业者" in j:
            OccupationCode[i][6]=1
            pass
        else:
            print(j)        
        pass
    OccupationCode=pd.DataFrame(OccupationCode,columns=[u'无职业',u'个体',u'教师',u"医护人员",
                                                        u"职员",u"公务员",u"金融从业者"])
######################################################################    
#网购
    OnlineShopping=Data[u"网购记录"]
    #数据标准化
    scaler = MinMaxScaler()
    scaler.fit(OnlineShopping.values.reshape(-1, 1))
    OnlineShopping = scaler.transform(OnlineShopping.values.reshape(-1, 1))
    OnlineShopping=pd.DataFrame(OnlineShopping,columns=[u"网购记录"])
######################################################################    
#违法 
    Illegality=Data[u"违法记录"]
    temp=[]
    for i in Illegality:
        if  "无" in str(i):
            temp.append(0)
        else:
            temp.append(1)
        pass
    Illegality=pd.DataFrame(temp,columns=[u"违法记录"])
######################################################################    
#芝麻信用
    Sesamecredit=Data[u"芝麻分"]
    temp=[]
    for i in Sesamecredit:
        if i > 680:
            temp.append(0)
        else:
            temp.append(1)
            pass
        pass
    Sesamecredit=pd.DataFrame(temp,columns=[u"芝麻信用"])
    #芝麻信用,芝麻分,违约，年龄，收入，受教育，房地产，公积金，支付宝年纪，职业,贷款(车贷，房贷，信用贷)，婚姻，网购记录，违法记录
    result=pd.concat([Sesamecredit,Data[u"芝麻分"],BreachContractCode,AgeOneHot,IncomeOneHot,Education,RealEstateCode,AccumulationFundCode,AlipayAgeCode,OccupationCode,LoanCode,Marriage,OnlineShopping,Illegality],axis=1)
    
    result.to_csv("../data/result.csv",encoding='GBK',index=False)
    #增加样本数量
    result=pd.concat([result,result[result[u"违约记录"]==1]])
    
    
#    data=result.iloc[:,2:]
    
    #生成训练数据和测试数据
    testlDataTest, TrainlDataTrain = train_test_split(result, train_size=0.3, random_state=1)
    
       
    XdataTrain = TrainlDataTrain.iloc[:,3:]
    TagTrain = TrainlDataTrain.iloc[:,2]
    
    XdataTest = testlDataTest.iloc[:,3:]
    TagTest = testlDataTest.iloc[:,2]
    
     #KNN
    model = KNeighborsClassifier(n_neighbors=3)
    n_neighbors = list(range(1,21,2))
    neigh = GridSearchCV(model, param_grid={'n_neighbors': n_neighbors}, cv=5)
    
    neigh.fit(XdataTrain, TagTrain) 
    #预测结果        
    Result = neigh.predict(XdataTest)
    #预测结果        
    print("============================KNN=========================================")
    Result = neigh.predict(XdataTest)
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #混淆矩阵
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))

    #svm
    
    #训练模型
    model = svm.SVC(kernel='rbf')
    c_can = np.logspace(-2, 2, 10)
    gamma_can = np.logspace(-2, 2, 10)
    svc = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
    svc.fit(XdataTrain, TagTrain)
    

    #预测结果        
    print("============================svm=========================================")
    Result = svc.predict(XdataTest)
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
    
    
    #MNB
    #训练模型    
    MNB=MultinomialNB()
    MNB.fit(XdataTrain, TagTrain.values.ravel())
    #预测结果        
    Result = MNB.predict(XdataTest)
    #预测结果        
    print("============================MNB=========================================")
    Result = MNB.predict(XdataTest)
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
    
    #Dtree
    Dtree = tree.DecisionTreeClassifier()
    Dtree.fit(XdataTrain, TagTrain)
    
    #预测结果        
    print("============================Dtree=========================================")
    Result = Dtree.predict(XdataTest)
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
    
    #REF
    ref = RandomForestClassifier()
    ref.fit(XdataTrain, TagTrain)
    
    #预测结果        
    print("============================REF=========================================")
    Result = ref.predict(XdataTest)
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))   
    
    # 使用随机森林作为分类器，分类器有20课树
    clf = RandomForestClassifier(n_estimators=20)
    
    
    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
    
    
    # 设置可能学习的参数
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    
    # 随机搜索， randomized search
    n_iter_search = 35
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)
    #起始时间
    start = time()
    random_search.fit(XdataTrain, TagTrain)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)
    
    print("=============================================")

    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    
    # 网格搜索， grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(XdataTrain, TagTrain)
    
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)
    
    #对比分类效果
    #random_search
    #预测结果        
    print("============================random_search=========================================")
    Result = random_search.predict(XdataTest)
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #混淆矩阵
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
   
    #预测结果        
    print("============================grid_search=========================================")
    Result = grid_search.predict(XdataTest)
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #混淆矩阵
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
    
    
    print("============================+++++=========================================")
    Result = result[u'芝麻信用']
    TagTest = result[u'违约记录']
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #混淆矩阵
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
    
    
    
        
        
        
        
        
    
        
        
        
        
        