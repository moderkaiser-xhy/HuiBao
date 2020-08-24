import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import os, shutil
from MySQL import extract_data
import numpy as np
import pandas


def transpose(matrix):
    return zip(*matrix)

#数据库提取数据
def MysqlDataExtract(DataName=None, table_name=None):
    data = extract_data(table_name, DataName)
    name = np.array(data[0])
    DataValue = data[1]
    DataValue = transpose(DataValue)
    x = pandas.DataFrame(DataValue)
    x.columns = name
    return x

#判断界面是否有输入
def TEST(X=[], Y=[]):    #没有输入时，传入后端为['', '',...]
    X.append(Y[0])
    for i in X:
        if i == '':
            a = 0
            break
        else:
            a = 1
    X.pop()
    return a

#获得训练集与测试集数据
def CH_RetrieveData(Data_X=[], Data_Y=[], UserID=None):
   T_name = str(UserID) + 'train1'
   a = TEST(X=Data_X, Y=Data_Y)
   if a == 1:
       x = MysqlDataExtract(DataName=Data_X, table_name=T_name)
       y = MysqlDataExtract(DataName=Data_Y[:1], table_name=T_name)     #从数据库提取数据
       x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=33)
       vec = DictVectorizer()
       x_train = vec.fit_transform(x_train.to_dict(orient="record"))
       x_test = vec.transform(x_test.to_dict(orient="record"))
       return [x_train, x_test, y_train, y_test]
   else:
       return []

#获取预测数据  ####修改#######################################################
def CH_RetrievePrediction(DataList=[], UserID=None):
    T_name = str(UserID) + 'prediction1'
    DataPrediction = MysqlDataExtract(DataName=DataList, table_name=T_name)
    return DataPrediction

#生成随机森林结果文件
def RandomForestFileResult(DataX=[], DataY=[], TrainAccuracy=[], TestAccuracy=[], n_estimators=[], max_depth=[], UserID=None):
    c = DataX.pop()
    DataY = DataY[:1]
    TrainAccuracy = TrainAccuracy[:1]
    TestAccuracy = TestAccuracy[:1]
    n_estimators = n_estimators[:1]
    max_depth = max_depth[:1]
    for i in DataX:
        DataY.append('')
        TestAccuracy.append('')
        TrainAccuracy.append('')
        n_estimators.append('')
        max_depth.append('')
    DataX.append(c)
    dict1 = {'训练集准确率': TrainAccuracy, '测试集准确率': TestAccuracy, 'n_estimators': n_estimators, 'max_depth': max_depth, \
             '输入特征': DataX, '标签':DataY}
    DataFrameResult = pd.DataFrame(dict1)
    path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/UserFiles/'+UserID+'/Result/RandomForest.csv'
    DataFrameResult.to_csv(path)
    return

#随机森林分类
def CH_RandomForestClassifier(n_estimators=100, max_depth=11, max_features=1, DataX=[], DataY=[], UserID=None, prediction=None):
    data = CH_RetrieveData(Data_X=DataX, Data_Y=DataY, UserID=UserID)
    if data:
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rfc.fit(data[0], data[2])
        P_train = rfc.score(data[0], data[2])
        P_test = rfc.score(data[1], data[3])
        RandomForestFileResult(DataX=DataX, DataY=DataY, max_depth=[max_depth], n_estimators=[n_estimators], TrainAccuracy=[P_train], \
                   TestAccuracy=[P_test], UserID=UserID)    #生成数据分析文件
        PredictFile = pd.DataFrame()
        path = '../UserFiles/' + str(UserID) + '/Result/' + 'PredictRandomForest.csv'
        PredictFile.to_csv(path)
        return [P_train, P_test]
    else:
        error = '请选择训练数据'
        return error

#生成决策树/朴素贝叶斯分类结果文件
def DecisionTreeFileResult(DataX=[], DataY=[], TrainAccuracy=[], TestAccuracy=[], UserID=None, Model=None):
    c = DataX.pop()
    DataY = DataY[:1]
    TrainAccuracy = TrainAccuracy[:1]
    TestAccuracy = TestAccuracy[:1]
    for i in DataX:
        DataY.append('')
        TestAccuracy.append('')
        TrainAccuracy.append('')
    DataX.append(c)
    dict1 = {'训练集准确率': TrainAccuracy, '测试集准确率': TestAccuracy, '输入特征': DataX, '标签': DataY}
    DataFrameResult = pd.DataFrame(dict1)
    DataFrameResult.to_csv(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/UserFiles/'+ str(UserID) + '/Result/'+ Model +'.csv')
    return

#决策树分类
def CH_DecisionTree(DataX=[], DataY=[], UserID=None, prediction=None):
    data = CH_RetrieveData(Data_X=DataX, Data_Y=DataY, UserID=UserID)
    if data:
        dtc = DecisionTreeClassifier()
        dtc.fit(data[0], data[2])
        P_train = dtc.score(data[0], data[2])
        P_test = dtc.score(data[1], data[3])

        DecisionTreeFileResult(DataX=DataX, DataY=DataY, TestAccuracy=[P_test], TrainAccuracy=[P_train], UserID=UserID, Model='DecisionTree')    #生成结果分析文件
        PredictFile = pd.DataFrame()
        path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/UserFiles/' + str(UserID) + '/Result/' + 'PredictDecisionTree.csv'
        PredictFile.to_csv(path)
        return [P_train, P_test]
    else:
        error = '请选择训练数据'
        return error

#朴素贝叶斯分类
def CH_NaiveBayes(DataX=[], DataY=[], UserID=None, prediction=None):
    data = CH_RetrieveData(Data_X=DataX, Data_Y=DataY, UserID=UserID)
    if data:
        mnb = MultinomialNB()
        mnb.fit(data[0], data[2])
        P_train = mnb.score(data[0], data[2])
        P_test =  mnb.score(data[1], data[3])
        #---------------------------------------------------------------------
        DecisionTreeFileResult(DataX=DataX, DataY=DataY, TestAccuracy=[P_test], TrainAccuracy=[P_train], UserID=UserID, Model='NaiveBayes')
        PredictFile = pd.DataFrame()
        path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/UserFiles/' + str(UserID) + '/Result/' + 'PredictNaiveBayes.csv'
        PredictFile.to_csv(path)
        return [P_train, P_test]
    else:
        error = '请选择训练数据'
        return error

def CreatDirectory(UserID):
    List = ['Predict', 'Train', 'Result']
    for i in List:
        path = '../UserFiles/' + str(UserID) + '/' + i
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
    return
def DeleteFile(UserID):
    shutil.rmtree(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/UserFiles/' + str(UserID))
    return

if __name__ == '__main__':
    result = CH_RandomForestClassifier(DataX=['GENDER', 'AGE'], DataY=['ACTIVE_DAY_09'], UserID='123')
    print(result)
    result1 = CH_DecisionTree(DataX=['GENDER', 'AGE'], DataY=['ACTIVE_DAY_09'], UserID='123')
    print(result1)
    result2 = CH_NaiveBayes(DataX=['GENDER', 'AGE'], DataY=['ACTIVE_DAY_09'], UserID='123')
    print(result2)

