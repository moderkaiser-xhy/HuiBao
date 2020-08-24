import numpy as np
import os
import pandas as pd
import shutil
import time
from decimal import *

from MySQL import extract_data


#随机森林结果分析文件
def RandomForestFileResult(X, Y, Accuracy, n_estimators, max_depth, min_sample_leaf, max_leaf_nodes, max_features, min_samples_split, UserID):
    X_columns = ['输入特征', '标签', '分类测试准确率', '树的数量', '最大深度', '最小样本叶片', '最大子叶节点', '最大特征数', '节点再划分所需最小样本数']
    List = [X, Y, Accuracy, n_estimators, max_depth, min_sample_leaf, max_leaf_nodes, max_features, min_samples_split]
    res = pd.DataFrame()
    for i in List:
        df = pd.DataFrame(i)
        res = pd.concat([res, df], axis=1, ignore_index=True, join='outer')
    res.columns = X_columns
    path = os.path.abspath('../UserFiles/'+str(UserID)+'/Result/RandomForest.csv')
    res.to_csv(path)
    return

#朴素贝叶斯结果分析文件
def NaiveBayesFileResult(X, Y, Accuracy, UserID):
    X_columns = ['输入特征', '标签', '分类测试准确率']
    List = [X, Y, Accuracy]
    res = pd.DataFrame()
    for i in List:
        df = pd.DataFrame(i)
        res = pd.concat([res, df], axis=1, ignore_index=True, join='outer')
    res.columns = X_columns
    path =  os.path.abspath('../UserFiles/'+ str(UserID) + '/Result/NaiveBayes.csv')
    res.to_csv(path)
    return

#决策树结果分析文件
def DecisionTreeFileResult(X, Y, Accuracy, max_depth, min_sample_leaf, max_leaf_nodes, max_features, min_samples_split, UserID):
    X_columns = ['输入特征', '标签', '分类测试准确率', '最大深度', '最小样本叶片', '最大子叶节点', '最大特征数', '节点再划分所需最小样本数']
    List = [X, Y, Accuracy, max_depth, min_sample_leaf, max_leaf_nodes, max_features, min_samples_split]
    res = pd.DataFrame()
    for i in List:
        df = pd.DataFrame(i)
        res = pd.concat([res, df], axis=1, ignore_index=True, join='outer')
    res.columns = X_columns
    path = os.path.abspath('../UserFiles/'+ str(UserID) +'/Result/DecisionTree.csv')
    res.to_csv(path)
    return

#生成预测结果文件
def PredictFile(Model=None, ID=None, Result=None, UserID=None): #X为输入特征列表
    #####
    T_name = UserID + "predict1"
    data = extract_data(T_name, ID)
    name = np.array(data[0])
    Value = np.array(data[1])
    Value = Value.transpose()
    x = pd.DataFrame(Value)
    x.columns = name
    #####
    a = len(x.columns)
    x.insert(a, '预测结果', Result)
    path = os.path.abspath('../UserFiles/' + str(UserID) + '/Result/Predict' + str(Model) + '.csv')
    x.to_csv(path, index=False)
    return

#文件目录创建
def CreatDirectory(UserID):
    List = ['/Predict/1', '/Train/1', '/Result']
    for i in List:
        path = os.path.abspath('../UserFiles/' + str(UserID)  + i)
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
    return

#删除文件目录
def DeleteFile(UserID):
    path = os.path.abspath('../UserFiles/' + str(UserID))
    isExists = os.path.exists(path)
    if isExists:
        shutil.rmtree(path)
    return

#上下拼接
def FileUp_Down(UserID, Style, file):
    path = os.path.abspath('../UserFiles/' + str(UserID) + '/' + str(Style)) + '/'
    #files = os.listdir(path)
    #files_csv = list(filter(lambda x: x[-4:] == '.csv', files))  #获取所有csv文件
    files_csv = file
    a = files_csv.pop()
    try:
        res = pd.read_csv(path + a, encoding='utf8')
    except:
        res = pd.read_csv(path + a, encoding='gbk')
    for i in files_csv:
        try:
            df = pd.read_csv(path + i, encoding='utf8')
        except:
            df = pd.read_csv(path + i, encoding='gbk')
        res = pd.concat([res, df], axis=0, ignore_index=True, join='inner')
    path_save = os.path.abspath('../UserFiles/' + str(UserID) + '/'+ str(Style) +'/1/file.csv')
    res.to_csv(path_save, index=False)
    return

#左右拼接
def FileLeft_Right(columns, UserID, Style): #columns = [[文件名1, 关联列名1], [文件名2, 关联列名2], ...]
    path = os.path.abspath('../UserFiles/' + str(UserID) + '/' + str(Style)) + '/'
    for i in columns:
        file_name = i[0]
        print(file_name)
        Col_name = i[1]
        try:
            df = pd.read_csv(path + file_name, encoding='utf8')
        except:
            df = pd.read_csv(path + file_name, encoding='gbk')
        length = df.shape[0] - 2
        b = len(df.groupby([Col_name]).size())
        if b < length:
            b = 0
            break
    if b != 0:
        a = columns.pop()
        try:
            res = pd.read_csv(path + a[0], encoding='utf8')
        except:
            res = pd.read_csv(path + a[0], encoding='gbk')
        for i in columns:
            file_name = i[0]
            Col_name = i[1]
            try:
                df = pd.read_csv(path + file_name, encoding='utf8')
            except:
                df = pd.read_csv(path + file_name, encoding='gbk')
            res = pd.merge(res, df, how='inner', left_on=a[1], right_on=Col_name)
        path_save = os.path.abspath('../UserFiles/' + str(UserID) + '/'+ str(Style) +'/1/file.csv')
        res.to_csv(path_save, index=False)
        return 1
    return 0

#显示文件列名函数
def ShowName(UserID, Style):
    path = os.path.abspath('../UserFiles/' + str(UserID) + '/' + str(Style)) + '/'
    files = os.listdir(path)
    files_csv = list(filter(lambda x: x[-4:] == '.csv', files))  #获取所有csv文件
    print(files_csv)
    Dict_name = {}
    for i in files_csv:
        try:
            df = pd.read_csv(path + i, encoding='gbk')
        except:
            df = pd.read_csv(path + i, encoding='utf8')
        name = df.columns.tolist()
        Dict_name[i] = name
    return Dict_name

#用户已上传的文件
def userFiles(UserID, Type):
    trainPath = os.path.abspath('../UserFiles/' + str(UserID) + '/Train') + '/'
    trainFiles = os.listdir(trainPath)
    trainFiles.remove("1")
    predictPath = os.path.abspath('../UserFiles/' + str(UserID) + '/Predict') + '/'
    predictFiles = os.listdir(predictPath)
    predictFiles.remove("1")
    fileList = []
    if Type == "Train":
        for i in trainFiles:        #获取训练文件
            filePath = trainPath + i
            fileSize = str(Decimal(str(os.path.getsize(filePath)/1000)).quantize(Decimal('0.0'))) + 'KB'

            DICT = {"filename": i, "filepath": filePath, "filesize": fileSize, "filetype": "Train"}
            fileList.append(DICT)

        #fileDict["Type"] = "Train"
        return fileList
    elif Type == "Predict":
        for j in predictFiles:
            filePath = predictPath + j
            fileSize = str(Decimal(str(os.path.getsize(filePath)/1000)).quantize(Decimal('0.0'))) + 'KB'
            DICT = {"filename": j, "filepath": filePath, "filesize": fileSize, "filetype": "Predict"}
            fileList.append(DICT)
        #fileDict["Type"] = "Predict"
        return fileList
    else:
        return "error"

#删除文件
def det_files(UserID, Type, File_name):
    filePath =  os.path.abspath('../UserFiles/' + str(UserID)) + '/' + Type + '/' + File_name
    print(os.path.exists(filePath))
    if os.path.exists(filePath):
        try:
            os.remove(filePath)
        except:
            shutil.rmtree(filePath)
        return "success"
    else:
        return "文件不存在"

def fileName(filename, type, UserID):
    time = Localtime()
    path = os.path.abspath('../UserFiles/' + UserID+ '/' + type)
    print(path)
    Files = os.listdir(path)
    for i in Files:
        if filename == i:
            filename = time + '_' + filename
            break
    print(filename)
    return filename


#服务器时间获取
def Localtime():
    localtime = time.localtime
    localtime = time.localtime(time.time())
    a = str(localtime.tm_year) + '-' + str(localtime.tm_mon) + "-" + str(localtime.tm_mday) + '-' + str(
        localtime.tm_hour) + ':' + str(localtime.tm_min) + ':' + str(localtime.tm_sec)
    return a

if __name__ == "__main__":
    '''
    X = [1,1,1,1]; Y = [2]; A = [3]; n_estimators = [4]
    max_depth = [5]; min_sample_leaf = [6]; max_leaf_nodes = [7]
    max_features = [8]; min_samples_split = [9]
    UserID = '123456'
    RandomForestFileResult( X , Y , A , n_estimators, max_depth, min_sample_leaf, max_leaf_nodes, max_features, min_samples_split,UserID)
    '''
    '''

    file = "package.xml"
    UserID = '123456'
    T = "Train"
    fileName(file, UserID, T)
    a = userFiles('123456', "Predict")
    print(a)
    print( det_files("123456", 'Predict', "1245.csa"))

    UserID = '123456'
    type = 'Train'
    filename = "123"
    fileName(filename, type, UserID)
    #ShowName(UserID, Style)
    a = userFiles('123456', "Predict")
    print(a)
    '''
    UserID = "123456"
    Type = "Predict"
    File_name = "1"
    print( det_files(UserID, Type, File_name))


