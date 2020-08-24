from MySQL import extract_data, excel_create_table
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

#数据库提取数据
def CH_DataFrame(T_name, X):
    data = extract_data(T_name, X)
    name = np.array(data[0])
    print(name)
    Value = data[1]
    Value = zip(*Value)
    x = pd.DataFrame(Value)
    x.columns = name
    x=x.replace("",np.NAN)
    x = x.fillna(method='pad')
    #print (x)
    return x

#获得训练集与测试集数据
def CH_RetrieveData(Data_X, Data_Y, UserID):
    flag = 0
    for i in Data_X:
        if i is not None:
            flag = 1
            break
    if flag == 1 and Data_Y[0] != None:
        T_name = str(UserID) + 'train1'
        x = CH_DataFrame(T_name, Data_X)
        print (type(x))
        aaa = x.values.T.tolist()
        b=aaa[0]
        y = CH_DataFrame(T_name, Data_Y[:1])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=33)
        vec = DictVectorizer()
        x = vec.fit_transform(x.to_dict(orient="record"))
        x_test = vec.transform(x_test.to_dict(orient="record"))
        return [x, x_test, y, y_test]
    return 0

#获得预测数据
def CH_RetrievePredict(DataList, UserID):
    flag = 0
    for i in DataList:
        if i is not None:
            flag = 1
            break
    if flag == 1:
        T_name = str(UserID) + 'predict1'
        x = CH_DataFrame(T_name, DataList)
        aaa = x.values.T.tolist()
        b=aaa[0]
        #print (aaa)
        print (type(b[5]))
        vec = DictVectorizer()
        x = vec.fit_transform(x.to_dict(orient="record"))
        print (vec.feature_names_)
        print("---------------------------------------------------------------")
        print(x)
        return x
    return 0

#数据库建两张表
def DatabaseCreatTrain(UserID):
    T_name1 = str(UserID) + 'train'
    path_train = os.path.abspath('../UserFiles/' + str(UserID) + '/Train/1/file.csv')
    aa=excel_create_table(table_name=T_name1, csv_path=path_train)
    return aa
def DatabaseCreatPredict(UserID):
    T_name2 = str(UserID) + 'predict'
    path_predict = os.path.abspath('../UserFiles/' + str(UserID) + '/Predict/1/file.csv')
    aa=excel_create_table(table_name=T_name2, csv_path=path_predict)
    return aa

#数据库获取表名
def show_name(user_id):
    train_name = extract_data(user_id+'train1', ['*'])
    train_name = np.array(train_name[0]).tolist()
    return (train_name)

if __name__ == '__main__':
    x = [1, None]
    a=[1]; b=2
    CH_RetrieveData(x, a, b)
