from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from CreatFile import RandomForestFileResult, NaiveBayesFileResult, PredictFile, DecisionTreeFileResult
from OutDataInput import CH_RetrieveData, CH_RetrievePredict
import os

#随机森林分类训练
def CH_RandomForest(UserID, X, Y, P=[100, 50, 2, 50, 0.8, 2]):
    data = CH_RetrieveData(Data_X=X, Data_Y=Y, UserID=UserID)
    print(data[0])
    if data:
        rfc = RandomForestClassifier(n_estimators=P[0], max_depth=P[1], min_samples_leaf=P[2], max_leaf_nodes=P[3], max_features=P[4], min_samples_split=P[5])
        rfc.fit(data[0], data[2])
        P_test = [rfc.score(data[1], data[3])]
        RandomForestFileResult(X, Y, P_test, P[:1], P[1:2], P[2:3], P[3:4], P[4:5], P[5:6], UserID)
        #保存模型
        path = os.path.abspath('../UserFiles/'+ str(UserID) +'/Result/0.pkl')
        joblib.dump(rfc, path)
        return P_test
    else:
        error = '请选择训练数据'
        return error

#随机森林预测
def CH_RandomForestPredict(UserID, X,ID):
    Predict = CH_RetrievePredict(DataList=X, UserID=UserID)
    if [Predict]:
        path = os.path.abspath('../UserFiles/'+ str(UserID) +'/Result/0.pkl')
        rfc = joblib.load(path)
        print(Predict)
        Result_predict = rfc.predict(Predict)
        PredictFile(Model='RandomForest', ID=ID, Result=Result_predict, UserID=UserID)
        return 1
    return 0

#决策树分类训练
def CH_DecisionTree(UserID, X, Y, P=[50, 2, 50, 0.8, 2]):
    data = CH_RetrieveData(Data_X=X, Data_Y=Y, UserID=UserID)
    if data:
        dtc = DecisionTreeClassifier(max_depth=P[0], min_samples_leaf=P[1], max_leaf_nodes=P[2], max_features=P[3], min_samples_split=P[4])
        dtc.fit(data[0], data[2])
        P_test = [dtc.score(data[1], data[3])]
        DecisionTreeFileResult(X, Y, P_test, P[:1], P[1:2], P[2:3], P[3:4], P[4:5], UserID)   #生成结果分析文件
        #保存模型
        path = os.path.abspath('../UserFiles/'+ str(UserID) +'/Result/DecisionTree.pkl')
        joblib.dump(dtc, path)
        return P_test
    else:
        error = '请选择训练数据'
        return error

#决策树预测
def CH_DecisionTreePredict(UserID, X,ID):
    Predict = CH_RetrievePredict(DataList=X, UserID=UserID)
    if [Predict]:
        path = os.path.abspath('../UserFiles/'+ str(UserID) +'/Result/DecisionTree.pkl')
        dtc = joblib.load(path)
        Result_predict = dtc.predict(Predict)
        PredictFile(Model='DecisionTree', ID=ID, Result=Result_predict, UserID=UserID)
        return 1
    return 0

#朴素贝叶斯分类训练
def CH_NaiveBayes(UserID, X, Y):
    data = CH_RetrieveData(Data_X=X, Data_Y=Y, UserID=UserID)
    if data:
        mnb = MultinomialNB()
        mnb.fit(data[0], data[2])
        P_test = [mnb.score(data[1], data[3])]
        NaiveBayesFileResult(X, Y, P_test,UserID)
        #保存模型
        path = os.path.abspath('../UserFiles/' + str(UserID) + '/Result/NaiveBayes.pkl')
        joblib.dump(mnb, path)
        return P_test
    else:
        error = '请选择训练数据'
        return error

#    朴素贝叶斯预测
def CH_NaiveBayesPredict(UserID, X,ID):
    Predict = CH_RetrievePredict(DataList=X, UserID=UserID)
    if [Predict]:
        path = os.path.abspath('../UserFiles/'+ str(UserID) +'/Result/NaiveBayes.pkl')
        mnb = joblib.load(path)
        Result_predict = mnb.predict(Predict)
        PredictFile(Model='NaiveBayes', ID=ID, Result=Result_predict, UserID=UserID)
        return 1
    return 0

#神经网络分类


if __name__ == '__main__':
    Result1 = CH_RandomForestClassifier(DataX=['AGE', 'GENDER'], DataY=['GENDER'], UserID=123456)
    print(Result1)
    Result2 = CH_DecisionTree(DataX=['AGE', 'GENDER'], DataY=['GENDER'], UserID=123456)
    print(Result2)
    Result3 = CH_NaiveBayes(DataX=['AGE', 'GENDER'], DataY=['GENDER'], UserID=123456)
    print(Result3)



