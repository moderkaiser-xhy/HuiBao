import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from MySQL import extract_data
import joblib

#线性回归模型
def Linear_Regression_Model(User_id,xfeature,yfeature):
    #从数据库中获取数据
    try:
        table_name=""+User_id+"train1"
        table_name1 = ""+User_id+"predict1"
        dataset_1 = extract_data(table_name,xfeature)
        dataset_2 = extract_data(table_name,yfeature)
        dataset_3 = extract_data(table_name1, xfeature)
        name_x = np.array(dataset_1[0])
        name_y = np.array(dataset_2[0])
        name_xp = np.array(dataset_3[0])
        x_dataset = dataset_1[1]
        y_dataset = dataset_2[1]
        xp_dataset = dataset_3[1]
        def transpose(matrix):
            return zip(*matrix)
        x_dataset = transpose(x_dataset)
        x = pd.DataFrame(x_dataset)
        x.columns = name_x
        xp_dataset = transpose(xp_dataset)
        xp = pd.DataFrame(xp_dataset)
        xp.columns = name_xp
        y_dataset = transpose(y_dataset)
        y = pd.DataFrame(y_dataset)
        y.columns = name_y

        # 空值填充
       	x = x.replace('',np.NAN)
        xp = xp.replace('', np.NAN)
        y = y.replace('',np.NAN)
        x = x.fillna(method='pad')
        xp = xp.fillna(method='pad')
        y = y.fillna(method='pad')


        #线性回归
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.abspath(os.path.join(os.getcwd(), "..") + "/UserFiles/" + User_id + "/Result/" + User_id + "_Linear_Regression_Model.pkl"))
        yp = model.predict(xp)
        yp = yp.tolist()
        yp = pd.DataFrame(yp)
        yp.columns = [""+yfeature[0]+"预测结果"]
        pred_result = pd.concat((xp, yp), axis=1, ignore_index=False)
        pred_result.to_csv(os.path.abspath(os.path.join(os.getcwd(), "..") + "/UserFiles/" + User_id + "/Result/" + User_id + "_Linear_Regression_Model_Pred_Result.csv"),index=False)
        test_score  = model.score(X_test, y_test)

        #结果输出到txt文件
        def result_write(User_id, list1, list2, yfeature, test_score):
            q = ''
            for i in range(0, len(list2)):
                q = q + "X" + str(i + 1) + ":" + list2[i] + ","
            w = "Y:" + yfeature[0] + "\n"
            Result = "Y = (" + list1[0] + ")+"
            for i in range(1, len(list1)):
                Result = Result + "(" + list1[i] + "*X" + str(i) + ")+"
            Result = Result[:-1] + '\n'
            q = q[:-1] + '\n'
            score = "score = " + str(test_score) + "" + '\n'
            with open(os.path.abspath(os.path.join(os.getcwd(), "..") + "/UserFiles/" + User_id + "/Result/" + User_id + "_Linear_Regression_Model_Result.txt"), "w", encoding='utf-8') as f:
                f.write(str(Result))  # 表达式
                f.write(str(q))  # 表达式中x说明
                f.write(str(w))  # 表达式中y说明
                f.write(str(score))  # 测试集得分
                f.close()
            return
        list1 = list(model.intercept_)+list(model.coef_[0])
        list1 = [str(i) for i in list1]
        list2 = list(x.columns)
        result_write(User_id, list1, list2, yfeature, test_score)
        #画图
        predicted = model.predict(x)
        #设置标题
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('true_vs_prediction')
        plt.scatter(y, predicted, color='y',marker='o',label='predicted data')
        plt.scatter(y, y, color='g', marker='+',label='Original data')
        plt.legend(['predicted data', 'Original data'], loc = 'upper right')
        plt.savefig(os.path.abspath(os.path.join(os.getcwd(), "..") + "/UserFiles/" + User_id + "/Result/" + User_id + "_Linear_Regression_map.png"))#保存图片
        return 1
    except:
        return 0

if __name__ =='__main__':

    User_id = '123456789'
    xfeature1=['AGE','GENDER']
    yfeature1=['ACTIVE_DAY_09']
    xfeature2 = ['AGE', 'GENDER']
    yfeature2 = ['AGE']
    c=Linear_Regression_Model(User_id,xfeature1,yfeature1)
    d=Linear_Regression_Model(User_id,xfeature2,yfeature2)
    print(c)
    print(d)


#线性回归模型展示结果
def Display_Linear_Regression_Model_Rsults(User_id):
    def text_read(User_id):
        try:
            file_path = "../UserFiles/" + User_id + "/Result/" + User_id + "_Linear_Regression_Model_Result.txt"
            file = open(file_path, 'r')
        except IOError:
            error = []
            return error
        content = file.readlines()
        file.close()
        return content                                                                                                
    Result = text_read(User_id)[0]
    xfeature = text_read(User_id)[1]
    yfeature = text_read(User_id)[2]
    Score = text_read (User_id)[3]
    true_vs_prediction_map = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ User_id +'_true_vs_prediction_map.png'
    return(xfeature,yfeature,Result,Score,true_vs_prediction_map)



























