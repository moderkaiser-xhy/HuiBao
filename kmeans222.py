import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from MySQL import extract_data
from pandas.core.frame import DataFrame
import os
import matplotlib.pyplot as plt
'''
 k均值算法：
     1 随机选择k个样本作为k个类别的中心
     2 从k个样本出发，选取最近的样本归为和自己同一个分类，一直到所有样本都有分类
     3 对k个分类重新计算中心样本
     4 从k个新中心样本出发重复23，
         如果据类结果和上一次一样，则停止
         否则重复234
       
'''

#对聚类数据根据手肘法求最优K值
def Clustering_train(User_id,feature_1,max_k):
    #从数据库导入数据，将list的数据转化为Dataframe
    try:
        dataset_x = extract_data(User_id+'predict1', feature_1)
        name_x = np.array(dataset_x[0])
        x_dataset0 = dataset_x[1]
        def transpose(matrix):
            return zip(*matrix)

        x_dataset = transpose(x_dataset0)
        x = pd.DataFrame(x_dataset)
        x.columns = name_x
    except:
        return 0
    #将空字符串数据转化为Nan然后进行顺序填充
    x = x.replace('', np.NaN)
    x = x.fillna(method= 'pad')
    if max_k >30:
        max_k = 30
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1, max_k+1):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(x)
        SSE.append(estimator.inertia_)
    X = range(1, max_k+1)
    print(X,SSE)


    plt.plot(X, SSE, 'o-')
    plt.xlabel('K')

    plt.ylabel('SSE')
    path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles' +'/'+ str(User_id) +'/'+ 'Result' +'/' + str(User_id) + '_cluster_trend_map.png'

    #os.remove(path)
    plt.savefig(path)
    plt.close()#缺少此句 下一次绘图图片会叠加在上图
    return 1
#根据选择出的最优K值进行预测
def Clustering_test(User_id, feature_1, perfect_k):
    dataset_y = extract_data(User_id + 'predict1', feature_1)
    name_y = np.array(dataset_y[0])
    y_dataset0 = dataset_y[1]

    def transpose(matrix):
        return zip(*matrix)

    y_dataset = transpose(y_dataset0)
    x = pd.DataFrame(y_dataset)
    x.columns = name_y

    # digits_train = pd.read_csv('D:\\test(1).csv',encoding='gbk')
    # 从样本中抽取出64维度像素特征和1维度目标
    # x = digits_train[['GENDER', 'AGE', 'IN_NET_DUR','STAR_LEVEL']]
    x = x.replace('', np.NaN)
    x = x.fillna(method='pad')
    if perfect_k > 30:
        perfect_k = 30


    kmeans = KMeans(n_clusters=perfect_k)
    y_predict = kmeans.fit_predict(x)
    labels = kmeans.labels_ #预测出的类别标签
    #print(labels)
    SSE_Min = kmeans.inertia_
    cluster_centers = kmeans.cluster_centers_  #聚类中心
    #print(cluster_centers)
    #cluster_centers = np.array(cluster_centers).reshape(1,cluster_centers.shape[0]) #将其转化保存在csv文件方便读取模块读取
    #print(type(cluster_centers))


    y_predict = y_predict.reshape(y_predict.shape[0], 1)
    x_train_list = np.array(x).tolist()
    z = [x_train_list, y_predict]
    #print(z)

    data = DataFrame(z)  # 这时候是以行为标准写入的

    data = data.T  # 转置之后得到想要的结果
    data.columns = [feature_1,'聚类结果']
    #print(data)
    data.to_csv(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles'+'/'+ str(User_id) +'/'+ 'Result' +'/' + str(User_id) + '_Clustering_result.csv',index=False)


    def text_save(content, User_id,feature_1,perfect_k,mode = 'a'):
        # Try to save a list variable in txt file.
        t = os.path.abspath(os.path.join(os.getcwd(), ".."))
        newfile = t + '/'+ 'UserFiles' +'/'+ str(User_id) +'/'+ 'Result' +'/'  + str(User_id) +'_Clustering_result_record.csv'
        if not os.path.exists(newfile):
            f = open(newfile, 'w')
            print
            newfile
            f.close()
            print
            newfile + " created."
        else:
            print
            newfile + " already existed."
        file = open(newfile,mode)
        for i in range(len(content)):
            file.write(str(content[i]) + '\n')
        file.close()
    text_save(("特征：",feature_1,"最佳分类K值：",perfect_k,"最小残差平方和：",SSE_Min,"聚类中心点",cluster_centers),User_id,feature_1,perfect_k,mode = 'a')
    return 1

def Display_Clustering_Results(User_id,feature_1,perfect_k):
    try:
        file_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/' + str(User_id)  + '_Clustering_result_record.csv'
        # print(file_path)
        file = open(file_path, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()

    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]

    file.close()
    return content
if __name__ =='__main__':
    a = Clustering_train('wangshuqi',['GENDER',"AGE"],8)
    b = Clustering_test('wangshuqi',['GENDER',"AGE"],2)
    #Display_Clustering_Results('wangshuqi',['IPTV_USER_NUMBER'],20)