from flask import Flask, request, jsonify, render_template, json, make_response, send_from_directory, make_response
import numpy as np
from Linear_Regression_Model import Linear_Regression_Model, Display_Linear_Regression_Model_Rsults
import os
from Lstm_Model import Lstm_Model_train, Lstm_Model_test, Display_LSTM_Model_Results
from nnw import Neural_Network_Model_train, Neural_Network_Model_test, Display_NNW_Model_Results
from kmeans222 import Clustering_train, Display_Clustering_Results, Clustering_test
from Quality_analysis import Quality_analysis
from Data_preprocessing import Data_preprocessing
from Null_value_padding import Null_value_padding
from MySQL import delete_characteristic
from RandomClass import CH_DecisionTree, CH_RandomForestClassifier, CH_NaiveBayes, CreatDirectory, DeleteFile
from MySQL import excel_create_table, drop_table
import pandas as pd
from display_name import display
from show_name1 import show_name1
from access import access
from flask_cors import CORS
from Class import CH_RandomForest, CH_RandomForestPredict, CH_DecisionTree, CH_DecisionTreePredict, CH_NaiveBayes, \
    CH_NaiveBayesPredict
from OutDataInput import DatabaseCreatTrain, DatabaseCreatPredict, show_name
from CreatFile import CreatDirectory, DeleteFile, ShowName, FileUp_Down, FileLeft_Right, Localtime, userFiles, fileName, \
    det_files
import keras

app = Flask(__name__)

# --------------------------跨域调用------------------------------

CORS(app, supports_credentials=True)


@app.after_request
def af_request(resp):
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp


# ----------------------------------------------------------------
@app.route('/show_na', methods=['GET', 'POST'])  # 路由
def show_na():
    if request.method == 'POST':
        print(request)
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        print(User_id)

        if User_id != '':
            name = display(User_id)
            if name == [] or name[0] == "E":
                error1 = 'unable to fecth data'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": name, "msg": "成功"}
                return res
        else:
            error1 = '输入参数错误'
            res = {"code": 500, "msg": error1}
            return res


@app.route('/show_na1', methods=['GET', 'POST'])  # 路由
def show_na1():
    print(request)
    data = request.get_json()
    User_id = data["User_id"]

    if User_id != '':
        name = show_name1(User_id)
        if name == 0:
            error1 = 'unable to fecth data'
            res = {"code": 500, "msg": error1}
            return res
        else:
            res = {"code": 200, "data": name, "msg": "成功"}
            return res
    else:
        error1 = '输入参数错误'
        res = {"code": 500, "msg": error1}
        return res


# --------------------------权限控制------------------------------
@app.route('/', methods=['get', 'POST'])
def ac():
    return render_template('auth.html')


@app.route('/auth', methods=['get', 'POST'])
def auth():
    data = request.json
    userID = data["userID"]
    userName = data["userName"]
    token = data["token"]
    print(userID, userName, token)
    P = access(userID, userName, token)
    if P == "成功":
        res = {"code": 200, "msg": "成功"}
        return res
    else:
        error1 = "失败"
        res = {"code": 500, "msg": error1}
        return res


# ----------------------------------------------------------------


# --------------------------线性回归------------------------------

lrmodel_state = 1


@app.route('/lr1', methods=['GET', 'POST'])  # 路由
def lr1():
    global lrmodel_state
    if lrmodel_state == 1:
        if request.method == 'POST':
            print(request)
            data = request.get_json()
            print(data)
            User_id = data["User_id"]
            xfeature = data["xfeature"]
            yfeature = data["yfeature"]
            yfeature = [yfeature]
            if User_id != "" and xfeature != [""] and yfeature != [""]:
                print(User_id)
                print(xfeature)
                print(yfeature)
                lrmodel_state = 0
                state = Linear_Regression_Model(User_id, xfeature, yfeature)
                lrmodel_state = 1
                if state == 1:
                    res = {"code": 200, "msg": "线性回归成功"}
                    return res
                else:
                    res = {"code": 500, "msg": "线性回归失败"}
                    return res
            else:
                error1 = '输入参数错误'
                res = {"code": 500, "msg": error1}
                return res
    else:
        res = {"code": 500, "msg": "模型被占用"}
        return res


@app.route('/dr', methods=['get', 'POST'])
def dr():
    global lrmodel_state
    if lrmodel_state == 1:
        if request.method == 'POST':
            data = request.get_json()
            User_id = data["User_id"]
            filename = "" + User_id + "_Linear_Regression_Model_Result.txt"
            directory = "UserFiles/" + User_id + "/Result/"
            path = directory + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/" + path
            print(directory1)
            if os.path.isfile(directory1):
                res = {"code": 200, "data": path, "msg": "成功"}
                print(res)
                return res
            else:
                error1 = '无模型文件！'
                res = {"code": 500, "msg": error1}
                return res
    else:
        res = {"code": 500, "msg": "模型被占用"}
        return res


@app.route('/dp', methods=['get', 'POST'])
def dp():
    global lrmodel_state
    if lrmodel_state == 1:
        if request.method == 'POST':
            data = request.get_json()
            User_id = data["User_id"]

            filename = "" + User_id + "_Linear_Regression_Model_Pred_Result.csv"
            directory = "UserFiles/" + User_id + "/Result/"
            path = directory + filename
            print(path)
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/" + path
            print(directory1)
            if os.path.isfile(directory1):
                res = {"code": 200, "data": path, "msg": "成功"}
                print(res)
                return res
            else:
                error1 = '无模型文件！'
                res = {"code": 500, "msg": error1}
                return res
    else:
        res = {"code": 500, "msg": "模型被占用"}
        return res


@app.route('/dm', methods=['get', 'POST'])
def dm():
    global lrmodel_state
    if lrmodel_state == 1:
        if request.method == 'POST':
            data = request.get_json()
            User_id = data["User_id"]

            filename = "" + User_id + "_Linear_Regression_map.png"
            directory = "UserFiles/" + User_id + "/Result/"
            path = directory + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/" + path
            if os.path.isfile(directory1):
                res = {"code": 200, "data": path, "msg": "成功"}
                print(res)
                return res
            else:
                error1 = '无模型文件！'
                res = {"code": 500, "msg": error1}
                return res
    else:
        res = {"code": 500, "msg": "模型被占用"}
        return res


# ----------------------------------------------------------------


# -------------------------xhy------------------------------
Flag = 1


@app.route('/TrainFile', methods=['GET', 'POST'])
def TrainFile():
    try:
        if request.method == 'POST':
            UserID = request.form.get('UserID')
            CreatDirectory(UserID)
            f = request.files['file1']
            name = f.filename
            filename = fileName(name, "Train", UserID)
            path = os.path.abspath('../UserFiles/' + UserID + '/Train/' + filename)
            f.save(path)
            res = {"code": 200, "data": {}, "message": "上传成功"}
            print(res)
            return jsonify(res)
    except:
        res = {"code": 500, "message": "请选择上传文件"}
        print(res)
        return jsonify(res)


@app.route('/PredictFile', methods=['GET', 'POST'])
def PredictFile():
    try:
        if request.method == 'POST':
            UserID = request.form.get('UserID')
            CreatDirectory(UserID)
            f = request.files['file1']
            ''''''
            name = f.filename
            filename = fileName(name, "Predict", UserID)
            Path = os.path.abspath('../UserFiles/' + UserID + '/Predict/' + filename)
            f.save(Path)
            res = {"code": 200, "data": {}, "message": "上传成功"}
            print(res)
            return jsonify(res)
    except:
        res = {"code": 500, "message": "请选择上传文件"}
        print(res)
        return jsonify(res)


# 显示用户文件
@app.route('/DisplayFiles', methods=['GET', 'POST'])
def DisplayFiles():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        UserID = str(data["UserID"])
        CreatDirectory(UserID)
        Type = str(data["Type"])
        result = userFiles(UserID, Type)
        res = {"code": 200, "data": result, "message": "成功"}
        print(res)
        return jsonify(res)


# 删除文件
@app.route('/Del_file', methods=['GET', 'POST'])
def Del_file():
    if request.method == "POST":
        data = request.get_json()
        print(data)
        UserID = data['UserID']
        Type = data['Type']
        file_name = data["file_name"]
        result = det_files(UserID, Type, file_name)
        if result == "success":
            res = {"code": 200, "message": "删除成功", "data": {}}
            print(res)
            return jsonify(res)
        elif result == "文件不存在":
            res = {"code": 500, "message": "文件不存在"}
            return jsonify(res)


# 显示文件ID列名
@app.route('/File_stitch_show', methods=['GET', 'POST'])
def File_stitch_show():
    try:
        if request.method == "POST":
            data = request.get_json()
            print(data)
            UserID = str(data["UserID"])
            name1 = ShowName(UserID, 'Train')
            name2 = ShowName(UserID, 'Predict')
            res = {"code": 200, "data": {'Train': name1, 'Predict': name2}, "message": '成功'}
            print(res)
            return jsonify(res)
    except:
        res = {"code": 500, "message": "error1"}
        print(res)
        return jsonify(res)


# 文件拼接
@app.route('/FileStitch_Do', methods=['GET', 'POST'])
def FileStitchDo():
    data = request.get_json()
    print(data)
    UserID = str(data['UserID'])
    Type = int(data['Type'])
    Method = int(data['Method'])
    if Type == 0:  # 训练文件
        if Method == 0:  # 上下拼接
            TrainName = data['TrainName']  # [[文件名1, 关联列1], [文件名2, 关联列2]...]
            file = []
            for i in TrainName:
                file.append(i[0])
            print(file)
            FileUp_Down(UserID, 'Train', file)
            aa = DatabaseCreatTrain(UserID)
            print(aa)
            if aa == 1:
                res = {"data": {}, "code": 200, "message": '拼接成功'}
                print(res)
                return jsonify(res)
            else:
                res = {"code": 500, "message": "请使用不包含特殊字符的文件特征名"}
                print(res)
                return jsonify(res)
        elif Method == 1:  # 左右拼接
            ID_File1 = data['TrainName']  # [[文件名1, 关联列1], [文件名2, 关联列2]...]
            bb = FileLeft_Right(ID_File1, UserID, 'Train')
            if bb == 1:
                aa = DatabaseCreatTrain(UserID)
                print(aa)
                if aa == 1:
                    res = {"data": {}, "code": 200, "message": '拼接成功'}
                    print(res)
                    return jsonify(res)
                else:
                    res = {"code": 500, "message": "请使用不包含特殊字符的文件特征名"}
                    print(res)
                    return jsonify(res)
            if bb == 0:
                res = {"code": 500, "message": "关联列选取无效"}
                print(res)
                return jsonify(res)
    elif Type == 1:  # 预测文件
        if Method == 0:  # 上下拼接
            TrainName = data['TrainName']  # [[文件名1, 关联列1], [文件名2, 关联列2]...]
            file = []
            for i in TrainName:
                file.append(i[0])
            print(file)
            FileUp_Down(UserID, 'Predict', file)
            aa = DatabaseCreatPredict(UserID)
            print(aa)
            if aa == 1:
                res = {"data": {}, "code": 200, "message": '拼接成功'}
                print(res)
                return jsonify(res)
            else:
                res = {"code": 500, "message": "请使用不包含特殊字符的文件特征名"}
                print(res)
                return jsonify(res)
        elif Method == 1:  # 左右拼接
            ID_File1 = data['TrainName']  # [[文件名1, 关联列1], [文件名2, 关联列2]...]
            bb = FileLeft_Right(ID_File1, UserID, 'Predict')
            if bb == 1:
                aa = DatabaseCreatPredict(UserID)
                print(aa)
                if aa == 1:
                    res = {"data": {}, "code": 200, "message": '拼接成功'}
                    print(res)
                    return jsonify(res)
                else:
                    res = {"code": 500, "message": "请使用不包含特殊字符的文件特征名"}
                    print(res)
                    return jsonify(res)
            if bb == 0:
                res = {"code": 500, "message": "关联列选取无效"}
                print(res)
                return jsonify(res)


@app.route('/Data_show', methods=['GET', 'POST'])
def Data_show():
    print(request)
    data = request.get_json()
    User_id = data["UserID"]

    if User_id != '':
        name = show_name1(User_id)
        if name == 0:
            error1 = 'unable to fecth data'
            res = {"code": 500, "msg": error1}
            return res
        else:
            res = {"code": 200, "data": name, "msg": "成功"}
            return res
    else:
        error1 = '输入参数错误'
        res = {"code": 500, "msg": error1}
        return res


@app.route('/Classification', methods=['GET', 'POST'])
def Classification():
    global Flag

    # try:
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        UserID = str(data['UserID'])
        Model = int(data['Model'])
        XFeatures = data['X']
        YFeatures = data['Y']
        # try:
        if Model == 0:  # 使用随机森林模型分类
            a = data['n_estimators'];
            b = data['max_depth']
            c = data['min_sample_leaf'];
            d = data['max_leaf_nodes']
            e = data['max_features'];
            f = data['min_samples_split']
            Parameter = [a, b, c, d, e, f]
            Result = CH_RandomForest(UserID, XFeatures, YFeatures, Parameter)  # 训练
        elif Model == 1:  # 使用决策树模型分类
            b = data['max_depth'];
            c = data['min_sample_leaf']
            d = data['max_leaf_nodes'];
            e = data['max_features'];
            f = data['min_samples_split']
            Parameter = [b, c, d, e, f]
            Result = CH_DecisionTree(UserID, XFeatures, YFeatures, Parameter)
        elif Model == 2:  # 使用朴素贝叶斯模型分类
            Result = CH_NaiveBayes(UserID, XFeatures, YFeatures)
        res = {"data": {}, "code": 200, "message": "训练成功"}
        print(res)
        return jsonify(res)
        # except:
        # res = {"data":{}, "code": 500, "message": "训练失败"}
        # return jsonify(res)
    # except:
    # res = {'code': 500, "message": '数据错误'}
    # return jsonify(res)


@app.route('/ClassPredict', methods=['GET', 'POST'])
def ClassPredict():
    global Flag
    if request.method == "POST":
        data = request.get_json()
        print(data)
        UserID = data['UserID']
        Model = int(data['Model'])
        X = data['X']
        ID = data["ID"]
        if Model == 0:
            CH_RandomForestPredict(UserID, X, ID)
        elif Model == 1:
            CH_DecisionTreePredict(UserID, X, ID)
        elif Model == 2:
            CH_NaiveBayesPredict(UserID, X, ID)
        res = {"data": {}, 'code': 200, 'message': '预测成功'}
        print(res)
        Flag = 1
        return jsonify(res)
        res = {"data": {}, "code": 500, "message": '预测失败'}


# 下载结果文件
@app.route('/Down_Model', methods=['GET', 'POST'])
def Down_Model():
    try:
        if Flag == 0:
            res = {"code": 300, "message": "模型运行中"}
            print(res)
            return jsonify(res)
        else:
            if request.method == "POST":
                data = request.get_json()
                print(data)
                UserID = data['UserID']
                Model = data['Model']
                if Model == 0:
                    path = 'UserFiles/' + str(UserID) + '/Result/RandomForest.pkl'
                    file_path = path
                    res = {'data': {"FilePath": file_path}, "code": 200, 'message': '成功'}
                    print(res)
                    return jsonify(res)
                elif Model == 1:
                    path = 'UserFiles/' + str(UserID) + '/Result/DecisionTree.pkl'
                    file_path = path
                    res = {'data': {"FilePath": file_path}, "code": 200, 'message': '成功'}
                    return jsonify(res)
                elif Model == 2:
                    path = 'UserFiles/' + str(UserID) + '/Result/NaiveBayes.pkl'
                    file_path = path
                    res = {'data': {"FilePath": file_path}, "code": 200, 'message': '成功'}
                    return jsonify(res)
    except:
        res = {'code': 500, 'message': 'error1'}
        print(res)
        return jsonify(res)


@app.route('/Down_Predict', methods=['GET', 'POST'])
def Down_Predict():
    try:
        if Flag == 0:
            res = {"code": 300, "message": "模型运行中"}
            print(res)
            return jsonify(res)
        else:
            data = request.get_json()
            UserID = data['UserID']
            Model = data['Model']
            if Model == 0:
                path = 'UserFiles/' + str(UserID) + '/Result/PredictRandomForest.csv'
                file_path = path
                res = {'data': {"FilePath": file_path}, "code": 200, 'message': '成功'}
                return jsonify(res)
            elif Model == 1:
                path = 'UserFiles/' + str(UserID) + '/Result/PredictDecisionTree.csv'
                file_path = path
                res = {'data': {"FilePath": file_path}, "code": 200, 'message': '成功'}
                return jsonify(res)
            elif Model == 2:
                path = 'UserFiles/' + str(UserID) + '/Result/PredictNaiveBayes.csv'
                file_path = path
                res = {'data': {"FilePath": file_path}, "code": 200, 'message': '成功'}
                return jsonify(res)
    except:
        res = {'code': 500, 'message': 'error1'}
        return jsonify(res)


@app.route('/Down_Model_Report', methods=['GET', 'POST'])
def Down_Model_Report():
    try:
        if Flag == 0:
            res = {"code": 300, "message": "模型运行中"}
            print(res)
            return jsonify(res)
        else:
            data = request.get_json()
            UserID = data['UserID']
            Model = data['Model']
            if Model == 0:
                path = 'UserFiles/' + str(UserID) + '/Result/RandomForest.csv'
                file_path = path
                res = {'data': {"FilePath": file_path}, "code": 200, 'message': '成功'}
                return jsonify(res)
            elif Model == 1:
                path = 'UserFiles/' + str(UserID) + '/Result/DecisionTree.csv'
                file_path = path
                res = {'data': {"FilePath": file_path}, "code": 200, 'message': '成功'}
                return jsonify(res)
            elif Model == 2:
                path = 'UserFiles/' + str(UserID) + '/Result/NaiveBayes.csv'
                file_path = path
                res = {'data': {"FilePath": file_path}, "code": 200, 'message': '成功'}
                return jsonify(res)
    except:
        res = {'code': 500, 'message': 'error1'}
        return jsonify(res)


@app.route('/Exit', methods=['GET', 'POST'])
def Exit():
    try:
        if request.method == 'POST':
            data = request.get_json()
            UserID = data['UserID']
            DeleteFile(UserID)
            try:
                drop_table(UserID)
            except:
                aaaa = 1
            res = {"code": 200, "data": {}, "message": "success"}
            print(res)
            return jsonify(res)
    except:
        res = {"code": 500, "message": "error1"}
        return jsonify(res)


# ----------------------------------------------------------------


# -------------------------wsq------------------------------
def Judge(data):
    list1 = list(data.keys())
    print(len(list1))
    for i in range(len(list1)):
        a = []
        a.append(data["" + list1[i] + ""])
        b = np.array(a)
        c = b.ndim
        if c > 1:
            x = 2
            break
        else:
            x = 1
    return x


@app.route('/qaa_ch', methods=['GET', 'POST'])  # 路由
def qaa_ch():
    return render_template('Quality_analysis.html')


# 质量分析页面输出训练文件与预测文件的数据缺失率，列值占比，平均值
@app.route('/qaa_ch0', methods=['GET', 'POST'])  # 路由
def qaa_ch0():
    if request.method == 'POST':
        print(request)
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        max_m = data['max_m']
        max_v = data['max_v']
        max_c = data['max_c']
        # print(User_id)

        if User_id != '':

            try:
                if max_m == '':
                    max_m = 100000
                if max_v == '':
                    max_v = 100000
                if max_c == '':
                    max_c = 100000
                max_m = int(max_m)
                max_v = int(max_v)
                max_c = int(max_c)
                result = Quality_analysis(User_id, max_m, max_v, max_c)

                miss = result.miss()
                Standard_deviation_train = result.Standard_deviation_train()
                Correlation_coefficient_train = result.Correlation_coefficient_train()
                Discrete_coefficient_train = result.Discrete_coefficient_train()
                miss_test = result.miss_test()
                Standard_deviation_test = result.Standard_deviation_test()
                Correlation_coefficient_test = result.Correlation_coefficient_test()
                Discrete_coefficient_test = result.Discrete_coefficient_test()
                qqa_result1 = {
                    'train_miss': miss,
                    'test_Correlation_coefficient': Correlation_coefficient_test,
                    'train_Standard_deviation': Standard_deviation_train,
                    'train_Correlation_coefficient': Correlation_coefficient_train,
                    'train_Discrete_coefficient': Discrete_coefficient_train,
                    'test_miss': miss_test,
                    'test_Standard_deviation': Standard_deviation_test,

                    'test_Discrete_coefficient': Discrete_coefficient_test
                }
                res = {"code": 200, "data": qqa_result1, "msg": "成功"}
                return res
            except:
                error1 = '缺少预测文件！'
                res = {"code": 500, "msg": error1}
                return res
        else:
            error1 = '输入参数错误'
            res = {"code": 500, "msg": error1}
            return res


@app.route('/Fillnull_ch', methods=['GET', 'POST'])  # 路由
def Fillnull_ch():
    if request.method == 'POST':
        print(request)
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        type1 = data["type"]  # type  1：训练   2：预测
        item = data["item"]  # 空值项填充0：0填充 1：顺序  2：倒序
        feature = data["feature"]
        any_num = data["any_num"]
        # print(User_id,type1,item,feature)

        if int(type1) == 1:
            if ('' not in feature) and User_id != '':
                result = Null_value_padding(int(type1), item, feature, User_id, any_num)
                state = result.fill_nall()

                if state == 1:
                    res = {"code": 200, "msg": "空值填充成功"}
                    return res
                else:
                    error1 = '空值填充失败'
                    res = {"code": 500, "msg": error1}
                    return res

            else:

                error1 = '输入参数错误'
                res = {"code": 500, "msg": error1}
                return res




        else:
            if ('' not in feature) and User_id != '':
                result = Null_value_padding(int(type1), item, feature, User_id, any_num)
                state = result.fill_nall()

                if state == 1:
                    res = {"code": 200, "msg": "空值填充成功"}
                    return res
                else:
                    error1 = '空值填充失败'
                    res = {"code": 500, "msg": error1}
                    return res

            else:

                error1 = '输入参数错误'
                res = {"code": 500, "msg": error1}
                return res


# 从文件拼接后的两个数据库表user_id+train  和  中提取数据user_id+predict+predict,选择特征保存成新表train1和predict1
@app.route('/Datap_ch0', methods=['GET', 'POST'])  # 路由
def Datap_ch0():
    if request.method == 'POST':
        print(request)
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        type1 = data["type"]  # type  1：保存   2：删除
        feature = data["feature"]
        # print(User_id,type1,item,feature)

        if int(type1) == 2:
            if ('' not in feature) and User_id != '':

                state = delete_characteristic(User_id + 'train1', feature)
                state1 = delete_characteristic(User_id + 'predict1', feature)
                print(state, state1)
                if state == 'ERROR' or state1 == "ERROR":
                    error1 = '训练文件与测试文件特征名不一致！'
                    res = {"code": 500, "msg": error1}
                    return res
                elif state == 1 and state1 == 1:
                    res = {"code": 200, "msg": "删除成功"}
                    return res
                else:
                    error1 = '删除失败'
                    res = {"code": 500, "msg": error1}
                    return res

            else:

                error1 = '输入参数错误'
                res = {"code": 500, "msg": error1}
                return res




        else:
            if User_id != '' and ('' not in feature):

                try:

                    result = Data_preprocessing(feature, User_id)
                    state = result.Num_change1()
                except:
                    error1 = '训练文件与测试文件特征名不一致！'
                    res = {"code": 500, "msg": error1}
                    return res
                if state == 1:
                    res = {"code": 200, "msg": "保存成功"}
                    return res
                else:
                    error1 = '保存失败'
                    res = {"code": 500, "msg": error1}
                    return res

            else:

                error1 = '输入参数错误'
                res = {"code": 500, "msg": error1}
                return res


@app.route('/nnw_ch', methods=['GET', 'POST'])  # 路由
def nnw_ch():
    return render_template('nnw_ch.html')


check_nnw_state = 1


@app.route('/nnw_ch0', methods=['GET', 'POST'])  # 路由
def nnw_ch0():
    global check_nnw_state
    if 1 == check_nnw_state:

        print(request)
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        feature = data["feature"]
        N_Last_Layer_Neurons = data["N_Last_Layer_Neurons"]
        N_Layers1 = data["N_Layers1"]
        N_First_Layer_Neurons = data["N_First_Layer_Neurons"]
        N_Layer_Neurons2 = data["N_Layer_Neurons2"]
        N_Layer_Neurons3 = data["N_Layer_Neurons3"]
        N_Layer_Neurons4 = data["N_Layer_Neurons4"]
        N_Layer_Neurons5 = data["N_Layer_Neurons5"]
        epoch = data["epoch"]
        if User_id != '' and ('' not in feature) and ('' not in N_Last_Layer_Neurons):
            try:

                if N_First_Layer_Neurons == '':
                    N_First_Layer_Neurons = 16
                if N_Layers1 == '':
                    N_Layers1 = 2
                if N_Layer_Neurons2 == '':
                    N_Layer_Neurons2 = 64
                if N_Layer_Neurons3 == '':
                    N_Layer_Neurons3 = 128
                if N_Layer_Neurons4 == '':
                    N_Layer_Neurons4 = 256
                if N_Layer_Neurons5 == '':
                    N_Layer_Neurons5 = 256
                if epoch == '':
                    epoch = 10

                N_Layers1 = int(N_Layers1)
                N_First_Layer_Neurons = int(N_First_Layer_Neurons)
                N_Layer_Neurons2 = int(N_Layer_Neurons2)
                N_Layer_Neurons3 = int(N_Layer_Neurons3)
                N_Layer_Neurons4 = int(N_Layer_Neurons4)
                N_Layer_Neurons5 = int(N_Layer_Neurons5)
                epoch = int(epoch)
                if epoch > 500:
                    epoch = 500
                check_nnw_state = 0
                keras.backend.clear_session()
                state = Neural_Network_Model_train(User_id, feature, N_Last_Layer_Neurons, N_Layers1,
                                                   N_First_Layer_Neurons, N_Layer_Neurons2, N_Layer_Neurons3,
                                                   N_Layer_Neurons4, N_Layer_Neurons5, epoch)
                check_nnw_state = 1

                if state == 1:
                    res = {"code": 200, "msg": "训练成功"}
                    print(res)
                    return res
                elif state == 2:
                    check_nnw_state = 1
                    res = {"code": 500, "msg": "获取数据失败"}
                    print(res)
                    return res

                else:
                    check_nnw_state = 1
                    res = {"code": 500, "msg": "训练失败"}
                    return res
            except:
                check_nnw_state = 1
                error1 = '预测文件中缺少选中特征！'
                res = {"code": 500, "msg": error1}
                return res

        else:
            check_nnw_state = 1
            error1 = '输入参数错误'
            res = {"code": 500, "msg": error1}
            return res
    else:

        res = {"code": 500, "msg": "模型被占用"}
        return res


@app.route('/nnw_result', methods=['GET', 'POST'])  # 路由
def nnw_result():
    global check_nnw_state
    if check_nnw_state == 1:

        if request.method == 'POST':
            print(request)
            data = request.get_json()
            print(data)
            ID = data['ID']
            User_id = data["User_id"]
            feature = data["feature"]
            N_Last_Layer_Neurons = data["N_Last_Layer_Neurons"]
            N_Layers1 = data["N_Layers1"]
            N_First_Layer_Neurons = data["N_First_Layer_Neurons"]
            N_Layer_Neurons2 = data["N_Layer_Neurons2"]
            N_Layer_Neurons3 = data["N_Layer_Neurons3"]
            N_Layer_Neurons4 = data["N_Layer_Neurons4"]
            N_Layer_Neurons5 = data["N_Layer_Neurons5"]
            epoch = data["epoch"]
            if User_id != '' and ('' not in feature) and ('' not in N_Last_Layer_Neurons) and ('' not in ID):
                if N_First_Layer_Neurons == '':
                    N_First_Layer_Neurons = 16
                if N_Layers1 == '':
                    N_Layers1 = 2
                if N_Layer_Neurons2 == '':
                    N_Layer_Neurons2 = 64
                if N_Layer_Neurons3 == '':
                    N_Layer_Neurons3 = 128
                if N_Layer_Neurons4 == '':
                    N_Layer_Neurons4 = 256
                if N_Layer_Neurons5 == '':
                    N_Layer_Neurons5 = 256
                if epoch == '':
                    epoch = 10
                N_Layers1 = int(N_Layers1)
                N_First_Layer_Neurons = int(N_First_Layer_Neurons)
                N_Layer_Neurons2 = int(N_Layer_Neurons2)
                N_Layer_Neurons3 = int(N_Layer_Neurons3)
                N_Layer_Neurons4 = int(N_Layer_Neurons4)
                N_Layer_Neurons5 = int(N_Layer_Neurons5)
                epoch = int(epoch)
                if epoch > 500:
                    epoch = 500
                check_nnw_state = 0
                keras.backend.clear_session()
                state = Neural_Network_Model_test(User_id, ID, feature, N_Last_Layer_Neurons, N_Layers1,
                                                  N_First_Layer_Neurons, N_Layer_Neurons2, N_Layer_Neurons3,
                                                  N_Layer_Neurons4, N_Layer_Neurons5, epoch)
                check_nnw_state = 1
                if state == 1:
                    res = {"code": 200, "msg": "预测成功"}
                    return res
                else:
                    check_nnw_state = 1
                    error1 = '预测失败'
                    res = {"code": 500, "msg": error1}
                    return res

            else:
                check_nnw_state = 1
                error1 = '输入参数错误'
                res = {"code": 500, "msg": error1}
                return res


    else:
        res = {"code": 500, "msg": "模型被占用"}
        return res


@app.route('/file_jiegou', methods=['get', 'POST'])
def file_jiegou():
    print(request)
    global check_nnw_state
    if check_nnw_state == 1:
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        if User_id != '':

            filename = User_id + '_Neural_Network_Model_weightsss.h5'
            directory = 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(
                User_id) + '/' + 'Result' + '/' + filename
            if os.path.exists(directory1) == False:
                error1 = '无模型文件！'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": directory, "msg": "成功"}
                return res
    else:
        error1 = '模型运行中！'
        res = {"code": 300, "msg": error1}
        return res


@app.route('/file_baogao', methods=['get', 'POST'])
def file_baogao():
    print(request)
    global check_nnw_state
    if check_nnw_state == 1:
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        if User_id != '':

            filename = User_id + '_Neural_Network_Model.csv'
            directory = 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(
                User_id) + '/' + 'Result' + '/' + filename
            if os.path.exists(directory1) == False:
                error1 = '无报告文件！'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": directory, "msg": "成功"}
                return res
    else:
        error1 = '模型运行中！'
        res = {"code": 300, "msg": error1}
        return res


@app.route('/file_pre', methods=['get', 'POST'])
def file_pre():
    print(request)
    global check_nnw_state
    if check_nnw_state == 1:
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        if User_id != '':

            filename = User_id + '_Neural_Network_Model_Pred_result.csv'
            directory = 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(
                User_id) + '/' + 'Result' + '/' + filename
            if os.path.exists(directory1) == False:
                error1 = '无预测文件！'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": directory, "msg": "成功"}
                return res
    else:
        error1 = '模型运行中！'
        res = {"code": 300, "msg": error1}
        return res


@app.route('/LSTM_ch', methods=['GET', 'POST'])  # 路由
def lstm_ch():
    return render_template('LSTM_ch.html')


check_lstm_state = 1


@app.route('/LSTM_ch0', methods=['GET', 'POST'])  # 路由
def LSTM_ch0():
    global check_lstm_state
    if check_lstm_state == 1:

        if request.method == 'POST':
            print(request)
            data = request.get_json()
            print(data)
            User_id = data["User_id"]
            X_Data_Characteristic_n0 = data["X_Data_Characteristic_n0"]
            Y_Data_Characteristic = data["Y_Data_Characteristic"]
            N_Layers2 = data["N_Layers2"]
            N_First_Layer_Neurons = data["N_First_Layer_Neurons"]
            N_Layer_Neurons2 = data["N_Layer_Neurons2"]
            N_Layer_Neurons3 = data["N_Layer_Neurons3"]
            N_Layer_Neurons4 = data["N_Layer_Neurons4"]
            N_Layer_Neurons5 = data["N_Layer_Neurons5"]
            epoch = data["epoch"]
            if User_id != '' and ('' not in X_Data_Characteristic_n0) and ('' not in Y_Data_Characteristic):

                if N_First_Layer_Neurons == '':
                    N_First_Layer_Neurons = 5
                if N_Layers2 == '':
                    N_Layers1 = 2
                if N_Layer_Neurons2 == '':
                    N_Layer_Neurons2 = 5
                if N_Layer_Neurons3 == '':
                    N_Layer_Neurons3 = 5
                if N_Layer_Neurons4 == '':
                    N_Layer_Neurons4 = 5
                if N_Layer_Neurons5 == '':
                    N_Layer_Neurons5 = 5
                if epoch == '':
                    epoch = 10

                N_Layers2 = int(N_Layers2)
                N_First_Layer_Neurons = int(N_First_Layer_Neurons)
                N_Layer_Neurons2 = int(N_Layer_Neurons2)
                N_Layer_Neurons3 = int(N_Layer_Neurons3)
                N_Layer_Neurons4 = int(N_Layer_Neurons4)
                N_Layer_Neurons5 = int(N_Layer_Neurons5)
                epoch = int(epoch)
                if epoch > 500:
                    epoch = 500
                check_lstm_state = 0
                keras.backend.clear_session()
                state = Lstm_Model_train(User_id, X_Data_Characteristic_n0, Y_Data_Characteristic, N_Layers2,
                                         N_First_Layer_Neurons, N_Layer_Neurons2, N_Layer_Neurons3, N_Layer_Neurons4,
                                         N_Layer_Neurons5, epoch)
                check_lstm_state = 1
                if state == 1:
                    res = {"code": 200, "msg": "训练成功"}
                    return res
                else:
                    check_lstm_state == 1
                    res = {"code": 500, "msg": "训练失败"}
                    return res


            else:
                check_lstm_state == 1
                error1 = '输入参数错误'
                res = {"code": 500, "msg": error1}
                return res
    else:
        res = {"code": 500, "msg": "模型被占用"}
        return res


@app.route('/lstm_result', methods=['GET', 'POST'])  # 路由
def lstm_result():
    global check_lstm_state
    if check_lstm_state == 1:

        if request.method == 'POST':
            print(request)
            data = request.get_json()
            print(data)
            User_id = data["User_id"]
            X_Data_Characteristic_n0 = data["X_Data_Characteristic_n0"]
            Y_Data_Characteristic = data["Y_Data_Characteristic"]
            N_Layers2 = data["N_Layers2"]
            N_First_Layer_Neurons = data["N_First_Layer_Neurons"]
            N_Layer_Neurons2 = data["N_Layer_Neurons2"]
            N_Layer_Neurons3 = data["N_Layer_Neurons3"]
            N_Layer_Neurons4 = data["N_Layer_Neurons4"]
            N_Layer_Neurons5 = data["N_Layer_Neurons5"]
            epoch = data["epoch"]
            if User_id != '' and ('' not in X_Data_Characteristic_n0) and ('' not in Y_Data_Characteristic):
                try:
                    if N_First_Layer_Neurons == '':
                        N_First_Layer_Neurons = 5
                    if N_Layers2 == '':
                        N_Layers1 = 2
                    if N_Layer_Neurons2 == '':
                        N_Layer_Neurons2 = 5
                    if N_Layer_Neurons3 == '':
                        N_Layer_Neurons3 = 5
                    if N_Layer_Neurons4 == '':
                        N_Layer_Neurons4 = 5
                    if N_Layer_Neurons5 == '':
                        N_Layer_Neurons5 = 5
                    if epoch == '':
                        epoch = 10

                    N_Layers2 = int(N_Layers2)
                    N_First_Layer_Neurons = int(N_First_Layer_Neurons)
                    N_Layer_Neurons2 = int(N_Layer_Neurons2)
                    N_Layer_Neurons3 = int(N_Layer_Neurons3)
                    N_Layer_Neurons4 = int(N_Layer_Neurons4)
                    N_Layer_Neurons5 = int(N_Layer_Neurons5)
                    epoch = int(epoch)
                    if epoch > 500:
                        epoch = 500
                    check_lstm_state = 0
                    keras.backend.clear_session()
                    state = Lstm_Model_test(User_id, X_Data_Characteristic_n0, Y_Data_Characteristic, N_Layers2,
                                            N_First_Layer_Neurons,
                                            N_Layer_Neurons2, N_Layer_Neurons3, N_Layer_Neurons4, N_Layer_Neurons5,
                                            epoch)
                    check_lstm_state = 1
                    if state == 1:
                        res = {"code": 200, "msg": "预测成功"}
                        return res
                    else:
                        check_lstm_state == 1
                        error1 = '预测失败'
                        res = {"code": 500, "msg": error1}
                        return res
                except:
                    check_lstm_state == 1
                    error1 = '预测文件中缺少选中特征！'
                    res = {"code": 500, "msg": error1}
                    return res
            else:

                error1 = '输入参数错误'
                res = {"code": 500, "msg": error1}
                return res



    else:
        res = {"code": 500, "msg": "模型被占用"}
        return res


@app.route('/lstm_jiegou', methods=['get', 'POST'])
def lstm_jiegou():
    print(request)
    global check_lstm_state
    if check_lstm_state == 1:
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        if User_id != '':

            filename = User_id + '_Lstm_Model_weightsss.h5'
            directory = 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(
                User_id) + '/' + 'Result' + '/' + filename
            if os.path.exists(directory1) == False:
                error1 = '无模型文件！'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": directory, "msg": "成功"}
                return res
    else:
        error1 = '模型运行中！'
        res = {"code": 300, "msg": error1}
        return res


@app.route('/lstm_baogao', methods=['get', 'POST'])
def lstm_baogao():
    print(request)
    global check_lstm_state
    if check_lstm_state == 1:
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        if User_id != '':

            filename = User_id + '_LSTM_Model.csv'
            directory = 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(
                User_id) + '/' + 'Result' + '/' + filename

            if os.path.exists(directory1) == False:
                error1 = '无报告文件！'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": directory, "msg": "成功"}
                return res

    else:
        error1 = '模型运行中！'
        res = {"code": 300, "msg": error1}
        return res


@app.route('/lstm_pre', methods=['get', 'POST'])
def lstm_pre():
    print(request)
    global check_lstm_state
    if check_lstm_state == 1:
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        if User_id != '':

            filename = User_id + '_LSTM_Model_Pred_result.csv'
            directory = 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(
                User_id) + '/' + 'Result' + '/' + filename

            if os.path.exists(directory1) == False:
                error1 = '无预测文件！'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": directory, "msg": "成功"}
                return res
    else:
        error1 = '模型运行中！'
        res = {"code": 300, "msg": error1}
        return res


@app.route('/lstm_map', methods=['get', 'POST'])
def lstm_map():
    print(request)
    global check_lstm_state
    if check_lstm_state == 1:
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        if User_id != '':

            filename = User_id + '_true_vs_prediction_map1.png'
            directory = 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(
                User_id) + '/' + 'Result' + '/' + filename

            if os.path.exists(directory1) == False:
                error1 = '无验证集图片！'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": directory, "msg": "成功"}
                return res
    else:
        error1 = '模型运行中！'
        res = {"code": 300, "msg": error1}
        return res


check_kmeans_state = 1


@app.route('/Kmeans_ch', methods=['GET', 'POST'])  # 路由
def Kmeans_ch():
    return render_template('Kmeans_ch.html')


@app.route('/Kmeans_ch0', methods=['GET', 'POST'])  # 路由
def Kmeans_ch0():
    global check_kmeans_state
    if check_kmeans_state == 1:
        if request.method == 'POST':
            print(request)
            data = request.get_json()
            print(data)
            User_id = data["User_id"]
            feature_1 = data["feature_1"]

            k = data["k"]

            if User_id != '' and ('' not in feature_1):

                if k == '':
                    k = 2
                k = int(k)
                check_kmeans_state = 0
                state = Clustering_train(User_id, feature_1, k)
                check_kmeans_state = 1
                if state == 1:
                    res = {"code": 200, "msg": "聚类成功"}
                    return res
                else:
                    error1 = '聚类失败'
                    res = {"code": 500, "msg": error1}
                    return res

            else:

                error1 = '输入参数错误'
                res = {"code": 500, "msg": error1}
                return res


@app.route('/Kmeans_result', methods=['GET', 'POST'])  # 路由
def Kmeans_result():
    global check_kmeans_state
    if check_kmeans_state == 1:
        if request.method == 'POST':
            print(request)
            data = request.get_json()
            print(data)
            User_id = data["User_id"]
            feature_1 = data["feature_1"]

            k = data["k"]

            if User_id != '' and ('' not in feature_1):

                try:

                    if k == '':
                        k = 2
                    k = int(k)
                    check_kmeans_state = 0
                    state = Clustering_test(User_id, feature_1, k)
                    check_kmeans_state = 1
                    if state == 1:
                        res = {"code": 200, "msg": "预测成功"}
                        return res
                    else:
                        res = {"code": 500, "msg": "预测失败"}
                        return res
                except:
                    error1 = '预测文件中缺少选中特征！'
                    res = {"code": 500, "msg": error1}
                    return res

            else:
                error1 = '输入参数错误'
                res = {"code": 500, "msg": error1}
                return res


@app.route('/kmeans_baogao', methods=['get', 'POST'])
def kmeans_baogao():
    print(request)
    global check_kmeans_state
    if check_kmeans_state == 1:
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        if User_id != '':

            filename = User_id + '_Clustering_result_record.csv'
            directory = 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(
                User_id) + '/' + 'Result' + '/' + filename

            if os.path.exists(directory1) == False:
                error1 = '无报告文件！'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": directory, "msg": "成功"}
                return res

    else:
        error1 = '模型运行中！'
        res = {"code": 300, "msg": error1}
        return res


@app.route('/kmeans_pre', methods=['get', 'POST'])
def kmeans_pre():
    print(request)
    global check_kmeans_state
    if check_kmeans_state == 1:
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        if User_id != '':

            filename = User_id + '_Clustering_result.csv'
            directory = 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(
                User_id) + '/' + 'Result' + '/' + filename

            if os.path.exists(directory1) == False:
                error1 = '无预测文件！'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": directory, "msg": "成功"}
                return res
    else:
        error1 = '模型运行中！'
        res = {"code": 300, "msg": error1}
        return res


@app.route('/kmeans_trend', methods=['get', 'POST'])
def kmeans_trend():
    print(request)
    global check_kmeans_state
    if check_kmeans_state == 1:
        data = request.get_json()
        print(data)
        User_id = data["User_id"]
        if User_id != '':
            filename = User_id + '_cluster_trend_map.png'
            directory = 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + filename
            directory1 = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(
                User_id) + '/' + 'Result' + '/' + filename

            if os.path.exists(directory1) == False:
                error1 = '无趋势图片！'
                res = {"code": 500, "msg": error1}
                return res
            else:
                res = {"code": 200, "data": directory, "msg": "成功"}
                return res
    else:
        error1 = '模型运行中！'
        res = {"code": 300, "msg": error1}
        return res


# ----------------------------------------------------------------

if __name__ == '__main__':
    app.run()
