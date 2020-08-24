import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import LSTM, Dense, Activation
import pandas as pd
from MySQL import extract_data
import os
from keras import backend as K
def Lstm_Model_train(User_id, X_Data_Characteristic_n0, Y_Data_Characteristic, N_Layers2, N_First_Layer_Neurons,N_Layer_Neurons2, N_Layer_Neurons3,N_Layer_Neurons4,N_Layer_Neurons5,epoch):

    try:
        N1 = N_First_Layer_Neurons
        dataset_x = extract_data(User_id+'train1',X_Data_Characteristic_n0)
        N_input_dim = len(dataset_x[0])
        dataset_y = extract_data(User_id+'train1',Y_Data_Characteristic)
        name_x = np.array(dataset_x[0])
        name_y = np.array(dataset_y[0])
        x_dataset = dataset_x[1]
        y_dataset = dataset_y[1]
        def transpose(matrix):
            return zip(*matrix)
        x_dataset = transpose(x_dataset)
        x = pd.DataFrame(x_dataset)
        x.columns = name_x
        y_dataset = transpose(y_dataset)
        y = pd.DataFrame(y_dataset)
        y.columns = name_y
        x = x.replace('', np.NaN)
        y = y.replace('', np.NaN)
        x=x.fillna(method='pad')
        y=y.fillna(method='pad')
        x=np.array(x)
        y=np.array(y)
        x_train = x.astype('float64')
        y_train = y.astype('float64')
        y_train=np.array(y_train).reshape(y_train.shape[0],1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train = scaler.fit_transform(x_train)
        y_train = scaler.fit_transform(y_train)

        # split into train and test sets
        train_size = int(len(x_train) * 0.8)

        test_size = len(x_train) - train_size
        train_x, test_x = x_train[0:train_size, :], x_train[train_size:len(x_train), :]
        train_y, test_y = y_train[0:train_size, :], y_train[train_size:len(x_train), :]

        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
        K.clear_session()
        list = [N_First_Layer_Neurons, N_Layer_Neurons2, N_Layer_Neurons3, N_Layer_Neurons4, N_Layer_Neurons5]
        list1 = []
        for i in list:
            if i > 256:
                i = 256
                list1.append(i)
            else:
                list1.append(i)

        N_First_Layer_Neurons = list1[0]
        N_Layer_Neurons2 = list1[1]
        N_Layer_Neurons3 = list1[2]
        N_Layer_Neurons4 = list1[3]
        N_Layer_Neurons5 = list1[4]
        if N_Layers2 > 5:
            N_Layers2 = 5

        model = Sequential()
        def creat_full_lstm_model(N_Layers2, N_input_dim,N_First_Layer_Neurons,N_Layer_Neurons2, N_Layer_Neurons3,N_Layer_Neurons4,N_Layer_Neurons5):
            if N_Layers2 == 2:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N_First_Layer_Neurons, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons2, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
            if N_Layers2 ==3:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N_First_Layer_Neurons, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons2, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons3, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
            if N_Layers2==4:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N_First_Layer_Neurons, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons2, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons3, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons4, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
            if N_Layers2==5:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N_First_Layer_Neurons, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons2, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons3, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons4, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons5, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
        def create_lstm_model(N_Layers2, N_input_dim):
            if N_Layers2>5 and N_Layers2<10:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))#
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
            if N_Layers2>9 and N_Layers2 <20:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))  #
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
            if N_Layers2>19:
                model.add(LSTM(N1,input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))  #
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
        if N_Layers2>5:
            model = create_lstm_model(N_Layers2,N_input_dim)
        if N_Layers2<6:
            medel = creat_full_lstm_model(N_Layers2, N_input_dim,N_First_Layer_Neurons,N_Layer_Neurons2, N_Layer_Neurons3,N_Layer_Neurons4,N_Layer_Neurons5)
        print(model.summary())
        import keras
        #自定义优化器
        rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
        adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        adamax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #R2 评价指标
        def r_square(y_true, y_pred):
            SSR = K.mean(K.square(y_pred - K.mean(y_true)), axis=-1)
            SST = K.mean(K.square(y_true - K.mean(y_true)), axis=-1)
            return SSR / SST

        model.compile(loss='mse', optimizer=adamax,metrics=[r_square])

        train_history=model.fit(train_x, train_y, batch_size=20, nb_epoch=epoch, validation_split=0.1,verbose=1)
        model.save_weights(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/' + str(User_id) +'_Lstm_Model_weightsss.h5')
        import matplotlib.pyplot as plt

        cost = model.evaluate(test_x, test_y, batch_size = 100)
        Forecast_Result_loss = cost[1]

        Data_Characteristic_n2 = X_Data_Characteristic_n0 + Y_Data_Characteristic
        test_y_pred = model.predict(test_x)

        fig1, ax1 = plt.subplots()
        ax1.scatter(test_y, test_y_pred, edgecolors=(0, 0, 0))
        ax1.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
        '''
        my_x_ticks = np.arange(-1, 1, 0.05)
        my_y_ticks = np.arange(-1, 1, 0.03)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        '''



        ax1.set_xlabel('True')
        ax1.set_ylabel('Measured')

        plt.savefig(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/' + str(User_id)+ '_true_vs_prediction_map1.png')
    except:
        return 0
    def text_save(content, User_id, mode='a',N_Layer_Neurons2=10):
        # Try to save a list variable in txt file.
        t = os.path.abspath(os.path.join(os.getcwd(), ".."))
        newfile = t + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/' + str(User_id)+ '_LSTM_Model.csv'
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
        file = open(newfile, mode)
        for i in range(len(content)):
            file.write(str(content[i]) + '\n')
        file.close()
    text_save(("输入特征X:",X_Data_Characteristic_n0,"输出特征Y：",Y_Data_Characteristic,"预测结果得分：",Forecast_Result_loss),User_id, mode='a',N_Layer_Neurons2=10)
    return 1


#lstm预测网络
def Lstm_Model_test(User_id, X_Data_Characteristic_n0, Y_Data_Characteristic, N_Layers2, N_First_Layer_Neurons,N_Layer_Neurons2, N_Layer_Neurons3,N_Layer_Neurons4,N_Layer_Neurons5,epoch):
    try:
        N_input_dim = len(X_Data_Characteristic_n0)
        N1 = N_First_Layer_Neurons
        dataset_x = extract_data(User_id+'predict1',X_Data_Characteristic_n0)
        dataset_y = extract_data(User_id+'train1',Y_Data_Characteristic)
        name_x = np.array(dataset_x[0])
        name_y = np.array(dataset_y[0])
        x_dataset = dataset_x[1]
        y_dataset = dataset_y[1]
        def transpose(matrix):
            return zip(*matrix)
        x_dataset = transpose(x_dataset)
        x = pd.DataFrame(x_dataset)
        x.columns = name_x
        y_dataset = transpose(y_dataset)
        y = pd.DataFrame(y_dataset)
        y.columns = name_y

        x = x.replace('', np.NaN)
        y = y.replace('', np.NaN)
        x=x.fillna(method='pad')
        y=y.fillna(method='pad')
        x=np.array(x)
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train = scaler.fit_transform(x)
        y_train = scaler.fit_transform(y)

        x_train = x_train.astype('float64')

        train_x = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

        K.clear_session()
        list = [N_First_Layer_Neurons, N_Layer_Neurons2, N_Layer_Neurons3, N_Layer_Neurons4, N_Layer_Neurons5]
        list1 = []
        for i in list:
            if i > 256:
                i = 256
                list1.append(i)
            else:
                list1.append(i)

        N_First_Layer_Neurons = list1[0]
        N_Layer_Neurons2 = list1[1]
        N_Layer_Neurons3 = list1[2]
        N_Layer_Neurons4 = list1[3]
        N_Layer_Neurons5 = list1[4]
        if N_Layers2 > 5:
            N_Layers2 = 5

        model = Sequential()
        def creat_full_lstm_model(N_Layers2, N_input_dim,N_First_Layer_Neurons,N_Layer_Neurons2, N_Layer_Neurons3,N_Layer_Neurons4,N_Layer_Neurons5):
            if N_Layers2 == 2:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N_First_Layer_Neurons, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons2, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
            if N_Layers2 ==3:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N_First_Layer_Neurons, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons2, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons3, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
            if N_Layers2==4:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N_First_Layer_Neurons, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons2, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons3, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons4, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
            if N_Layers2==5:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N_First_Layer_Neurons, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons2, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons3, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons4, return_sequences=True))
                model.add(LSTM(N_Layer_Neurons5, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
        def create_lstm_model(N_Layers2, N_input_dim):
            if N_Layers2>5 and N_Layers2<10:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))#
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
            if N_Layers2>9 and N_Layers2 <20:
                model.add(LSTM(input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))  #
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
            if N_Layers2>19:
                model.add(LSTM(N1,input_dim=N_input_dim, units=N_input_dim, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))  #
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=True))
                model.add(LSTM(N1, return_sequences=False))
                model.add(Dense(output_dim=1))
                model.add(Activation('linear'))
                return model
        if N_Layers2>5:
            model = create_lstm_model(N_Layers2,N_input_dim)
        if N_Layers2<6:
            medel = creat_full_lstm_model(N_Layers2, N_input_dim,N_First_Layer_Neurons,N_Layer_Neurons2, N_Layer_Neurons3,N_Layer_Neurons4,N_Layer_Neurons5)

        import keras
        #自定义优化器
        rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
        adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        adamax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)


        model.compile(loss='mse', optimizer=adamax)
    except:
        return 2
    try:
        model.load_weights(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/' + str(User_id) +'_Lstm_Model_weightsss.h5')
        pred_y = model.predict(train_x)
    except ValueError:
        return 0
    else:
        pred_y =pred_y.reshape(pred_y.shape[0],1)
        predict_10 = scaler.inverse_transform(pred_y)

        predict_10 = pd.DataFrame(predict_10)
        predict_10.columns = ['预测结果']

        test_x = pd.DataFrame(x)
        test_x.columns = name_x

        pred_result = pd.concat((test_x, predict_10), axis=1, ignore_index=False)

        pred_result.to_csv(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/'  + User_id + '_LSTM_Model_Pred_result.csv',index=False)
        return 1

def Display_LSTM_Model_Results(User_id,X_Data_Characteristic_n0,N_First_Layer_Neurons,Y_Data_Characteristic,N_Layers2,N_Layer_Neurons2):
    try:
        if N_Layers2>5:
            N_Layers2=5
        file_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/' + str(User_id)+ '_LSTM_Model.csv'
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
    a = Lstm_Model_train('wangshuqi',['AGE'],['AGE'],2,1,10,1,1,1,5)
    #print(Display_LSTM_Model_Results('wangshuqi',['AGE'],1,['AGE'],2,10))
    b = Lstm_Model_test('wangshuqi',['AGE'],['AGE'],2,1,10,1,1,1,5)
