import keras
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution1D, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import SGD
import pandas as pd
import numpy as np
import os
import tensorflow as tf
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from MySQL import extract_data
def  Neural_Network_Model_train(User_id,feature,N_Last_Layer_Neurons,N_Layers1,N_First_Layer_Neurons,N_Layer_Neurons2, N_Layer_Neurons3, N_Layer_Neurons4, N_Layer_Neurons5,epoch):
    N1 = N_First_Layer_Neurons
    try:
        dataset_x = extract_data(User_id+'train1',feature)
        dataset_y = extract_data(User_id+'train1',N_Last_Layer_Neurons)
        name_x = np.array(dataset_x[0])
        leng_x = len(name_x)
        name_y = np.array(dataset_y[0])
        x_dataset0 = dataset_x[1]
        y_dataset0 = dataset_y[1]
        def transpose(matrix):
            return zip(*matrix)

        x_dataset = transpose(x_dataset0)
        x = pd.DataFrame(x_dataset)
        x.columns = name_x
        y_dataset = transpose(y_dataset0)
        y = pd.DataFrame(y_dataset)
        y.columns = name_y
    except:
        return 2
    try:
        x = x.replace('', np.NaN)
        y = y.replace('', np.NaN)
        imp = Imputer(missing_values='NaN',strategy='mean',verbose=0)
        imp.fit(x)
        x=imp.transform(x)
        y= y.fillna(method='pad')

        x=np.array(x)

        y=np.array(y).tolist()
        set1 = [item for sublist in y for item in sublist]

        N_last = len(set(set1))

        scaler = MinMaxScaler(feature_range=(0, 1))
        x = scaler.fit_transform(x)

        x = x.reshape(x.shape[0],1,leng_x)

        encoder = LabelEncoder()
        encoder.fit(y)
        y_train_dataset = encoder.transform(y)
        if np.any(y_train_dataset):
            y = np_utils.to_categorical(y_train_dataset, num_classes=N_last)
        x_train = x.astype('float64')
        y_train = y.astype('float64')
    except:
        return 3
    train_x, test_x, train_y, test_y = train_test_split(x_train, y_train,train_size=0.8, random_state=40)


    '''
    print(train_x.shape)
    print(train_x)
    print(train_y.shape)
    print(train_y)

    print(test_x.shape)
    print(test_x)
    print(test_y.shape)
    print(test_y)
    '''
    from keras.models import Sequential
    from keras import regularizers

    keras.backend.clear_session()
    #判断输入层神经元个数是否超过最大值256
    list = [N_First_Layer_Neurons, N_Layer_Neurons2,N_Layer_Neurons3,N_Layer_Neurons4,N_Layer_Neurons5]
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
    if N_Layers1>5:
        N_Layers1 = 5

    model = Sequential()
    def create_full_nnw_model(N_Layers1, N_First_Layer_Neurons, N_Layer_Neurons2, N_Layer_Neurons3,
                              N_Layer_Neurons4, N_Layer_Neurons5):

        if N_Layers1 == 2:
            model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
            model.add(Activation('relu'))
            model.add(Convolution1D(N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2 * N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2 * N1, 1))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(N_Layer_Neurons2, activation='relu'))
            # model.add(Dropout(0.2))
            #model.add(Dense(4 * N1, activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dropout(0.5))
            # model.add(Dense(64,  activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(64,  activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(N_last, activation='softmax'))
            return model
        if N_Layers1 == 3:
            model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
            model.add(Activation('relu'))
            model.add(Convolution1D(N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2 * N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2 * N1, 1))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(N_Layer_Neurons2, activation='relu'))
            model.add(Dense(N_Layer_Neurons3, activation='relu'))
            # model.add(Dropout(0.2))
            #model.add(Dense(4 * N1, activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dropout(0.5))
            # model.add(Dense(64,  activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(64,  activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(N_last, activation='softmax'))
            return model
        if N_Layers1 == 4:
            model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
            model.add(Activation('relu'))
            model.add(Convolution1D(N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2 * N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2 * N1, 1))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(N_Layer_Neurons2, activation='relu'))
            model.add(Dense(N_Layer_Neurons3, activation='relu'))
            model.add(Dense(N_Layer_Neurons4, activation='relu'))
            model.add(Dense(N_last, activation='softmax'))
            return model
        if N_Layers1 == 5:
            model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
            model.add(Activation('relu'))
            model.add(Convolution1D(N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2 * N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2 * N1, 1))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(N_Layer_Neurons2, activation='relu'))
            model.add(Dense(N_Layer_Neurons3, activation='relu'))
            model.add(Dense(N_Layer_Neurons4, activation='relu'))
            model.add(Dense(N_Layer_Neurons5, activation='relu'))
            model.add(Dense(N_last, activation='softmax'))
            return model
    def create_model(N_Layers1,N1):
        if N_Layers1>5 and N_Layers1<10:
            model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
            model.add(Activation('relu'))
            model.add(Convolution1D(N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2 * N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2 * N1, 1))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(4 * N1, activation='relu'))
            # model.add(Dropout(0.2))
            model.add(Dense(4 * N1, activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dropout(0.5))
            # model.add(Dense(64,  activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(64,  activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(N_last, activation='softmax'))
            return model
        if N_Layers1>9 and  N_Layers1<20:
            model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
            model.add(Activation('relu'))
            model.add(Convolution1D(N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2*N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2*N1, 1))
            model.add(Activation('relu'))
            #model.add(Dropout(0.2))
            '''
            model.add(Convolution1D(32, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(32, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(64, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(64, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(128, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(128, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(256, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(128, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(128, 1))
            model.add(Activation('relu'))
            '''
            model.add(Flatten())
            #model.add(Dense(256, input_dim=(4),activation='relu'))
            model.add(Dense(4*N1, activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(4*N1, activation='relu'))
            #model.add(Dropout(0.2))
            #model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            #model.add(Dropout(0.2))
            #model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            #model.add(Dropout(0.2))
            #model.add(Dropout(0.5))
            #model.add(Dense(64,  activation='relu'))
            #model.add(Dropout(0.2))
            #model.add(Dense(64,  activation='relu'))
            model.add(Dropout(0.2))
            #model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            model.add(Dense(8*N1, activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(8*N1,activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(8*N1, activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(8*N1, activation='relu'))
            model.add(Dropout(0.2))

            model.add(Dense(16*N1, activation='relu'))
            #model.add(Dropout(0.3))
            model.add(Dense(16*N1,activation='relu'))
            #model.add(Dropout(0.3))
            model.add(Dense(16*N1, activation='relu'))
            #model.add(Dropout(0.3))
            model.add(Dense(16*N1, activation='relu'))
            model.add(Dropout(0.3))
            #model.add(Dense(2048, activation='relu'))
            #model.add(Dense(64, activation='relu'))

            model.add(Dense(N_last, activation='softmax'))
            return model
        if N_Layers1>19 :
            model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
            model.add(Activation('relu'))
            model.add(Convolution1D(N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2*N1, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(2*N1, 1))
            model.add(Activation('relu'))
            #model.add(Dropout(0.2))
            '''
            model.add(Convolution1D(32, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(32, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(64, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(64, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(128, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(128, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(256, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(128, 1))
            model.add(Activation('relu'))
            model.add(Convolution1D(128, 1))
            model.add(Activation('relu'))
            '''
            model.add(Flatten())
            #model.add(Dense(256, input_dim=(4),activation='relu'))
            model.add(Dense(4*N1, activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(4*N1, activation='relu'))
            #model.add(Dropout(0.2))
            #model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            #model.add(Dropout(0.2))
            #model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            #model.add(Dropout(0.2))
            #model.add(Dropout(0.5))
            #model.add(Dense(64,  activation='relu'))
            #model.add(Dropout(0.2))
            #model.add(Dense(64,  activation='relu'))
            model.add(Dropout(0.2))
            #model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            model.add(Dense(8*N1, activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(8*N1,activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(8*N1, activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(8*N1, activation='relu'))
            model.add(Dropout(0.2))

            model.add(Dense(16*N1, activation='relu'))
            #model.add(Dropout(0.3))
            model.add(Dense(16*N1,activation='relu'))
            #model.add(Dropout(0.3))
            model.add(Dense(16*N1, activation='relu'))
            #model.add(Dropout(0.3))
            model.add(Dense(16*N1, activation='relu'))
            model.add(Dropout(0.3))
            #model.add(Dense(2048, activation='relu'))
            #model.add(Dense(64, activation='relu'))
            model.add(Dense(32 * N1, activation='relu'))
            # model.add(Dropout(0.2))
            model.add(Dense(32 * N1, activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dropout(0.5))
            # model.add(Dense(64,  activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(64,  activation='relu'))
            model.add(Dropout(0.2))
            # model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
            model.add(Dense(64 * N1, activation='relu'))
            # model.add(Dropout(0.2))
            model.add(Dense(64 * N1, activation='relu'))
            # model.add(Dropout(0.2))
            model.add(Dense(64 * N1, activation='relu'))
            # model.add(Dropout(0.2))
            model.add(Dense(64 * N1, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(N_last, activation='softmax'))
            return model
    #自定义优化器
    if N_Layers1>5:
        model = create_model(N_Layers1, N1)
    if N_Layers1<6:
        model = create_full_nnw_model(N_Layers1, N_First_Layer_Neurons, N_Layer_Neurons2, N_Layer_Neurons3,N_Layer_Neurons4, N_Layer_Neurons5)
    graph = tf.get_default_graph()
    model.summary()
    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
    adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    adamax = keras.optimizers.Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])



    train_history=model.fit(x=train_x,y=train_y,validation_split=0.2,epochs=epoch,batch_size=32,verbose=1)#,callbacks=[Checkpoint])

    model.save_weights(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/' + str(User_id) +'_Neural_Network_Model_weightsss.h5')
    '''
    import matplotlib.pyplot as plt
    def show_train_history(train_history,train,validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train','validation'],loc='upper left')
        plt.show()
    show_train_history(train_history, 'loss', 'val_loss')'''
    print('Testing')

    cost = model.evaluate(test_x, test_y, batch_size = 100)
    Classification_Result_loss = cost[0]
    Classification_Result_accuracy = cost[1]
    Data_Characteristic_n1 = feature+N_Last_Layer_Neurons
    #print(cost)
    #test_y_pred = model.predict(test_x)
    #print(test_y_pred)
    #print(test_y.shape)
    #test_y_pred=np.ravel(test_y_pred)
    #print(test_y_pred)
    #print(test_y_pred.shape)
    #init_lables = encoder.inverse_transform(test_y_pred)
    #print(test_y_pred.shape)
    #index_max = np.argmax(test_y_pred, axis=1)
    #max = test_y[range(test_y.shape[0]), index_max]
    #print(max)
    def text_save(content, User_id,N_Layer_Neurons2,mode = 'a'):
        # Try to save a list variable in txt file.
        t = os.path.abspath(os.path.join(os.getcwd(), ".."))
        newfile =t + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/' + str(User_id)+'_Neural_Network_Model.csv'
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
    text_save(("输入特征X:",feature,"输出特征Y：",N_Last_Layer_Neurons,"分类结果误差：",Classification_Result_loss,"分类结果准确度：",Classification_Result_accuracy),User_id,N_Layer_Neurons2,mode = 'a')
    #return (Data_Characteristic_n1, Classification_Result_loss, Classification_Result_accuracy)
    return 1

#预测网络：
def  Neural_Network_Model_test(User_id,ID,feature,N_Last_Layer_Neurons,N_Layers1,N_First_Layer_Neurons,N_Layer_Neurons2,N_Layer_Neurons3,N_Layer_Neurons4,N_Layer_Neurons5,epoch):
    # self.file_path = file_path
    try:
        N1 = N_First_Layer_Neurons
        #x从测试集的数据表提取值
        dataset_x = extract_data(User_id+'predict1', feature)
        #y从训练集的数据表提取值
        dataset_y = extract_data(User_id+'train1', N_Last_Layer_Neurons)

        dataset_ID = extract_data(User_id+'predict1', ID)
        name_x = np.array(dataset_x[0])
        leng_x = len(name_x)
        name_y = np.array(dataset_y[0])
        name_ID = np.array(dataset_ID[0])
        x_dataset0 = dataset_x[1]
        y_dataset0 = dataset_y[1]
        ID_dataset0 = dataset_ID[1]

        # print(y_dataset0)
        def transpose(matrix):
            return zip(*matrix)

        x_dataset = transpose(x_dataset0)
        x = pd.DataFrame(x_dataset)
        x.columns = name_x
        y_dataset = transpose(y_dataset0)
        y = pd.DataFrame(y_dataset)
        y.columns = name_y
        ID_dataset = transpose(ID_dataset0)
        ID_dataset2 = pd.DataFrame(ID_dataset)
        ID_dataset2.columns = name_ID
        x = x.replace('', np.NaN)
        y = y.replace('', np.NaN)
        imp = Imputer(missing_values='NaN', strategy='mean', verbose=0)
        imp.fit(x)
        x = imp.transform(x)
        y = y.fillna(method='pad')

        x = np.array(x)

        y = np.array(y).tolist()
        set1 = [item for sublist in y for item in sublist]

        N_last = len(set(set1))

        scaler = MinMaxScaler(feature_range=(0, 1))
        x_1 = scaler.fit_transform(x)
        x = x_1.reshape(x.shape[0], 1, leng_x)

        #encoder = LabelEncoder()
        #encoder.fit(y)
        #y_train_dataset = encoder.transform(y)
        #if np.any(y_train_dataset):
            #y = np_utils.to_categorical(y_train_dataset, num_classes=N_last)

        x_train = x.astype('float64')
        #y_train = y.astype('float64')

        #train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, train_size=0.8, random_state=40)
        '''
        print(train_x.shape)
        print(train_x)
        print(train_y.shape)
        print(train_y)
    
        print(test_x.shape)
        print(test_x)
        print(test_y.shape)
        print(test_y)
        '''
        from keras.models import Sequential
        from keras import regularizers

        keras.backend.clear_session()
        # 判断输入层神经元个数是否超过最大值256
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
        if N_Layers1 > 5:
            N_Layers1 = 5

        model = Sequential()

        def create_full_nnw_model(N_Layers1, N_First_Layer_Neurons, N_Layer_Neurons2, N_Layer_Neurons3,
                                  N_Layer_Neurons4, N_Layer_Neurons5):

            if N_Layers1 == 2:
                model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
                model.add(Activation('relu'))
                model.add(Convolution1D(N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Flatten())
                model.add(Dense(N_Layer_Neurons2, activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(4 * N1, activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dropout(0.5))
                # model.add(Dense(64,  activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(64,  activation='relu'))
                # model.add(Dropout(0.2))
                model.add(Dense(N_last, activation='softmax'))
                return model
            if N_Layers1 == 3:
                model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
                model.add(Activation('relu'))
                model.add(Convolution1D(N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Flatten())
                model.add(Dense(N_Layer_Neurons2, activation='relu'))
                model.add(Dense(N_Layer_Neurons3, activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(4 * N1, activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dropout(0.5))
                # model.add(Dense(64,  activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(64,  activation='relu'))
                # model.add(Dropout(0.2))
                model.add(Dense(N_last, activation='softmax'))
                return model
            if N_Layers1 == 4:
                model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
                model.add(Activation('relu'))
                model.add(Convolution1D(N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Flatten())
                model.add(Dense(N_Layer_Neurons2, activation='relu'))
                model.add(Dense(N_Layer_Neurons3, activation='relu'))
                model.add(Dense(N_Layer_Neurons4, activation='relu'))
                model.add(Dense(N_last, activation='softmax'))
                return model
            if N_Layers1 == 5:
                model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
                model.add(Activation('relu'))
                model.add(Convolution1D(N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Flatten())
                model.add(Dense(N_Layer_Neurons2, activation='relu'))
                model.add(Dense(N_Layer_Neurons3, activation='relu'))
                model.add(Dense(N_Layer_Neurons4, activation='relu'))
                model.add(Dense(N_Layer_Neurons5, activation='relu'))
                model.add(Dense(N_last, activation='softmax'))
                return model

        def create_model(N_Layers1, N1):
            if N_Layers1 > 5 and N_Layers1 < 10:
                model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
                model.add(Activation('relu'))
                model.add(Convolution1D(N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Flatten())
                model.add(Dense(4 * N1, activation='relu'))
                # model.add(Dropout(0.2))
                model.add(Dense(4 * N1, activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dropout(0.5))
                # model.add(Dense(64,  activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(64,  activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(N_last, activation='softmax'))
                return model
            if N_Layers1 > 9 and N_Layers1 < 20:
                model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
                model.add(Activation('relu'))
                model.add(Convolution1D(N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                # model.add(Dropout(0.2))
                '''
                model.add(Convolution1D(32, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(32, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(64, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(64, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(128, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(128, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(256, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(128, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(128, 1))
                model.add(Activation('relu'))
                '''
                model.add(Flatten())
                # model.add(Dense(256, input_dim=(4),activation='relu'))
                model.add(Dense(4 * N1, activation='relu'))
                # model.add(Dropout(0.2))
                model.add(Dense(4 * N1, activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dropout(0.5))
                # model.add(Dense(64,  activation='relu'))
                # model.add(Dropout(0.2))
                # model.add(Dense(64,  activation='relu'))
                model.add(Dropout(0.2))
                # model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
                model.add(Dense(8 * N1, activation='relu'))
                # model.add(Dropout(0.2))
                model.add(Dense(8 * N1, activation='relu'))
                # model.add(Dropout(0.2))
                model.add(Dense(8 * N1, activation='relu'))
                # model.add(Dropout(0.2))
                model.add(Dense(8 * N1, activation='relu'))
                model.add(Dropout(0.2))

                model.add(Dense(16 * N1, activation='relu'))
                # model.add(Dropout(0.3))
                model.add(Dense(16 * N1, activation='relu'))
                # model.add(Dropout(0.3))
                model.add(Dense(16 * N1, activation='relu'))
                # model.add(Dropout(0.3))
                model.add(Dense(16 * N1, activation='relu'))
                model.add(Dropout(0.3))


                model.add(Dense(N_last, activation='softmax'))
                return model
            if N_Layers1 > 19:
                model.add(Convolution1D(nb_filter=N1, filter_length=1, input_shape=(1, leng_x)))
                model.add(Activation('relu'))
                model.add(Convolution1D(N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))
                model.add(Convolution1D(2 * N1, 1))
                model.add(Activation('relu'))

                model.add(Flatten())
                model.add(Dense(4 * N1, activation='relu'))
                model.add(Dense(4 * N1, activation='relu'))

                model.add(Dropout(0.2))
                model.add(Dense(8 * N1, activation='relu'))
                model.add(Dense(8 * N1, activation='relu'))
                model.add(Dense(8 * N1, activation='relu'))
                model.add(Dense(8 * N1, activation='relu'))
                model.add(Dropout(0.2))

                model.add(Dense(16 * N1, activation='relu'))
                model.add(Dense(16 * N1, activation='relu'))
                model.add(Dense(16 * N1, activation='relu'))
                model.add(Dense(16 * N1, activation='relu'))
                model.add(Dropout(0.3))

                model.add(Dense(32 * N1, activation='relu'))
                model.add(Dense(32 * N1, activation='relu'))

                model.add(Dropout(0.2))
                model.add(Dense(64 * N1, activation='relu'))
                model.add(Dense(64 * N1, activation='relu'))
                model.add(Dense(64 * N1, activation='relu'))
                model.add(Dense(64 * N1, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(N_last, activation='softmax'))
                return model

        # 自定义优化器
        if N_Layers1 > 5:
            model = create_model(N_Layers1, N1)
        if N_Layers1 < 6:
            model = create_full_nnw_model(N_Layers1, N_First_Layer_Neurons, N_Layer_Neurons2, N_Layer_Neurons3,
                                          N_Layer_Neurons4, N_Layer_Neurons5)
        graph = tf.get_default_graph()
        model.summary()
        rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
        adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        adamax = keras.optimizers.Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    except:
        return 2

    try:
        model.load_weights(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/' + 'UserFiles' + '/' + str(User_id) + '/' + 'Result' + '/' + str(User_id) + '_Neural_Network_Model_weightsss.h5')
        pred_y = model.predict(x_train)
    except ValueError:
        return 0
    else:
        list_label = np.argmax(pred_y, axis=1)

        predict_10 = pd.DataFrame(list_label)
        predict_10.columns = name_y
        test_x = scaler.inverse_transform(x_1)
        test_x = pd.DataFrame(test_x)
        test_x.columns =   name_x
        print(ID_dataset2)
        pred_result = pd.concat((ID_dataset2, predict_10), axis=1, ignore_index=False)
        #print(pred_result)
        pred_result.to_csv(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/' + str(User_id)+'_Neural_Network_Model_Pred_result.csv',index=False)
        return 1
def Display_NNW_Model_Results(User_id,feature,N_Last_Layer_Neurons,N_Layers1,N_First_Layer_Neurons,N_Layer_Neurons2):

    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        if N_Layers1 > 5:
            N_Layers1 = 5
        file_path =os.path.abspath(os.path.join(os.getcwd(), "..")) + '/'+ 'UserFiles' +'/'+  str(User_id) +'/'+ 'Result' +'/' + str(User_id)+'_Neural_Network_Model.csv'
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
    a = Neural_Network_Model_train('123456',['AGE','IN_NET_DUR'],['GENDER'],2,16,5,45,12,54,5)
    #print(Display_NNW_Model_Results('wangshuqi',['AGE','IN_NET_DUR'],['GENDER'],2,16,5))
    c = Neural_Network_Model_test('123456',['SERV_ID'],['AGE','IN_NET_DUR'],['GENDER'],2,16,5,45,12,54,5)


