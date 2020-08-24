import pandas as pd
import numpy as np
from MySQL import extract_data,create_table
import collections
class Quality_analysis():

    def __init__(self,user_id,max_m,max_v,max_c):
        self.user_id = user_id
        self.max_m = max_m
        self.max_v = max_v
        self.max_c = max_c
        #sys.setrecursionlimit(100000000)
        #print(sys.getrecursionlimit())

        self.dataset = extract_data(user_id+'train',['*'])
        self.name = np.array(self.dataset[0])
        self.dataset1 = self.dataset[1]
        def transpose(matrix):
            return zip(*matrix)
        self.dataset1 = transpose(self.dataset1)
        self.Data_123 = pd.DataFrame(self.dataset1)
        self.Data_123.columns = self.name

    #print(self.Data_123)

        self.dataset_test = extract_data(user_id+'predict', ['*'])
        self.name_test = np.array(self.dataset_test[0])
        self.dataset1_test = self.dataset_test[1]

        def transpose(matrix):
            return zip(*matrix)

        self.dataset1_test = transpose(self.dataset1_test)
        self.Data_123_test = pd.DataFrame(self.dataset1_test)
        self.Data_123_test.columns = self.name_test

    #print(pd.isnull(data).sum())
    #def __get_key(self,dict, value):
    def change_data(self,user_id):

        dataset = extract_data(user_id + 'train1', ['*'])
        name = np.array(dataset[0])
        dataset1 = dataset[1]

        def transpose(matrix):
            return zip(*matrix)

        dataset1 = transpose(dataset1)
        Data_123 = pd.DataFrame(dataset1)
        Data_123.columns = name

        # print(self.Data_123)

        dataset_test = extract_data(user_id + 'predict1', ['*'])
        name_test = np.array(dataset_test[0])
        dataset1_test = dataset_test[1]

        def transpose(matrix):
            return zip(*matrix)

        dataset1_test = transpose(dataset1_test)
        Data_123_test = pd.DataFrame(dataset1_test)
        Data_123_test.columns = name_test
        return (Data_123,Data_123_test)
        #return [k for k, v in dict.items() if v == value]
    def Num_change(self):
        Data_12_train = self.Data_123.replace('', np.nan)
        Data_12_test = self.Data_123_test.replace('', np.nan)
        Data_12_train =self.Data_123.fillna(method='pad')
        Data_12_test = self.Data_123_test.fillna(method='pad')
        i = 0
        list1=[]
        list4=[]
        while i < Data_12_train.shape[1]:
            Data_1_train = Data_12_train[Data_12_train.columns[i]]
            Data_2_train = Data_1_train.to_list()
            d_1 = collections.Counter(Data_1_train)
            d_1 = pd.DataFrame(d_1.items(), columns=['key', 'cnt'])
            list_1 = d_1['key']
            d_1=np.array(d_1)
            leng_1 = d_1.shape[0]
            list2=dict(zip(list_1,range(1,leng_1+1)))
            list3=[]
            if self.__is_number(d_1[0][0])==False:
                for j in Data_1_train:
                    for k in list2.keys():
                        if j==k:
                            list3.append(list2.get(k))
                list1.append(list3)
            else:
                list1.append(Data_2_train)

            i+=1

        #print(list1)
        Data_12_train = np.array(list1).T.tolist()
        create_table(self.user_id + 'train1', self.name, Data_12_train)
        #测试集数据
        m = 0
        list5 = []

        list6 = []
        while m < Data_12_test.shape[1]:
            Data_1_test = Data_12_test[Data_12_test.columns[m]]
            Data_2_test = Data_1_test.to_list()
            d_2 = collections.Counter(Data_1_test)
            d_2 = pd.DataFrame(d_2.items(), columns=['key', 'cnt'])
            list_2 = d_2['key']
            d_2 = np.array(d_2)
            leng_2 = d_2.shape[0]
            list7 = dict(zip(list_2, range(1, leng_2 + 1)))
            list8 = []
            if self.__is_number(d_2[0][0]) == False:
                for n in Data_2_test:
                    for p in list7.keys():
                        if n == p:
                            list8.append(list7.get(p))
                            # else:
                            # list3.append(leng)
                list5.append(list8)
            else:
                list5.append(Data_2_test)

            m += 1

        Data_121_test = np.array(list5).T.tolist()


        create_table(self.user_id + 'predict1', self.name_test, Data_121_test)
        return 1

    def miss(self):

        x = self.Data_123.replace('', 'xiaokeai')
        #x = np.array(self.Data_123)
        #y = x[x.columns[5]]
        y = x[x[self.name] == 'xiaokeai'].count() / len(x)*100

        y = dict(y)

        #print(y.values())
        list_train = []
        for i in y.keys():
            if y[i]>self.max_m:
                list_train.append(i)
        for v in list_train:
            y.pop(v)

        #print(y)
        #print()
        #print(sum(np.isnan(x)))
        return(y)
    def miss_test(self):
        x_test = self.Data_123_test.replace('', 'xiaokeai')

        y_test = x_test[x_test[self.name_test] == 'xiaokeai'].count() / len(x_test)*100
        y_test = dict(y_test)
        list_test = []
        for i in y_test.keys():
            if y_test[i] > self.max_m:
                list_test.append(i)
        for v in list_test:
            y_test.pop(v)
        #print(y_test)
        return(y_test)


    def __is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def Standard_deviation_train(self):
        tt = self.Num_change()
        if tt==1:
            Data_123=self.change_data(self.user_id)[0]
            max_v = self.max_v
            i = 0
            list=[]
            values = []
            Data_123=Data_123.replace('',0)
            while i < Data_123.shape[1]:
                Data_1 = Data_123[Data_123.columns[i]]
                Data_1 = Data_1.to_list()
                #print(Data_1)
                if self.__is_number(Data_1[0]):
                    arr_std = np.std(Data_1, ddof=1)
                    #print(sum)
                    list.append(Data_123.columns[i])
                    values.append(arr_std)
                    #print(values)
                i += 1
            Standard_deviation_train = dict(zip(list,values))
            #print(Standard_deviation_train)
            list_train = []
            for i in Standard_deviation_train.keys():
                if Standard_deviation_train[i] > self.max_v:
                    list_train.append(i)
            for v in list_train:
                Standard_deviation_train.pop(v)
            return(Standard_deviation_train)
        else:
            return 0
        # 求标准差
        #arr_std = np.std(arr, ddof=1)
    def Standard_deviation_test(self):
        tt = self.Num_change()
        if tt==1:
            Data_123 = self.change_data(self.user_id)[1]
            max_v = self.max_v
            i = 0
            list=[]
            values = []
            Data_123=Data_123.replace('',0)
            while i < Data_123.shape[1]:
                Data_1 = Data_123[Data_123.columns[i]]
                Data_1 = Data_1.to_list()
                #print(Data_1)
                if self.__is_number(Data_1[0]):
                    arr_std = np.std(Data_1, ddof=1)
                    #print(sum)
                    list.append(Data_123.columns[i])
                    values.append(arr_std)
                    #print(values)
                i += 1
            Standard_deviation_test = dict(zip(list,values))
            #print(Standard_deviation_test)
            list_test = []
            for i in Standard_deviation_test.keys():
                if Standard_deviation_test[i] > self.max_v:
                    list_test.append(i)
            for v in list_test:
                Standard_deviation_test.pop(v)
            #print(Standard_deviation_test)
            return (Standard_deviation_test)
        else:
            return 0

    def Correlation_coefficient_train(self):
        tt = self.Num_change()
        if tt == 1:
            Data_123 = self.change_data(self.user_id)[0]
            #print(Data_123)

            max_c = self.max_c

            list1=[]
            values = []
            Data_123=Data_123.replace('',np.nan)
            Data_123=Data_123.dropna(axis=0)

            Data_1234 = Data_123.corr()

            #print(Data_1234)
            max_cc_train0 =[]
            len_train = list(Data_1234.columns.values)
            for i in len_train:
                for j in len_train[len_train.index(i):]:

                    cc_train = Data_1234[i][j]
                    if cc_train == np.nan or cc_train < max_c:
                        max_cc_train0.append(cc_train)
                        if j == len_train[-1]:
                            max_cc_train = max(max_cc_train0)
                            max_cc_train0 = []
                            list1.append(i)
                            values.append(max_cc_train)

            #print(len(list1))
            #print(len(values))
            print(3)
            c_train = dict(zip(list1, values))
            return (c_train)
            #print(cc_train)

            #cc_train_name=list(Data_123.columns.values)
            #print(cc_train_name)
        else:
            return 0

        #return(Standard_deviation_train)
        # 求标准差
        #arr_std = np.std(arr, ddof=1)
    def Correlation_coefficient_test(self):
        tt = self.Num_change()
        if tt == 1:
            Data_123 = self.change_data(self.user_id)[1]
            max_c = self.max_c

            list3 = []
            values3 = []
            Data_123 = Data_123.replace('', np.nan)
            Data_123 = Data_123.dropna(axis=0)

            Data_1234 = Data_123.corr()

            max_cc_train1 = []
            len_train1 = list(Data_1234.columns.values)
            for i in len_train1:
                for j in len_train1[len_train1.index(i):]:
                    cc_train1 = Data_1234[i][j]
                    if cc_train1 == np.nan or cc_train1 < max_c:
                        max_cc_train1.append(cc_train1)
                        print('len_train',len_train1[-1])
                        if j == len_train1[-2]:
                            max_cc_train0 = max(max_cc_train1)
                            print(max_cc_train0)
                            max_cc_train1 = []
                            list3.append(i)
                            print(1)
                            print('列名：',list3)
                            values3.append(max_cc_train0)
                            print(2)
                            print('相关系数',values3)

            print(3)
            print(list3)
            print(values3)
            print(4)
            cc_test = dict(zip(list3, values3))
            return cc_test
            # print(cc_train)

            # cc_train_name=list(Data_123.columns.values)
            # print(cc_train_name)
        else:
            return 0
    def Discrete_coefficient_train(self):
        tt = self.Num_change()
        if tt==1:
            Data_123=self.change_data(self.user_id)[0]
            max_d = 100000
            i = 0
            list=[]
            values = []
            Data_123=Data_123.replace('',0)
            while i < Data_123.shape[1]:
                Data_1 = Data_123[Data_123.columns[i]]
                Data_1 = Data_1.to_list()
                #print(Data_1)
                if self.__is_number(Data_1[0]):
                    arr_std = np.std(Data_1,ddof=1)/np.mean(Data_1)
                    #print(sum)
                    list.append(Data_123.columns[i])
                    values.append(arr_std)
                    #print(values)
                i += 1
            Discrete_coefficient_train = dict(zip(list,values))
            list_train = []
            for i in Discrete_coefficient_train.keys():
                if Discrete_coefficient_train[i] > max_d:
                    list_train.append(i)
            for v in list_train:
                Discrete_coefficient_train.pop(v)
            return(Discrete_coefficient_train)
        else:
            return 0
        # 求标准差
        #arr_std = np.std(arr, ddof=1)
    def Discrete_coefficient_test(self):
        tt = self.Num_change()
        if tt==1:
            Data_123 = self.change_data(self.user_id)[1]
            max_d = 100000
            i = 0
            list=[]
            values = []
            Data_123=Data_123.replace('',0)
            while i < Data_123.shape[1]:
                Data_1 = Data_123[Data_123.columns[i]]
                Data_1 = Data_1.to_list()
                #print(Data_1)
                if self.__is_number(Data_1[0]):
                    arr_std = np.std(Data_1, ddof=1)/np.mean(Data_1)
                    #print(sum)
                    list.append(Data_123.columns[i])
                    values.append(arr_std)
                    #print(values)
                i += 1
            Discrete_coefficient_test = dict(zip(list,values))
            #print(Standard_deviation_test)
            list_test = []
            for i in Discrete_coefficient_test.keys():
                if Discrete_coefficient_test[i] > max_d:
                    list_test.append(i)
            for v in list_test:
                Discrete_coefficient_test.pop(v)
            return (Discrete_coefficient_test)
        else:
            return 0
if __name__ == '__main__':
    Quality_analysis = Quality_analysis('123456',10,1,1)
    '''
    print(Quality_analysis.miss())

    print(Quality_analysis.miss_test())
    

    
    print(Quality_analysis.mean())
    print(Quality_analysis.mean_test())
    '''
    print(Quality_analysis.Correlation_coefficient_train())
    print(Quality_analysis.Correlation_coefficient_test())
    print(Quality_analysis.Discrete_coefficient_train())
    print(Quality_analysis.Discrete_coefficient_test())
    print(Quality_analysis.miss())
    print(Quality_analysis.miss_test())
    print(Quality_analysis.Standard_deviation_test())
    print(Quality_analysis.Standard_deviation_train())