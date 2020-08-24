from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn import preprocessing
import collections
from MySQL import create_table,extract_data
class Null_value_padding():
    def __init__(self,type,item,feature,user_id,any_num=None):
        self.type= type
        #print(self.type)
        self.user_id = user_id
        self.any_num = any_num
        self.item=item
        self.feature = feature

        self.dataset_train = extract_data(user_id+'train',['*'])

        self.dataset_test = extract_data(user_id+'predict',['*'])
        self.name_train = np.array(self.dataset_train[0])
        self.dataset1_train = self.dataset_train[1]
        self.name_test = np.array(self.dataset_test[0])
        self.dataset1_test = self.dataset_test[1]
        def transpose(matrix):
            return zip(*matrix)
        self.dataset1_train = transpose(self.dataset1_train)
        self.Data_12_train = pd.DataFrame(self.dataset1_train)
        self.Data_12_train.columns = self.name_train
        self.dataset1_test = transpose(self.dataset1_test)
        self.Data_12_test = pd.DataFrame(self.dataset1_test)
        self.Data_12_test.columns = self.name_test
    def Num_change(self):
        Data_12_train = self.Data_12_train
        Data_12_test = self.Data_12_test
        Data_12_train =self.Data_12_train
        Data_12_test = self.Data_12_test
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
            if self.is_number(d_1[0][0])==False:
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
        create_table(self.user_id + 'train1', self.name_train, Data_12_train)
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
            if self.is_number(d_2[0][0]) == False:
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

        #print(Data_121_test)
        create_table(self.user_id + 'predict1', self.name_test, Data_121_test)
        return 1
    def num_change222(self):
        dataset_train = extract_data(self.user_id + 'train1', ['*'])

        dataset_test = extract_data(self.user_id + 'predict1', ['*'])
        name_train = np.array(dataset_train[0])
        dataset1_train = dataset_train[1]
        name_test = np.array(dataset_test[0])
        dataset1_test = dataset_test[1]

        def transpose(matrix):
            return zip(*matrix)

        dataset1_train = transpose(dataset1_train)
        Data_12_train = pd.DataFrame(dataset1_train)
        Data_12_train.columns = name_train
        dataset1_test = transpose(dataset1_test)
        Data_12_test = pd.DataFrame(dataset1_test)
        Data_12_test.columns = name_test
        return (Data_12_train,Data_12_test)
    def mean(self):

        Data_123 = self.num_change222()[0]
        #Data_123=self.Data_12_train
        #print(Data_123)
        #print(Data_123)
        i = 0
        list1=[]
        values = []
        Data_123=Data_123.replace('',0)
        while i < Data_123.shape[1]:
            Data_1 = Data_123[Data_123.columns[i]]
            Data_1 = Data_1.to_list()
            #print(Data_1)
            if self.is_number(Data_1[0])and self.is_number(Data_1[2]) and self.is_number(Data_1[4]) and self.is_number(Data_1[6]):
                data_mean = np.mean(Data_1)
                list1.append(Data_123.columns[i])
                values.append(data_mean)
                #print(values)
            i += 1
        mean0 = dict(zip(list1,values))

        return (mean0)
    def mean_test(self):
        #Data_123=self.Data_12_test

        Data_123 = self.num_change222()[1]
        i = 0
        list1=[]
        values = []
        Data_123=Data_123.replace('',0)
        while i < Data_123.shape[1]:
            Data_1 = Data_123[Data_123.columns[i]]
            Data_1 = Data_1.to_list()
            #print(Data_1)
            if self.is_number(Data_1[0])and self.is_number(Data_1[2]) and self.is_number(Data_1[4]) and self.is_number(Data_1[6]):
                data_mean = np.mean(Data_1)
                list1.append(Data_123.columns[i])
                values.append(data_mean)
                #print(values)
            i += 1
        mean1 = dict(zip(list1,values))
        return (mean1)
    def median(self):
        #Data_123=self.Data_12_train

        Data_123 = self.num_change222()[0]
        i = 0
        list1=[]
        values = []
        Data_123=Data_123.replace('',0)
        while i < Data_123.shape[1]:
            Data_1 = Data_123[Data_123.columns[i]]
            Data_1 = Data_1.to_list()
            #print(Data_1)
            if self.is_number(Data_1[0])and self.is_number(Data_1[2]) and self.is_number(Data_1[4]) and self.is_number(Data_1[6]):
                data_median = np.median(Data_1)
                list1.append(Data_123.columns[i])
                values.append(data_median)
                    #print(values)
            i += 1
        median0 = dict(zip(list1,values))
        #print(median0)
        return (median0)
    def median_test(self):
        #Data_123=self.Data_12_test

        Data_123 = self.num_change222()[1]
        i = 0
        list1=[]
        values = []
        Data_123=Data_123.replace('',0)
        while i < Data_123.shape[1]:
            Data_1 = Data_123[Data_123.columns[i]]
            Data_1 = Data_1.to_list()
            #print(Data_1)
            if self.is_number(Data_1[0])and self.is_number(Data_1[2]) and self.is_number(Data_1[4]) and self.is_number(Data_1[6]):
                data_mean = np.median(Data_1)
                list1.append(Data_123.columns[i])
                values.append(data_mean)
                #print(values)
            i += 1
        median1 = dict(zip(list1,values))
        return (median1)
    def zhongshu(self):
        #Data_123=self.Data_12_train

        Data_123 = self.num_change222()[0]
        i = 0
        list1=[]
        values = []
        Data_123=Data_123.replace('',np.nan)
        Data_123=Data_123.dropna(axis = 0)
        while i < Data_123.shape[1]:
            Data_1 = Data_123[Data_123.columns[i]]
            Data_1 = Data_1.to_list()
            #print(Data_1)
            if self.is_number(Data_1[0])and self.is_number(Data_1[2]) and self.is_number(Data_1[4]) and self.is_number(Data_1[6]):
                data_zhongshu = np.bincount(Data_1)
                data_zhongshu = np.argmax(data_zhongshu)
                list1.append(Data_123.columns[i])
                values.append(data_zhongshu)
                    #print(values)
            i += 1
        zhongshu = dict(zip(list1,values))
        #print(zhongshu)
        return (zhongshu)
    def zhongshu_test(self):
        #Data_123=self.Data_12_test

        Data_123 = self.num_change222()[1]
        i = 0
        list1=[]
        values = []
        Data_123 = Data_123.replace('', np.nan)
        Data_123 = Data_123.dropna(axis=0)
        while i < Data_123.shape[1]:
            Data_1 = Data_123[Data_123.columns[i]]
            Data_1 = Data_1.to_list()
            #print(Data_1)
            if self.is_number(Data_1[0])and self.is_number(Data_1[2]) and self.is_number(Data_1[4]) and self.is_number(Data_1[6]):
                data_zhongshu = np.bincount(Data_1)
                data_zhongshu = np.argmax(data_zhongshu)
                list1.append(Data_123.columns[i])
                values.append(data_zhongshu)
                #print(values)
            i += 1
        zhongshu1 = dict(zip(list1,values))
        return (zhongshu1)
    def fill_nall(self):
        try:
            shuruzhi = dict(zip(self.feature,self.item))

            #print(shuruzhi)
            list1=[]
            list2=[]
            values = []
            values1 = []
            if self.type==1:
                self.Num_change()
                Data_12_train = self.num_change222()[0]
                #Data_12_train = self.Data_12_train
                #print(Data_12_train)
                name_train = self.name_train
                #print(name_train)
                for i in name_train:
                    #print(i)
                    if i in self.feature:
                        list1.append(i)
                        values.append(self.fill_train(i,shuruzhi[i]))
                    else:
                        Data_12_train[i] = Data_12_train[i].replace('', np.nan)
                        Data_12_train[i] = Data_12_train[i].fillna(method='pad')
                        #print(type(Data_12_train[i]))
                        a1 = Data_12_train[i].tolist()
                        list1.append(i)
                        values.append(a1)
            #print(len(list1))
            #print(values)
                v11 = np.array(values).T.tolist()
                #print(v11)
                #print(v11)
                try:
                    create_table(self.user_id + 'train1', list1, v11)
                except:
                    print(111)
            if self.type == 2:
                #Data_12_test = self.Data_12_test
                self.Num_change()
                Data_12_test = self.num_change222()[1]
                name_test = self.name_test
                for i in name_test:
                    if i in self.feature:
                        list2.append(i)
                        values1.append(self.fill_test(i,shuruzhi[i]))
                    else:
                        Data_12_test[i] = Data_12_test[i].replace('', np.nan)
                        Data_12_test[i] = Data_12_test[i].fillna(method='pad')
                        b1 = Data_12_test[i].tolist()
                        list2.append(i)
                        values1.append(b1)
            #print(list1)
            #print(values)
                v12 = np.array(values1).T.tolist()
                #print(v12)
                create_table(self.user_id + 'predict1', list2, v12)
            return 1
        except:
            return 0
    def fill_train(self,feature_train,item_train):
        mean_train = self.mean()
        median_train = self.median()
        zhongshu_train = self.zhongshu()
        Data_12_train = self.Data_12_train
        teshuzhi = dict(zip(self.feature, self.any_num))
        if item_train==1:
            Data_12_train[feature_train] = Data_12_train[feature_train].replace('', np.nan)
            Data_12_train[feature_train] = Data_12_train[feature_train].fillna(method='pad')
            a = Data_12_train[feature_train].tolist()
            #print(type(a))
            return (a)
        '''
        else:
            Data_12_train[feature_train] = Data_12_train[feature_train].replace('', np.nan)
            Data_12_train[feature_train] = Data_12_train[feature_train].fillna(method='pad')
            return (Data_12_train[feature_train])
        '''
        if item_train==2:
            Data_12_train[feature_train] = Data_12_train[feature_train].replace('', np.nan)
            Data_12_train[feature_train] = Data_12_train[feature_train].fillna(method='bfill')
            a = Data_12_train[feature_train].tolist()
            return (a)

        if item_train == 3:
            Data_12_train[feature_train] = Data_12_train[feature_train].replace('', np.nan)
            Data_12_train[feature_train] = Data_12_train[feature_train].fillna(method='pad')
            a = Data_12_train[feature_train].tolist()
            return (a)

        if item_train == 4:
            Data_12_train[feature_train] = Data_12_train[feature_train].replace('', mean_train[feature_train])
            a = Data_12_train[feature_train].tolist()
            return (a)

        if item_train == 5:
            Data_12_train[feature_train] = Data_12_train[feature_train].replace('', median_train[feature_train])
            a = Data_12_train[feature_train].tolist()
            return (a)

        if item_train == 6:
            Data_12_train[feature_train] = Data_12_train[feature_train].replace('', zhongshu_train[feature_train])
            a = Data_12_train[feature_train].tolist()
            return (a)

        if item_train == 7:
            Data_12_train[feature_train] = Data_12_train[feature_train].replace('', teshuzhi[feature_train])
            a = Data_12_train[feature_train].tolist()
            return (a)



    def fill_test(self,feature_test,item_test):
        mean_test = self.mean_test()
        median_test = self.mean_test()
        zhongshu_test = self.zhongshu_test()
        Data_12_test = self.Data_12_test
        teshuzhi = dict(zip(self.feature,self.any_num))
        if item_test==1:
            Data_12_test[feature_test] = Data_12_test[feature_test].replace('', np.nan)
            Data_12_test[feature_test] = Data_12_test[feature_test].fillna(method='pad')
            b = Data_12_test[feature_test].tolist()
            return (b)
        '''
        else:
            Data_12_test[feature_test] = Data_12_test[feature_test].replace('', np.nan)
            Data_12_test[feature_test] = Data_12_test[feature_test].fillna(method='pad')
            return (Data_12_test[feature_test])
        '''
        if item_test==2:
            Data_12_test[feature_test] = Data_12_test[feature_test].replace('', np.nan)
            Data_12_test[feature_test] = Data_12_test[feature_test].fillna(method='bfill')
            b = Data_12_test[feature_test].tolist()
            return (b)

        if item_test == 3:
            Data_12_test[feature_test] = Data_12_test[feature_test].replace('', np.nan)
            Data_12_test[feature_test] = Data_12_test[feature_test].fillna(method='pad')
            b = Data_12_test[feature_test].tolist()
            return (b)

        if item_test == 4:
            Data_12_test[feature_test] = Data_12_test[feature_test].replace('', mean_test[feature_test])
            b = Data_12_test[feature_test].tolist()
            return (b)

        if item_test == 5:
            Data_12_test[feature_test] = Data_12_test[feature_test].replace('', median_test[feature_test])
            b = Data_12_test[feature_test].tolist()
            return (b)

        if item_test == 6:
            Data_12_test[feature_test] = Data_12_test[feature_test].replace('', zhongshu_test[feature_test])
            b = Data_12_test[feature_test].tolist()
            return (b)

        if item_test == 7:
            Data_12_test[feature_test] = Data_12_test[feature_test].replace('', teshuzhi[feature_test])
            b = Data_12_test[feature_test].tolist()
            return (b)






    def is_number(self,s):
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





if __name__ == '__main__':
    Null_value_padding = Null_value_padding(1,[7,7], ['AGE','RM_MODE_CD'],'123456',[0,200])

    #print(Null_value_padding.fill_nall())
    print(Null_value_padding.Num_change())

    #Data_preprocessing1 = Data_preprocessing_test(0, ['SERV_ID', 'PROD_INST_STATE', 'GENDER', 'AGE', 'IN_NET_DUR',


    #Data_preprocessing1.fill_nall(0)

    #print(Data_preprocessing1.Num_change())

    #print(Data_preprocessing.Data_normalization())
