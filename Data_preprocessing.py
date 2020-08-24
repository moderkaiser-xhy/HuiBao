from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn import preprocessing
import collections
from MySQL import create_table,extract_data
class Data_preprocessing():
    def __init__(self,feature,user_id):
        self.user_id = user_id
        self.feature = feature

        self.dataset_train = extract_data(user_id+'train1',self.feature)
        print(self.dataset_train)
        self.dataset_test = extract_data(user_id+'predict1',self.feature)
        print(1)
        print(self.dataset_test)
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

    def Num_change1(self):
        try:
            Data_12_train = self.Data_12_train
            print(Data_12_train)
            Data_12_test = self.Data_12_test
            print(Data_12_test)
            i = 0
            list1 = []
            list4 = []
            while i < Data_12_train.shape[1]:
                Data_1_train = Data_12_train[Data_12_train.columns[i]]
                Data_2_train = Data_1_train.to_list()
                d_1 = collections.Counter(Data_1_train)
                d_1 = pd.DataFrame(d_1.items(), columns=['key', 'cnt'])
                list_1 = d_1['key']
                d_1 = np.array(d_1)
                leng_1 = d_1.shape[0]
                list2 = dict(zip(list_1, range(1, leng_1 + 1)))
                list3 = []
                if self.is_number(d_1[0][0]) == False:
                    for j in Data_1_train:
                        for k in list2.keys():
                            if j == k:
                                list3.append(list2.get(k))
                    list1.append(list3)
                else:
                    list1.append(Data_2_train)

                i += 1

            Data_12_train = np.array(list1).T.tolist()
            create_table(self.user_id + 'train1', self.name_train, Data_12_train)
            # 测试集数据
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

            create_table(self.user_id + 'predict1', self.name_test, Data_121_test)
            return 1
        except:
            return 0


if __name__ == '__main__':
    Data_preprocessing = Data_preprocessing(['AGE','GENDER'],'123456')
    print(Data_preprocessing.Num_change1())

    #Data_preprocessing.fill_nall(6)

    #print(Data_preprocessing.Num_change())

    #Data_preprocessing1 = Data_preprocessing_test(0, ['SERV_ID', 'PROD_INST_STATE', 'GENDER', 'AGE', 'IN_NET_DUR',


    #Data_preprocessing1.fill_nall(0)

    #print(Data_preprocessing1.Num_change())

    #print(Data_preprocessing.Data_normalization())
