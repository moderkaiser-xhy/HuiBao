
import numpy as np
import pandas as pd
from math import log
from MySQL import extract_data
import operator

def calc_ent(x):
    """
        calculate shanno ent of x
    """
    x = np.array(x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent


def show_name1(user_id):
    try:
        dataset_train = extract_data(user_id + 'train1', ['*'])
        #print(dataset_train)
        name_train = np.array(dataset_train[0])
        dataset1_train =np.array(dataset_train[1])

        def transpose(matrix):
            return zip(*matrix)

        dataset1_train = transpose(dataset1_train)
        Data_12_train = pd.DataFrame(dataset1_train)
        Data_12_train.columns = name_train
        i = 0
        list1 = []

        while i < Data_12_train.shape[1]:
            Data_1_train = Data_12_train[Data_12_train.columns[i]]
            Data_2_train = Data_1_train.to_list()
            shang = calc_ent(Data_2_train)
            #print(shang)
            list1.append(shang)
            i+=1
        jieguoshang =  dict(zip(name_train, list1))
        #return(jieguoshang)
        a = sorted(jieguoshang.items(),key=operator.itemgetter(1))
        list2 = []

        for i in a:
            b = list(i)
            list2.append(b)
        dict2 = dict(list2)
        print(dict2)
        return dict2
    except:
        return 0
if __name__ == '__main__':
    show_name1('123456')

