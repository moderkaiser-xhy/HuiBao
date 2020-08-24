
import numpy as np


from MySQL import extract_data
def display(user_id):
    train_name = extract_data(user_id+'train',['*'])
    test_name = extract_data(user_id+'predict',['*'])
    if train_name == "Error: unable to fecth data" or test_name  == "Error: unable to fecth data":
        return ("Error: unable to fecth data")
    else:

        train_name = np.array(train_name[0]).tolist()
        test_name = np.array(test_name[0]).tolist()
        c = [x for x in train_name if x in test_name]
        print(train_name)



        return (c)



