import pandas as pd
import numpy as np
def CH_DictionaryToDataFrame(data={}):
   a = data.keys()
   list1 = list(a)
   for i in range (len(list1)):
      a = []
      a.append(data[""+ list1[i]+""])
      b=np.array(a)
      c= b.ndim
      if c >1:
         x=2
         break
      else:
         x=1
   if x==1:
      data = pd.DataFrame(data,index=[0])
   else:
      data = pd.DataFrame(data)

   x_data_list = data[data.columns[0]].tolist()
   y_data_list = data[data.columns[1]].tolist()  # 提取数据特征名
   return [x_data_list, y_data_list]

if __name__=='__main__':
   data = {'User_id': '213', 'xfeature': '14558555', 'yfeature': 'yeature'}
   data1= {'asda':[1, 2, 3],'sadsad':'sads', 'sada':[3, 3, 4]}
   a = CH_DictionaryToDataFrame(data=data1)
   print(a, type(a))
