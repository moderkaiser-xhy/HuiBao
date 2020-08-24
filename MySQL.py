import pymysql
import pandas as pd
import numpy as np
import sys
import configparser
import os


######################　判断是够为数字　######################


def is_number(s):
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

######################　获得数据库参数　######################
def get_cf():
    root_dir = os.path.dirname(os.path.abspath('.'))  # 获取当前文件所在目录的上一级目录，即项目所在目录E:\Crawler
    print(root_dir)
    cf = configparser.ConfigParser()
    cf.read(root_dir+"/Parameter/config.ini")  # 拼接得到config.ini文件的路径，直接使用
    host = cf.get("Mysql-Database", "host")  # 获取[Mysql-Database]中host对应的值
    user = cf.get("Mysql-Database", "user")  # 获取[Mysql-Database]中user对应的值
    password = cf.get("Mysql-Database", "password")  # 获取[Mysql-Database]中password对应的值
    database = cf.get("Mysql-Database", "database")  # 获取[Mysql-Database]中database对应的值
    charset = cf.get("Mysql-Database", "charset")  # 获取[Mysql-Database]中charset对应的值
    return(host,user,password,database,charset)



######################　将csv文件建表　######################


def excel_create_table (table_name,csv_path):        # 创建表表名时不能全为数字
    cf=get_cf()
    host=cf[0]
    user = cf[1]
    password = cf[2]
    database = cf[3]
    charset = cf[4]
    try:
        data = pd.read_csv(""+csv_path+"", encoding='utf8')   # 读取excel文件
    except:
        data = pd.read_csv("" + csv_path + "", encoding='gbk')  # 读取excel文件
    data = data.replace(np.NAN, '')

    ncols = data.shape[1]  #列数
    field = list(data) #获取特征
    value = data.values.tolist()
    db = pymysql.connect(host, user, password, database, use_unicode=True, charset=charset)   # 连接数据库
    cursor = db.cursor()
    try:
        cursor.execute("drop table  if exists "+table_name+";")
        cursor.execute("create table "+table_name+"("+field[0]+" varchar(100) NOT NULL);")#创建表
        cursor.execute("ALTER TABLE "+table_name+" CONVERT TO CHARACTER SET utf8mb4;")#输入汉字时需要将编码方式改变

        for i in range(1,ncols):
            cursor.execute("alter table "+table_name+" add "+field[i]+" varchar(100) NOT NULL")#添加特征

        val=''
        for i in range(0 , ncols):
            val = val + '%s,'

        cursor.executemany("insert into "+table_name+" values("+val[:-1]+");",value)#插入数据
        db.commit()
        return 1
    except:
        info = sys.exc_info()
        print(info[0], ":", info[1])
        return info
        db.rollback()
    finally:
        db.close()
if __name__ == "__main__":
    table_name = 'test10'
    csv_path = '/home/lkp/桌面/trians.csv'
    a=excel_create_table(table_name, csv_path)
    print(a)
    print(1232)


######################　建表　######################


def create_table (table_name,field_list,value_list):         #创建表表名时不能全为数字
    cf=get_cf()
    host=cf[0]
    user = cf[1]
    password = cf[2]
    database = cf[3]
    charset = cf[4]
    db = pymysql.connect(host, user, password, database, use_unicode=True, charset=charset)   # 连接数据库
    cursor = db.cursor()

    field = field_list#数据特征名（一维list）
    value = value_list#数据特征名对应的数据（二维list）
    n = len(field)
    try:
        cursor.execute("drop table  if exists " + table_name + ";")
        cursor.execute("create table "+table_name+"("+field[0]+" varchar(100));")#创建表
        cursor.execute("alter table " + table_name + " convert to character set utf8mb4;")#改变编码形式
        for i in range(1,n):
            cursor.execute("alter table "+table_name+" add "+field[i]+" varchar(100);")#添加特征

        val = ''
        for i in range(0, n):
            val = val + '%s,'

        cursor.executemany("insert into "+table_name+" values("+val[:-1]+");" , value)#插入数据
        db.commit()
        return 1
    except:
        info = sys.exc_info()
        print(info[0], ":", info[1])
        return "ERROR"
        db.rollback()
    finally:
        db.close()



######################　删除某一行数据　######################


def delete_IDdata (User_ID):
    cf=get_cf()
    host=cf[0]
    user = cf[1]
    password = cf[2]
    database = cf[3]
    charset = cf[4]
    db = pymysql.connect(host, user, password, database, use_unicode=True, charset=charset)   # 连接数据库
    cursor = db.cursor()

    sql = "delete from  IDtable  where ID = '%s';"%User_ID
    cursor.execute(sql)
    db.commit()  # 在改变数据库数据时必须加
    db.close()
    return 1


######################　删除某一列数据　######################


def delete_characteristic (table_name,characteristic):
    cf=get_cf()
    host=cf[0]
    user = cf[1]
    password = cf[2]
    database = cf[3]
    charset = cf[4]
    db = pymysql.connect(host, user, password, database, use_unicode=True, charset=charset)   # 连接数据库
    cursor = db.cursor()
    try:
        if len(characteristic) ==1:
            characteristic = characteristic[0]
            sql = "alter table " + table_name + "  drop column " + characteristic + ";"
        else:
            a = ''
            print(1)
            for i in range(0,len(characteristic)):
                print(i)
                a = a+" drop column "+characteristic[i]+","
            print(a)
            a=a[:-1]
            print(a)
            sql = "alter table " + table_name + "" + a + ";"

        cursor.execute(sql)
        db.commit()  # 在改变数据库数据时必须加
        return 1
    except:
        info = sys.exc_info()
        print(info[0], ":", info[1])
        return "ERROR"
        db.rollback()
    finally:
        db.close()



######################　插入数据　######################



def insert_data (table_name,field_list,value_list):
    cf=get_cf()
    host=cf[0]
    user = cf[1]
    password = cf[2]
    database = cf[3]
    charset = cf[4]
    db = pymysql.connect(host, user, password, database, use_unicode=True, charset=charset)   # 连接数据库
    cursor = db.cursor()

    field = field_list  # 数据特征名（一维list）
    value = value_list  # 数据特征名对应的数据（二维list）

    n = len(field)

    a = ''
    for i in range(0, n):
        a = a + field[i] + ','
    val = ''
    for i in range(0, n):
        val = val + '%s,'
    try:
        sql = "insert into " + table_name + "(" + a[:-1] + ") values("+  val[:-1] +")"
        cursor.executemany(sql, value)
        db.commit()
        return 1
    except:
        info = sys.exc_info()
        print(info[0], ":", info[1])
        return "ERROR"
        db.rollback()
    finally:
        db.close()





######################　从数据库中提取数据　######################



def extract_data(table_name, feature_list):
    cf=get_cf()
    host=cf[0]
    user = cf[1]
    password = cf[2]
    database = cf[3]
    charset = cf[4]
    m = ''
    db = pymysql.connect(host, user, password, database, use_unicode=True, charset=charset)   # 连接数据库
    cursor = db.cursor()
    if feature_list == ["*"]:
        sql = "SELECT * FROM " + table_name + ""
    else:
        for i in range(len(feature_list)):
            m = m + feature_list[i]
            m = m + ","
        sql = "SELECT " + m[:-1] + "  FROM " + table_name + ""
    b = []
    c = []
    try:
        cursor.execute(sql)
        results = cursor.fetchall()  # 搜取所有结果
        fields = cursor.description  # 获取MYSQL里的数据字段

        for i in range(len(fields)):
            c.append(fields[i][0])  # 将字段生成列表
        for i in range(len(c)):
            a = []
            for row in results:
                if is_number(row[i]):
                    a.append(float(row[i]))
                else:
                    a.append(row[i])
            b.append(a)  # 将数据生成多维列表
        return (c, b)
    except:
        print("Error: unable to fecth data")
        return("Error: unable to fecth data")
    finally:
        db.close()  # 关闭数据库连接




######################　删除数据库数据　######################


def drop_table (User_ID):
    cf=get_cf()
    host=cf[0]
    user = cf[1]
    password = cf[2]
    database = cf[3]
    charset = cf[4]
    db = pymysql.connect(host, user, password, database, use_unicode=True, charset=charset)   # 连接数据库
    cursor = db.cursor()

    sql = "DROP TABLE "+User_ID+"train,"+User_ID+"predict,"+User_ID+"train1,"+User_ID+"predict1;"
    print(sql)
    cursor.execute(sql)
    db.commit()  # 在改变数据库数据时必须加
    db.close()
    return 1

