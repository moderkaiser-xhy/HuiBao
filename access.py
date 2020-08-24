import time
import os
import hashlib

def access(userID,userName,token):
    localtime = time.localtime(time.time())
    a = str(localtime.tm_year) +"-"+ str(localtime.tm_mon) +"-"+ str(localtime.tm_mday)
    b = str(localtime.tm_hour) + ":" + str(localtime.tm_min) + ":" + str(localtime.tm_sec)
    path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/Log/"+a+".txt")
    c = a +" "+ b
    d = str(localtime.tm_year).rjust(4,'0') + str(localtime.tm_mon).rjust(2,'0') + str(localtime.tm_mday).rjust(2,'0')

    hl = hashlib.md5()
    strs = "kdhbdsj" + d + userID
    hl.update(strs.encode("utf8"))
    token1 = hl.hexdigest()
    print('MD5加密前为 ：', strs)
    print('MD5加密后为 ：', token1)
    if token==token1:
        access = "成功"
    else:
        access = "失败"

    if os.path.isfile(path):
        with open(path, "a", encoding='utf-8') as f:
            f.write(str(c).rjust(20,' '))
            f.write(str(userName).rjust(20, ' '))
            f.write(str(userID).rjust(20,' '))
            f.write(str(access).rjust(20,' ')+"\n")
            f.close()

    else:
        with open(path, "a", encoding='utf-8') as f:
            f.write("时间".rjust(18,' '))
            f.write("用户名".rjust(18, ' '))
            f.write("用户ID".rjust(18,' '))
            f.write("验证结果".rjust(18,' ')+"\n")
            f.write(str(c).rjust(20,' '))
            f.write(str(userName).rjust(20, ' '))
            f.write(str(userID).rjust(20,' '))
            f.write(str(access).rjust(20,' ') + "\n")
            f.close()


    return access

if __name__ == '__main__':
    userID = '12345678'
    userName=1
    token = 'aecf513bfeca1e24d70edb27e8de709c'
    n = access(userID, userName, token)
    print(n)