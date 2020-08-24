from flask import Flask, request, make_response,render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

CORS(app, supports_credentials=True)

@app.after_request
def af_request(resp):
    """
    #请求钩子，在所有的请求发生后执行，加入headers。
    :param resp:
    :return:
    """
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp

'''
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    res={"code":200,"data":{"context":"Hello World!"},"msg":"成功"}
    return res


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():

    res = {"code": 200, "msg": "成功"}
    print(res)

    print('请求方式为------->', request.method)
    args = request.args.get("name") or "args没有参数"
    print('args参数是------->', args)
    form = request.form.get('name') or 'form 没有参数'
    print('form参数是------->', form)

    print("fangshishiPOSt")

    userId=request.values.get("userId");
    print(userId)
    if request.files:
     f = request.files['file1']
     fileName=f.filename
     print("filename:===>>>"+fileName)
     f.save(secure_filename(f.filename))
    else:
        res = {"code": 301, "msg": "没有找到文件啊"}

    return res'''


#--------------------------权限控制------------------------------
@app.route('/', methods=['get', 'POST'])
def ac():
    return render_template('auth.html')

@app.route('/auth', methods=['get', 'POST'])
def auth():
    data =request.json
    userID = data["userID"]
    userName = data["userName"]
    token = data["token"]
    print(userID,userName,token)
    P = access(userID,userName,token)
    if P == "成功":
        return jsonify({'code': 200,'message': P})
    else:
        return jsonify({'code': 500, 'message': P})


if __name__ == '__main__':
    app.run(debug=True)
