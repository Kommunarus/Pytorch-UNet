from flask import Flask, jsonify, request
# from flask_cors import CORS
from flask_restful import Api, Resource, reqparse
# from ml.test import run_for_restapi, prepar_models
from eval2 import run_net
import os
import uuid
import requests
import multiprocessing
import base64

app = Flask(__name__)
# CORS(app)
api = Api(app)


def run_model(path_to_img, id_file, x_y):
    out = run_net(path_to_img, id_file, x_y)
    # print(out)


def make_lab(path_to_img, x_y):

    id_file = str(uuid.uuid4())
    p = multiprocessing.Process(target=run_model, args=(path_to_img, id_file, x_y))
    p.start()

    return {'id': id_file,
            }


def getinput(link, UPLOAD_FOLDER):
    try:
        id_file = str(uuid.uuid4())
        basename = os.path.basename(link)
        name_file = id_file + '.' + basename.split('.')[1]
        fuul_name_file = os.path.join(UPLOAD_FOLDER, name_file)
        r = requests.get(link)
        with open(fuul_name_file, 'wb') as handler:
            for chunk in r.iter_content(chunk_size=255):
                if chunk:
                    handler.write(chunk)

        return fuul_name_file, id_file
    except:
        print('err')
        return None, ''


class Predict(Resource):
    def get(self):
        data = reqparse.request.args['url']
        xy = reqparse.request.args['xy']
        path_to_img, id_file = getinput(data, 'local/')
        return make_lab(path_to_img, xy)

    def post(self):
        file = request.data
        rv_string = base64.b64decode(file)

        xy = reqparse.request.args['xy']

        # path_to_img = '{}.jpg'.format(uuid.uuid4())
        path_to_img = 'local/{}.jpg'.format(uuid.uuid4())
        with open(path_to_img, 'wb') as f:
            f.write(rv_string)
        return make_lab(path_to_img, xy)

    def put(self):
        file = request.data
        xy = reqparse.request.args['xy']

        # path_to_img = '{}.jpg'.format(uuid.uuid4())
        path_to_img = 'local/{}.jpg'.format(uuid.uuid4())
        with open(path_to_img, 'wb') as f:
            f.write(file)
        return make_lab(path_to_img, xy)


class GetInfo(Resource):
    def get(self):
        data = reqparse.request.args['id']
        if os.path.exists(os.path.join('out', data+'.jpg')):
            file_cont = open(os.path.join('out', data+'.txt'))
            text = file_cont.read()

            return {'status': 'ok',
                    'contours': text}
        else:
            return {'status': 'non', }

api.add_resource(Predict, '/predict')
api.add_resource(GetInfo, '/info')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
