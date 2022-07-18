from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
from eval import run_net
import os
import uuid
import requests
import multiprocessing

app = Flask(__name__)
api = Api(app)


def run_model(path_to_img, id_file, path_to_img_cross):
    out = run_net(path_to_img, id_file, path_to_img_cross)


def make_lab(path_to_img, path_to_img_cross):

    id_file = str(uuid.uuid4())
    p = multiprocessing.Process(target=run_model, args=(path_to_img, id_file, path_to_img_cross))
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
        data_cross = reqparse.request.args['url_cross']
        path_to_data, id_file = getinput(data, 'local/')
        path_to_data_cross, id_file_cross = getinput(data_cross, 'local/')
        return make_lab(path_to_data, path_to_data_cross)


class GetInfo(Resource):
    def get(self):
        data = reqparse.request.args['id']
        if os.path.exists(os.path.join('out', data+'.jpg')):
                return {'status': 'ok',}
        else:
            return {'status': 'non', }

api.add_resource(Predict, '/predict')
api.add_resource(GetInfo, '/info')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=5001)
