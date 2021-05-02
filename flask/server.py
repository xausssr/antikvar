import datetime
import os
import random
import ssl

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import offline
from sqlalchemy import create_engine
from werkzeug.utils import secure_filename

from flask import Flask, flash, redirect, render_template, request
from nnmodule import InternetSearch as yas
from nnmodule import NNSearch as nns
from utills import *

UPLOAD_FOLDER = '/antikvar/flask/static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

seacher = nns()
yandex_seacher = yas()

with open("/antikvar/flask/credits") as f:
    login_str = f.readline()[:-1]
engine = create_engine(f"postgresql+psycopg2://{login_str}@localhost/parsing")

server = Flask(__name__)
server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def generate_random_images() -> dict:

    choice = np.random.randint(0, len(seacher.base), 8)
    images = {}
    for i in range(1, 7):
        images[f"i{i}"] = "/static/images_base/" + str(seacher.base.iloc[choice[i-1], 1]) + ".jpg"

    return images

images = generate_random_images()

@server.before_request
def before_request():
    if request.is_secure:
        return

    url = request.url.replace('http://', 'https://', 1)
    code = 301
    return redirect(url, code=code)

@server.route('/result_our', methods=["GET"])
def result():
    path_to_img = "/antikvar/flask/"
    # КОСТЫЛЬ ПОКА ЧТО
    if request.values.get('id') not in ["i1", "i2", "i3", "i4", "i5"]:
        print("Загруженное фигачим!")
        uploaded_name = UPLOAD_FOLDER + "/" + request.values.get('id')
        a,b,c,d = seacher.search(uploaded_name, n_top=5, save_graphs="/antikvar/flask/static/temp/", path_to_images="/antikvar/flask/static/images_base/")
        answer = {"sample": "static/uploads/" + request.values.get('id')}
    else:
        a,b,c,d = seacher.search(path_to_img + images[request.values.get('id')], n_top=5, save_graphs="/antikvar/flask/static/temp/", path_to_images="/antikvar/flask/static/images_base/")
        answer = {"sample": images[request.values.get('id')]}
    for ix, i in enumerate(b):
        answer[f"i1{ix}"] = "/static/images_base/" + i + ".jpg"

    for ix, i in enumerate(d):
        answer[f"i2{ix}"] = "/static/temp/" + i.split("/")[-1]

    return render_template("result_our.html", answer=answer)

@server.route('/', methods=['GET', 'POST'])
@server.route('/home', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            print("Лажа 1")
            return redirect("home")
        file = request.files['image']
        if file.filename == '':
            print("Лажа 2")
            return redirect("home")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(server.config['UPLOAD_FOLDER'], filename))
            uploaded_name = process_uploaded_file(os.path.join(server.config['UPLOAD_FOLDER'], filename), engine)
            return redirect(f"https://vasilisc.ru/result_our?id={uploaded_name.split('/')[-1]}")
    return render_template("index.html", images=images)

@server.route('/team')
def team():
    return render_template("team.html")

@server.route('/materials')
def materials():
    return render_template("materials.html")

@server.route('/statistics')
def statistics():
    stat = {}
    stat["obj"] = engine.execute("select max(ID) from main").fetchall()[0][0]
    stat["img"] = engine.execute("select sum(count_images) from main").fetchall()[0][0]
    for_graph = pd.read_sql_query('select * from main',con=engine)
    for_graph["date"] = pd.to_datetime(for_graph["date"])
    stat["day"] = len(for_graph[for_graph["date"].dt.date == datetime.datetime.now().date()])
    for_graph = for_graph.groupby(pd.to_datetime(for_graph.date).dt.date).agg({'count_images': 'sum'}).reset_index()
    stat["uploads"] = engine.execute("select count(*) from uploads").fetchall()[0][0]
    fig = go.Figure([go.Scatter(x=for_graph["date"], y=for_graph['count_images'])])
    fig.update_layout(
        title=f"Динамика парсинга",
        xaxis_title="Дата",
        yaxis_title="Изображений",
        font=dict(
            family="Play",
            size=12,
        ),
        )
    stat["graph"] = offline.plot(fig, include_plotlyjs=False, output_type='div')

    return render_template("statistics.html", stat=stat)

@server.route('/result_yandex', methods=["GET"])
def result_yandex():
    # test
    #answer = {"sample": images[request.values.get('id')]}
    #result = yandex_seacher.search(images=["vasilisc.ru" + answer['sample']])
    # КОСТЫЛЬ ПОКА ЧТО
    if request.values.get('id') not in ["i1", "i2", "i3", "i4", "i5"]:
        print("Загруженное фигачим!")
        uploaded_name = "static/uploads/" + request.values.get('id')
        answer = {"sample": uploaded_name}
        result = yandex_seacher.search(uploaded_name)
    else:
        answer = {"sample": "https://03.img.avito.st/image/1/9k_QIrayWqbmi5ijsGWcAzaBWqBwg1g"}
        result = yandex_seacher.search("https://03.img.avito.st/image/1/9k_QIrayWqbmi5ijsGWcAzaBWqBwg1g")

    for i in range(5):
        answer[f"url{i}"] = "home"
        answer[f"name{i}"] = "Пустой"
        answer[f"img{i}"] = "static/images/905318667-0.jpeg"

    for idx, item in enumerate(result):
        answer[f"name{idx}"] = item[1]
        answer[f"url{idx}"] = item[0]
        answer[f"img{idx}"] = item[2]

    return render_template("result_yandex.html", answer=answer)

@server.route('/model_yandex', methods=['GET', 'POST'])
def model_yandex():
    if request.method == 'POST':
        if 'image' not in request.files:
            print("Лажа 1")
            return redirect("home")
        file = request.files['image']
        if file.filename == '':
            print("Лажа 2")
            return redirect("home")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(server.config['UPLOAD_FOLDER'], filename))
            uploaded_name = process_uploaded_file(os.path.join(server.config['UPLOAD_FOLDER'], filename), engine)
            return redirect(f"https://vasilisc.ru/result_yandex?id={uploaded_name.split('/')[-1]}")
    return render_template("model_yandex.html", images=images)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.load_cert_chain("certificate.crt", "cert")
    server.run(host='0.0.0.0', port=443, ssl_context=context)
