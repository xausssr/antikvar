import random
import ssl

import numpy as np

from flask import Flask, redirect, render_template, request
from nnmodule import InternetSearch as yas
from nnmodule import NNSearch as nns

seacher = nns()
yandex_seacher = yas()

server = Flask(__name__)

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
    a,b,c,d = seacher.search(path_to_img + images[request.values.get('id')], n_top=5, save_graphs="/antikvar/flask/static/temp/", path_to_images="/antikvar/flask/static/images_base/")
    answer = {"sample": images[request.values.get('id')]}
    for ix, i in enumerate(b):
        answer[f"i1{ix}"] = "/static/images_base/" + i + ".jpg"

    for ix, i in enumerate(d):
        answer[f"i2{ix}"] = "/static/temp/" + i.split("/")[-1]

    print(answer)
    return render_template("result_our.html", answer=answer)


@server.route('/')
@server.route('/home')
def index():
    return render_template("index.html", images=images)


@server.route('/team')
def team():
    return render_template("team.html")


@server.route('/materials')
def materials():
    return render_template("materials.html")


@server.route('/statistics')
def statistics():
    return render_template("statistics.html")


@server.route('/result_yandex', methods=["GET"])
def result_yandex():
    # test
    #answer = {"sample": images[request.values.get('id')]}
    #result = yandex_seacher.search(images=["vasilisc.ru" + answer['sample']])
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

@server.route('/model_yandex')
def model_yandex():
    return render_template("model_yandex.html", images=images)


if __name__ == '__main__':
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.load_cert_chain("certificate.crt", "cert")
    server.run(host='0.0.0.0', port=443, ssl_context=context)
