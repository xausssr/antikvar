import random
import numpy as np

from flask import Flask, render_template, request

from nnmodule import NNSearch as nns

seacher = nns()

server = Flask(__name__)

def generate_random_images() -> dict:

    choice = np.random.randint(0, len(seacher.base), 8)
    images = {}
    for i in range(1, 9):
        images[f"i{i}"] = "/static/images_base/" + str(seacher.base.iloc[choice[i-1], 1]) + ".jpg"

    return images


images = generate_random_images()

@server.route('/result', methods=["GET"])
def result():
    print(f"Нажата карусель {request.values.get('id')}")
    path_to_img = "C:/Users/GROM/PycharmProjects/Flask"
    a,b,c,d = seacher.search(path_to_img + images[request.values.get('id')], n_top=5, save_graphs="C:/Users/GROM/PycharmProjects/Flask/static/temp/", path_to_images="C:/Users/GROM/PycharmProjects/Flask/static/images_base/", debug=False)
    print(a, "\n", b, "\n", c, "\n", d, "\n")
    print(images)
    answer = {"sample": images[request.values.get('id')]}
    for ix, i in enumerate(b):
        answer[f"i1{ix}"] = "/static/images_base/" + i + ".jpg"

    for ix, i in enumerate(d):
        answer[f"i2{ix}"] = "/static/temp/" + i.split("/")[-1]

    print(answer)
    return render_template("result.html", images=images, answer=answer)

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

if __name__ == '__main__':
    server.run(debug=True)
