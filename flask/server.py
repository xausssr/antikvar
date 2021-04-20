from flask import Flask

app = Flask(__name__)

@app.route('/')
def main_page():
    return 'Ну вот, теперь еще и вэб разрабатывать :('
