import hashlib
import os
from datetime import date

import numpy as np
from PIL import Image


def process_uploaded_file(name:str , engine) -> str:

    img = Image.open(name)
    np_img = np.array(img)
    if np_img.shape[0] > 1000 or np_img.shape[1] > 1000:
        scale = min(np_img.shape[0], np_img.shape[1]) / 1000
        img = img.resize((int(np_img.shape[0] / scale), int(np_img.shape[1] / scale)))
    
    hs = hashlib.md5(img.tobytes()).hexdigest()

    to_save_path = f"/antikvar/flask/static/uploads/{hs}.jpg"
    if len(engine.execute(f"select * from uploads  where path = '{to_save_path}'").fetchall()) > 0:
        os.remove(name)
        return to_save_path

    img.save(to_save_path)
    today = date.today()
    engine.execute(f"INSERT INTO uploads (path, date, checked_our, checked_yandex) VALUES ('{to_save_path}', '{today.year}.{today.month}.{today.day}', 'false', 'false')")
    os.remove(name)
    return to_save_path
