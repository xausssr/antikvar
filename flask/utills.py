import hashlib
import os
from datetime import date

import numpy as np
from PIL import Image


def process_uploaded_file(name:str , engine) -> str:

    img = Image.open(name).convert('RGB')

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
