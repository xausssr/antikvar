import hashlib
import os
import shutil
import time

import numpy as np
import pandas as pd
import requests
from matplotlib import pylab as plt
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import (ElementClickInterceptedException,
                                        NoSuchElementException,
                                        StaleElementReferenceException)
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from tensorflow import keras as keras

PATH = os.path.realpath(__file__).split("nnmodule")[0]

class NNSearch():

    def __init__(self):
        self.flat_base = np.load(PATH + "res/flat_x_wo_aug.npy", allow_pickle=True)
        
        # Загрузка ResNet50V2
        self.resnet = keras.models.Sequential()
        self.resnet.add(keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
            pooling=True,
        ))
        self.resnet.add(keras.layers.AveragePooling2D(pool_size=(7,7)))
        self.resnet.add(keras.layers.Flatten())
        self.resnet.compile()

        #Загрузка нашей модели
        self.model = keras.models.load_model(PATH + "res/seacher_v.02")

        self.base = pd.read_csv(PATH + "base.csv")
        info_id = self.base.groupby("id").count()
        info_id = info_id[info_id["image"] > 2].index
        self.base = self.base[self.base['id'].isin(info_id)]
        self.base = self.base[~self.base["image"].str.contains("youtube")]

    def search(self, image_path:str, n_top:int=5, save_graphs:str="C:/temp/", path_to_images:str="C:/Devs/git/antikvar/NeuralNetworks/images/") -> tuple:
        """Ищет похожие изображения в базе
        Arguments:
            image_path: путь до изображения на диске
            n_top: сколько результатов выводить
            save_graphs: куда сохранять графики
            path_to_images: для дебага - где лежат изображения базы
            debug: если True показать картинкой результаты
        
        Returns:
            predicts[top]: проценты предсказания для каждой найденной
            list_of_images: пути до найденных изображений
            list_of_urls: пути до объявлений
            list_of_graphs: пути до графиков
        """
        flat_img = self.resnet.predict_proba(np.array(Image.open(image_path).resize((224,224))).reshape(1,224,224,3))
        temp_array = np.zeros(shape=(len(self.flat_base), 2 * self.flat_base.shape[-1]))

        for idx, item in enumerate(self.flat_base):
            temp_array[idx] = np.hstack([flat_img, item.reshape(1,2048)])
        
        predicts = self.model.predict_proba(temp_array)
        del temp_array

        predicts = predicts[:, 1]
        top = predicts.argsort()[-n_top:][::-1]
        
        predicts = (predicts - np.min(predicts)) / (np.max(predicts) - np.min(predicts)) * 100.0
        list_of_images = self.base.iloc[top, 1].to_list()
        list_of_urls = self.base.iloc[top, 2].to_list()
        list_of_graphs = []

        img_name = hashlib.sha256(image_path.encode("utf-8")).hexdigest()
        for i in range(n_top):
            plt.rcParams.update({'font.size':14})
            plt.scatter([x for x in range(self.flat_base.shape[-1])], self.flat_base[top[i]])
            plt.axis('off')
            plt.savefig(save_graphs + img_name + f"{i}.jpg", bbox_inches='tight')
            plt.close()
            list_of_graphs.append(save_graphs + img_name + f"{i}.jpg")
            
        return predicts[top], list_of_images, list_of_urls, list_of_graphs


class InternetSearch():
    
    """
    Поиск в рунете: создать объект InternetSearch, 
    передать ему список url`ов в метод .search(...) -> словарь из совпадений по торговым площадкам
    можно использовать много объектов распараллелить. 
    Использовать как парсер, так и оперативную функцию в веб-морде
    """

    def __init__(self) -> None:
        self.user = os.getlogin()
        options = Options()
        options.headless = True
        self.driver = webdriver.Firefox(options=options)

    def yandex_images_search(self, img:str) -> list:
    # идем на яндекс
        self.driver.get("https://yandex.ru/images/search")
        # нажимаем поиск по картинкам
        search_button = self.driver.find_element_by_xpath("/html/body/header/div/div[1]/div[2]/form/div[1]/span/span/div[2]/button")
        search_button.click()
        # подсовываем url картинки
        text_field = self.driver.find_element_by_xpath("/html/body/div[2]/div/div[1]/div/form[2]/span/span/input")
        text_field.send_keys(img)
        self.driver.find_element_by_xpath("/html/body/div[2]/div/div[1]/div/form[2]/button").click()
        time.sleep(10)
        #переходим в выдачу
        links = self.driver.find_elements_by_class_name("other-sites__snippet-title")
        imgs = self.driver.find_elements_by_class_name("other-sites__preview-link")

        if len(links) == 0:
            return
        
        results = []
        for (site, img) in zip(links, imgs):
            
            results.append((site.find_elements_by_xpath(".//*")[0].get_attribute("href"), site.text, img.get_attribute("href")))
            
        return results

    def bad_urls_check(self, list_of_links:list) -> list:
        result = []
        if list_of_links is None:
            return result
        for url in list_of_links:
            if "avito" in url[0] and "q=" not in url[0] and "phone" not in url[0] and "geo=" not in url[0]:
                result.append(url)
            if "ideipodarkov" in url[0]:
                result.append(url)
        return result

    def yandex_batch_execute(self, urls:list, inplace:bool=True) -> dict:
        result = {}
        for img in urls:
            if inplace:
                result[img] = self.bad_urls_check(self.yandex_images_search(img))
            else:
                result[img] = self.yandex_images_search(img)
            
        return result

    def search(self, image:str) -> dict:
        return self.bad_urls_check(self.yandex_images_search(image))
