import os

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from PIL import Image
from tensorflow import keras as keras

PATH = os.path.realpath(__file__).split("nnmodule")[0]

class NNSearch():

    def __init__(self):
        self.flat_base = np.load(PATH + "res/flat_x_wo_aug.npy")
        
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

    def search(self, image_path:str, n_top:int=5, save_graphs:str="C:/temp/", path_to_images:str="C:/Devs/git/antikvar/NeuralNetworks/images/", debug:bool=True) -> tuple:
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

        if debug:
            plt.rcParams.update({'font.size':14})
            fig, ax = plt.subplots(2, 6, figsize=(15,8))
            ax[0,0].imshow(Image.open(image_path))
            ax[0,0].set_title("Исходное")
            ax[0,0].set_axis_off()
            for j, i in enumerate(range(1, 6)):
                ax[0,i].imshow(Image.open(path_to_images + list_of_images[j] + ".jpg"))
                ax[0,i].set_title("{:.2f}%".format(predicts[j]))
                ax[0,i].set_axis_off()
                
            ax[1,0].scatter([x for x in range(self.flat_base.shape[-1])], flat_img)
            ax[1,0].set_title("Исходное")
            for j, i in enumerate(range(1, 6)):
                ax[1,i].scatter([x for x in range(self.flat_base.shape[-1])], self.flat_base[top[j]])
                ax[1,i].set_title("{:.2f}%".format(predicts[top[j]]))
                ax[1,i].get_yaxis().set_visible(False)
            plt.subplots_adjust(wspace=0.2, hspace=0.1)
            plt.show()

        for i in range(n_top):
            plt.rcParams.update({'font.size':14})
            plt.scatter([x for x in range(self.flat_base.shape[-1])], self.flat_base[top[i]])
            plt.xlabel("Номер признака")
            plt.ylabel("Активация")
            plt.savefig(save_graphs + f"{i}.jpg")
            plt.close()
            list_of_graphs.append(save_graphs + f"{i}.jpg")

        return predicts[top], list_of_images, list_of_urls, list_of_graphs
