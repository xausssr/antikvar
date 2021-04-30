import os
import urllib.request
from datetime import date
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from sqlalchemy import create_engine
from urllib import request
engine = create_engine("mysql+mysqldb://root:232061521161@localhost/parsing?charset=utf8mb4")
HEADERS = {'user-agent': UserAgent().random}
HOST = 'https://antikwariat.ru/'
URL = 'https://antikwariat.ru'
PATH = 'C:/Users/GROM/PycharmProjects/parse/'
# URLS всех предметов
urls_items = []
def get_html(url):
    r = requests.get(url, headers=HEADERS)
    return r
# Получаем URL всех категорий на сайте
html = get_html(URL)
if html.status_code != 200:
    print('Error')
    print(html.status_code)
else:
    soup = BeautifulSoup(html.text, 'html.parser')
    soup = soup.find_all('div', class_='index-categories-item-title')
    items = []
    for s in soup:
        items.append(*s.find_all('a'))
    urls_categories = [i.attrs['href'] for i in items]
    # получаем URL всех предметов
    for url_category in urls_categories:
        html = get_html(url_category)
        if html.status_code != 200:
            print('Error')
            print(html.status_code)
        else:
            # получаем количество страниц
            pages_soup = BeautifulSoup(html.text, 'html.parser')
            pages_soup = pages_soup.find_all('li', class_='hidden-xs')
            len_pages = len(pages_soup)
            # получаем предметы с каждой страницы
            for i in range(1, len_pages + 1):
                html = get_html(f'{url_category}?lt=1&page={i}')
                soup = BeautifulSoup(html.text, 'html.parser')
                name_category_soup = BeautifulSoup(html.text, 'html.parser')
                name_category = name_category_soup.find('title').contents[0].split(' -')[0]
                soup = soup.find_all('div', class_='sr-2-list-item-n-title')
                x = []
                for item in soup:
                    x.append(*item.find_all('a'))
                id1 = engine.execute("select max(ID) from parsing.table1").fetchall()[0][0]
                if id1 is None:
                    id1 = 0
                else:
                    id1 = int(id1) + 1
                for k in x:
                    url = k.attrs['href']
                    html = get_html(url)
                    if html.status_code != 200:
                        print('ERROR')
                        print(html.status_code)
                    else:
                        images_path = []
                        soup = BeautifulSoup(html.text, 'html.parser')
                        soup_description = soup.find("h1", class_="l-content-box-heading-title")
                        description = soup_description.contents
                        description = description[0]
                        description = description.strip()
                        soup = soup.find_all('a', class_='ad-images-zoom j-zoom')
                        images_urls = []
                        for z in soup:
                            images_urls.append(z.attrs['data-zoom'])
                        if len(images_urls) == 0:
                            break
                        search_str = f"select * from parsing.table1 where URL = '{url}' and description = '{description}' and category = '{name_category}'"
                        if len(engine.execute(search_str).fetchall()) == 0:
                            os.mkdir(f'images/{id1}')
                            path_images = ''
                            schet = 0
                            for z in images_urls:
                                count_images = len(images_urls)
                                temp_name = z.replace("/", "_").replace(":", "_").replace(".", "_")
                                resource = urllib.request.urlopen(z)
                                out = open(f"images/{id1}/{temp_name}_{id1}_{schet}.jpg", 'wb')
                                out.write(resource.read())
                                out.close()
                                path_images += PATH + f"/images/{id1}/{temp_name}_{id1}_{schet}.jpg,"
                                schet += 1
                            id1 += 1
                            path_images = path_images[:-1]
                            if len(path_images) == 0:
                                path_images = " "
                            print(path_images)

                            #print(f"INSERT parsing.table1 (URL, description, category, path, date) VALUES ('{url}', '{description}', '{name_category}', '{path_images}', '{today.year}.{today.month}.{today.day}')")
                            today = date.today()
                            engine.execute(f"INSERT parsing.table1 (URL, description, category, path, count_images, date) VALUES ('{url}', '{description}', '{name_category}', '{path_images}', '{count_images}', '{today.year}.{today.month}.{today.day}')")

