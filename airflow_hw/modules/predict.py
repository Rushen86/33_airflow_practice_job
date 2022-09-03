import logging
import os
import pandas as pd
import dill
import json
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
path = os.environ.get('PROJECT_PATH', 'airflow_hw')
# создаем экземпляр класса BaseModel для контроля входных данных предсказаний
class Form(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str

def predict():
    # загружаем файл с пайплайном обработки данных и моделью машинного обучения,
    # для этого берем файл с последней меткой времени
    # Сформируем список файлов pkl
    list_pkls = []
    path_pkl_search = Path(f'{path}/data/models')
    for file_ in os.listdir(path_pkl_search):
        long_path = Path(os.path.join(path_pkl_search, file_))
        if os.path.isfile(long_path) and ('.pkl' in file_):
            list_pkls.append(long_path)
    # далее найдем файл с максимальной меткой времени
    list_datetime = []
    for elem in list_pkls:
        str_date = Path(elem).stem.replace("cars_pipe_", "")
        list_datetime.append(datetime.strptime(str_date, "%Y%m%d%H%M"))
    my_pkl_file = Path(f'{path}/data/models/cars_pipe_{max(list_datetime).strftime("%Y%m%d%H%M")}.pkl')
    if my_pkl_file in list_pkls:
        with open(my_pkl_file, 'rb') as file_:
            last_model = dill.load(file_)
        logging.info(f'The file {my_pkl_file} will be used for prediction')
    else:
        logging.info(f'File {my_pkl_file} is not found')
        exit()
    # формируем список файлов json
    # объявляем пустой список, сюда будем складывать пути с названиями файлов
    list_files = []
    # формируем путь для поиска файлов json
    path_search = Path(f'{path}/data/test/')
    # в цикле перебираем все файлы в каталоге и складываем в список если это json файл
    for file_ in os.listdir(path_search):
        long_path = Path(os.path.join(path_search, file_))
        if os.path.isfile(long_path) and ('.json' in file_):
            list_files.append(long_path)
    # объявляем функцию предсказания категории цены для каждого json файла
    def pred_result(p_data: dict) -> pd.DataFrame:
        form = Form.parse_obj(p_data)
        df = pd.DataFrame.from_dict([form.dict()])
        pred = last_model.predict(df)
        df_out = pd.DataFrame({'id': form.dict().get('id'), 'price_category': pred})
        return df_out
    # в цикле перебираем каждый json файл, делаем predict, формируем dataframe со всеми данными
    # объявляем пустой список куда будем складывать датафреймы
    list_out = []
    # читаем каждый json файл, делаем предикт и дабавляем полученный датафрейм в список
    for list_ in list_files:
        with open(list_, 'rb') as file_:
            dict_ = json.load(file_)
        list_out.append(pred_result(dict_))
     # объединяем датавреймы в списке в один датафрейм
    df_out = pd.concat(list_out)
    # задаем индекс id
    df_out.set_index('id', inplace = True)
    # формируем строку где указываем путь для сохранения и имя полученного датафрейма
    save_name = Path(f'{path}/data/predictions/predict_out_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    # выводим данные в лог
    logging.info(f'my dataframe is : {df_out}')
    # сохраняем данные
    df_out.to_csv(save_name)

if __name__ == '__main__':
    predict()
