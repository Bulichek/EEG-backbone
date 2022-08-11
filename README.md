# Тестовое задание

Репозиторий имеет следующую структуру.

```
project
│───README.md
│───Model.ipynb -- Note with experiments
│───EDA.ipynb -- EDA
|───setup.py
|
|───src
|   |───utils.py
|   |───data_processing
|       │───filter_data.py -- remove redundant classes
|       |───split_data.py -- train/val split
```

Фильтрация данных, удаление семплов с классами, которые не используются для обучения.

```
python src\data_processing\filter_data.py \
    --data-path <path_to_data> \
    --path-to-save <path_to_save_filtered_data> \
    --required-labels <labels_to_be_left>
```

Разбиение данных на train/val.

```
python src\data_processing\split_data.py \
    --data-path <path_to_data> \
    --path-to-save <path_to_save_splitted_idx> \
    --train-frac <fraction_of_train>
```

Анализ данных производится в `EDA.ipynb`, эксперименты и выводы по ним содержатся в `Model.ipynb`. Для обучения моделей использовались только данные предоставленные преподавателями.
