За основу был взят алгоритм YOLOv5, ссылка на репозиторий: https://github.com/ultralytics/yolov5

0. Необходимо установить все требуемые библиотеки: 
    ```
    pip install -r requirements.txt
    ```
1. Исходные наборы данных располагаются в корне проекта в папках train и test. 
Прежде всего исходные метки классов были предобработаны с помощью скрипта preprocess_dataset.py, который также выполняет
разбиение на обучающий и валидационный наборы данных в соотношении 0.93/0.07. 
    ```
    python preprocess_dataset.py
    ```
   
2. Обучать сеть с нуля на датасете такого объема нецелесообразно, поэтому было решено использовать предобученные веса модели
   yolov5l6 с размером входного слоя 1280. Был создан файл config.yaml с путями к датасету. Также использовалась
   medium аугментация.
Команда для запуска процесса обучения:
   ``` 
   python ./yolov5/train.py --batch 16 --epochs 300 --hyp hyp.scratch-med.yaml \
   --data_yolo_format/config.yaml --weights yolov5l6.pt --img 1280 --cache --name training --project hackaton \
   --device 0,1
   ``` 
    Сеть обучалась на двух Tesla V100. Обучение заняло около 2-х часов и было остановлено после 107 эпох, 
наилучший результат был получен на 95-й эпохе. 

3. Команда для запуска детектирования на тестовом наборе данных: 
   ```
   python ./yolov5/detect.py --weights hackaton/training/weights/best.pt --data data_yolo_format/config.yaml \
   --img 1280 --conf 0.25 --source test/images --save-txt --save-conf
   ```

4. Получение итогового csv-файла
   ```
   python get_results.py
   ```

Лучшие метрики на валидации:

metric | value |
--- | --- |
epoch | 95 |
mAP_0.5 | 0.758 |
mAP_0.5:0.95 | 0.544 |
precision | 0.81 | 
recall | 0.739 |


Обучающий датасет дополнительно не размечался, тестовый датасет также не размечался и использовался исключительно для 
финального теста.

Прикрепляю ссылку на обученные веса: https://drive.google.com/file/d/1bFyA6pGY4aaMVebraq2TV-6aOC_NArxw/view?usp=sharing