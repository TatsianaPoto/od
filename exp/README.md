# Experiments

Сборник всех проводимых экспериментов.

* [denoise.py](denoise.py) - скрипт для очистки изображения от шума (motion deblur). 
* [segmentation.py](segmentation.py) - поиск альтернативы CRAFT. Возможно использование различных алгоритмов предобработки изображений (connected components, canny edge detection) для сегментации символов без использования нейронных сетей. Однако, на данный момент, результаты хуже CRAFT.
* [cluster_boxes.py](cluster_boxes.py) - использование connectec_components вместе с алгоритмами кластеризации для сегментации символов.
* [difflib_experiments.ipynb](difflib_experiments.ipynb) - ноутбук с экспериментами над объединением нескольких номеров в один. На данный момент используется расстояние Левенштейна, однако возможны и другие алгоритмы.