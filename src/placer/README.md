# Схема работы

## objects_search.py

Файл с алгоритмом для поиска нескольких предметов на фотографии со светлым фоном.

## polygon_search.py

Файл с алгоритмом для поиска нарисованного на листе бумаги многоугольника на фотографии со светлым фоном.

## solver.py

### Solver

Поля:

- `objects` - список экземпляров класса `common.object.Object`.

Методы:

- `run` - запускает конвейер, этапами которого являются: загрузка изображения, поиск предметов, поиск многоугольника (если не задан), проверка возможности упаковки предметов в многоугольник. Возвращает результат последнего этапа.
