import os

dir_path = "K:\MITM\wiretap_images"
title = "wiretap."
for file_name in os.listdir(dir_path):
    if file_name.endswith(".png"):  # если файл имеет расширение .png
        new_file_name = title + file_name  # создаем новое имя файла
        file_path = os.path.join(dir_path, file_name)
        new_file_path = os.path.join(dir_path, new_file_name)
        os.rename(file_path, new_file_path)
=======
import os

dir_path = "K:\MITM\wiretap_images"
title = "wiretap."
for file_name in os.listdir(dir_path):
    if file_name.endswith(".png"):  # если файл имеет расширение .png
        new_file_name = title + file_name  # создаем новое имя файла
        file_path = os.path.join(dir_path, file_name)
        new_file_path = os.path.join(dir_path, new_file_name)
        os.rename(file_path, new_file_path)
