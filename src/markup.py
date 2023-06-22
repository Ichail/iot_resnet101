import os

dir_path = input("Enter to directory path for markup: ")

title = input("Enter title: ")
for file_name in os.listdir(dir_path):
    if file_name.endswith(".png"):
        new_file_name = title + file_name
        file_path = os.path.join(dir_path, file_name)
        new_file_path = os.path.join(dir_path, new_file_name)
        os.rename(file_path, new_file_path)
