import os



def rename_file_in_folder(folder_path):
    #? Get folder_path/file_name for each file in folder_path
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    #? sort file by time
    sorted_files_by_time = sorted(file_list, key=os.path.getctime)
    print(sorted_files_by_time)

    #? os.rename(old_file_path, new_file_path)
    for i, file_name in enumerate(sorted_files_by_time):
        os.rename(file_name, f'{folder_path}\\{str(i+1)}.jpg')

rename_file_in_folder("Foundation of Prompt Engineer")