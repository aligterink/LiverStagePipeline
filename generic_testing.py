import os

def find_deepest_folder_with_string(base_path, target_string):
    deepest_folder = None
    max_depth = 0

    for root, dirs, _ in os.walk(base_path):
        if target_string in dirs and len(root.split(os.path.sep)) > max_depth:
            deepest_folder = root
            max_depth = len(root.split(os.path.sep))

    return deepest_folder

# Example usage
target_string = '/home/anton/Documents/code/hi/LiverStagePipeline/LiverStagePipelinesdfsdf'  # Replace with the path you want to search in
# target_string = 'LiverStagePipeline'

# parts = 


x = os.path.join(*(os.getcwd().split(os.path.sep)[:next((i for i in range(len(os.getcwd().split(os.path.sep)) -1, -1, -1) if 'LiverStagePipeline' in os.getcwd().split(os.path.sep)[i]), None)+1]))

string_list = target_string.split(os.path.sep)
last_index = next((i for i in range(len(string_list) - 1, -1, -1) if 'LiverStagePipeline' in string_list[i]), None)


print(x, string_list, last_index)
# deepest_folder = find_deepest_folder_with_string(base_path, target_string)
# print(deepest_folder)