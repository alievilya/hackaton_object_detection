import os

import pandas as pd


def get_solution_labels_df(path_to_txt_folder):
    simple_solution = []
    for detection_file in os.listdir(path_to_txt_folder):
        img_name = detection_file.split('.')[0] + '.jpg'
        with open(path_to_txt_folder + detection_file, 'r') as f:
            data = f.read()
            data = [i for i in data.split('\n') if i != '']
        for line in data:
            val = [float(i) for i in line.split()]
            class_id, xywh, conf = val[0], val[1:5], val[5]
            class_id = int(class_id)
            center_x, center_y, width, height = xywh
            xmin = center_x - (width / 2)
            xmax = center_x + (width / 2)
            ymin = center_y - (height / 2)
            ymax = center_y + (height / 2)
            simple_solution.append([img_name, class_id, conf, xmin, xmax, ymin, ymax])
    return simple_solution


exp_labels = 'yolov5/runs/detect/exp3/labels/'
solution = get_solution_labels_df(exp_labels)
solution_df = pd.DataFrame(solution,
                           columns=['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax'])

solution_df.to_csv("solution3.csv", sep=';', index=False)
df = pd.read_csv("solution3.csv", sep=';', index_col=None)
print(df)
