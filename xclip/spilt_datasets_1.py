import os
import random
import csv
from sklearn.model_selection import train_test_split


# 读取CSV文件，建立ID和名称的映射
def read_csv(csv_file):
    label_dict = {}
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过header
        for row in reader:
            label_dict[row[1].replace(' ', '_')] = row[0]  # 将名称中的空格替换为下划线
    return label_dict


# 查找所有视频文件并将名称与id对应
def collect_video_files(video_dir, label_dict):
    video_files = []
    unmatched_videos = []

    for file in os.listdir(video_dir):
        if file.endswith(".avi"):
            video_name = file.rsplit('.', 1)[0]  # 去掉扩展名
            matched = False
            for name in label_dict:
                if name in video_name:  # 匹配name，考虑到下划线替换
                    video_files.append((file, label_dict[name]))
                    matched = True
                    break
            if not matched:
                unmatched_videos.append(file)

    return video_files, unmatched_videos


# 创建训练和验证文件
def create_train_val_split(video_files, train_ratio=0.8):
    train_files, val_files = train_test_split(video_files, train_size=train_ratio, random_state=42)
    return train_files, val_files


# 写入到txt文件
def write_to_txt(file_list, output_file):
    with open(output_file, 'w') as f:
        for video_file, label in file_list:
            f.write(f"{video_file} {label}\n")

if __name__ == "__main__":
    csv_file = "E:\\yolo_track\\X-CLIP\\labels\\hmdb_51_labels.csv"  # CSV文件路径
    video_dir = "E:\\yolo_track\\X-CLIP\\dataset\\videos"      # 视频文件夹路径
    train_output = "E:\\yolo_track\\X-CLIP\\dataset\\train.txt"  # 训练集输出路径
    val_output = "E:\\yolo_track\\X-CLIP\\dataset\\val.txt"      # 验证集输出路径

    # 读取CSV，获取 name -> id 的映射
    label_dict = read_csv(csv_file)

    # 搜索视频文件，并匹配它们的label
    video_files, unmatched_videos = collect_video_files(video_dir, label_dict)

    # 划分为训练集和验证集
    train_files, val_files = create_train_val_split(video_files)

    # 写入到txt文件
    write_to_txt(train_files, train_output)
    write_to_txt(val_files, val_output)

    # 打印未匹配的视频文件
    if unmatched_videos:
        print("未匹配到的文件:")
        for video in unmatched_videos:
            print(video)

    print(f"训练集和验证集文件已生成: {train_output}, {val_output}")
