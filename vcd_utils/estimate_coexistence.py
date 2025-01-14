import json
import numpy as np
from collections import defaultdict
from itertools import combinations

def compute_cooccurrence_rate(annotation_file):
    # 打开并读取COCO注释文件
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # 获取类别ID到名称的映射
    categories = {category['id']: category['name'] for category in coco_data['categories']}
    category_ids = sorted(categories.keys())
    num_categories = len(category_ids)

    # 创建一个类别ID到索引的映射，用于矩阵索引
    category_to_index = {category_id: index for index, category_id in enumerate(category_ids)}

    # 初始化共现计数字典和单独出现计数字典
    cooccurrence_counts = np.zeros((num_categories, num_categories), dtype=int)
    category_counts = np.zeros(num_categories, dtype=int)

    # 图片ID到类别ID的映射
    image_to_categories = defaultdict(set)

    # 填充图片到类别的映射
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        image_to_categories[image_id].add(category_id)

    # 计算每张图片中的类别组合和单独出现次数
    for category_ids in image_to_categories.values():
        for category_id in category_ids:
            idx = category_to_index[category_id]
            category_counts[idx] += 1  # 增加单独出现次数

        for cat1, cat2 in combinations(category_ids, 2):
            idx1 = category_to_index[cat1]
            idx2 = category_to_index[cat2]
            # 更新共现计数
            cooccurrence_counts[idx1, idx2] += 1
            cooccurrence_counts[idx2, idx1] += 1  # 矩阵是对称的

    # 计算共现率矩阵
    cooccurrence_rate_matrix = np.zeros((num_categories, num_categories), dtype=float)

    for i in range(num_categories):
        for j in range(num_categories):
            if category_counts[i] > 0:  # 避免除以零
                cooccurrence_rate_matrix[i, j] = cooccurrence_counts[i, j] / category_counts[i]

    return cooccurrence_rate_matrix, categories

# 示例使用
annotation_file = '../../../data/coco/annotations/instances_train2014.json'  # 替换为你的COCO注释文件路径
cooccurrence_rate_matrix, categories = compute_cooccurrence_rate(annotation_file)


# 打印共现矩阵
# print("Co-occurrence Matrix:")
# print(cooccurrence_matrix)

# 打印类别名称
# print("Categories:")
# for idx, category_id in enumerate(sorted(categories.keys())):
#     print(f"Index {idx}: {categories[category_id]}")

flattened_matrix = cooccurrence_rate_matrix.flatten()

# 获取前3个最大值的索引
top_indices = np.argsort(flattened_matrix)[-30:]

# 将一维索引转换为二维坐标
top_values_and_coords = [(flattened_matrix[i], np.unravel_index(i, cooccurrence_rate_matrix.shape)) for i in top_indices]
import pdb;pdb.set_trace()