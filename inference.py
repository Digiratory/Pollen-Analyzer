import os
import re
import json
import torch
import pandas as pd

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from annoy import AnnoyIndex
from collections import Counter
from torchvision import transforms, models
from Instance_Segmentation.image_from_mask import segment_objects


def get_feature_extractor(model):
    if type(model) == models.inception.Inception3:
        feature_extractor = torch.nn.Sequential(
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            model.Mixed_5b,
            model.Mixed_5c,
            model.Mixed_5d,
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(start_dim=1),
        )
    else:
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    return feature_extractor


def extract_features(img_path: str, feature_extractor):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.squeeze().numpy()


def save_class_to_indices(class_to_indices: dict, filename: str):
    with open(filename, "w") as f:
        json.dump(class_to_indices, f)


def load_class_to_indices(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_class(db: AnnoyIndex, vector, index_to_class: dict, k: int = 5):
    neighbor_ids = db.get_nns_by_vector(vector, k, include_distances=False)
    
    neighbor_classes = [index_to_class[neighbor_id] for neighbor_id in neighbor_ids]
    most_common_class = Counter(neighbor_classes).most_common(1)[0][0]
    return most_common_class


def get_classificaion_params():
    labels = []

    loaded_class_to_indices = load_class_to_indices(legend_path)
    index_to_class = {}
    for class_name, indices in loaded_class_to_indices.items():
        labels += [class_name]
        for idx in indices:
            index_to_class[idx] = class_name    

    k = 10 # N-neighbours
    f = 768 # Vector size
    db = AnnoyIndex(f, 'euclidean') # Euclidean distance
    db_file_path = 'Data/db_folder/Pollen DB Swin Transformer.ann'
    db.load(db_file_path)

    return db, index_to_class, k, labels


if __name__ == '__main__':
    legend_path = r'Data/legend/Pollen DB Legend.json'
    path_to_model = r'Data/models/InSegModel_fb'
    input_dir = r'Data/CVAT_dataset/images/Test/Sample_1'
    output_dir = r'Save/tabels/test-1.csv'
    buff_dir = r'Save/buffer'

    score_threshold = 0.9
    mask_threshold = 0.75

    if not os.path.exists(buff_dir):
        os.makedirs(buff_dir)
    print('Object segmentation in progress:')
    for sample_name in tqdm(os.listdir(input_dir)):
        sample_path = os.path.join(input_dir, sample_name)
        segment_objects(path_to_model, 'buffer', sample_path, score_threshold, mask_threshold, sample_name)
        
    cl_model = models.swin_t(weights="DEFAULT")
    feature_extractor = get_feature_extractor(cl_model)

    db, index_to_class, k, labels = get_classificaion_params()

    df_pred = pd.DataFrame(columns=labels)
    print('Object classification in progress:')
    for obj_name in tqdm(os.listdir(buff_dir)):
        image_path = os.path.join(buff_dir, obj_name)
        vector = extract_features(image_path, feature_extractor)
        
        pollen_label = get_class(db, vector, index_to_class, k)
        img_num = int(re.search(r'\((\d+)\)', obj_name).group(1))

        if img_num in df_pred.index:
            df_pred.loc[img_num, pollen_label] += 1
        else:
            new_row = {col: 0 for col in df_pred.columns}
            new_row[pollen_label] = 1  
            df_pred.loc[img_num] = new_row

    list(map(os.unlink,(os.path.join(buff_dir,f) for f in os.listdir(buff_dir)))) # delete buffer
    df_pred.sort_index(inplace=True)
    df_pred.to_csv(output_dir)
