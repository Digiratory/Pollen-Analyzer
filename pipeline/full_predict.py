from typing import Callable
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch

from common.osc_classification import OpenSetClassificationModel, extract_features
from common.osc_classification import distance_to_scores, scores_to_class

from Instance_Segmentation import SegmentationModelWrapper


def classify_image(img_path: str,
                   classification_model: OpenSetClassificationModel,
                   prediction_to_class_name: Callable[[int,], str]):

    img = Image.open(img_path)  # .convert("RGB")
    features, pred_distances = extract_features(img, classification_model)
    vector = features[0]
    # classification by model
    scores = distance_to_scores(pred_distances)
    prediction = scores_to_class(scores, 8)
    predict_str = [prediction_to_class_name(
        l.item()) for l in prediction.cpu()]
    pred_class = predict_str[0]
    return pred_class, vector.cpu().numpy()


class ImageProcessor():
    def __init__(self,
                 segmentation_model_path: str,
                 classification_model_path: str,
                 prediction_to_class_name: Callable[[int,], str]):

        self._segmentation_model_path = segmentation_model_path
        self._segmentation_model = SegmentationModelWrapper(
            segmentation_model_path)

        self._prediction_to_class_name: Callable[[
            int,], str] = prediction_to_class_name

        self._classification_model = OpenSetClassificationModel(19)
        self._classification_model.load_state_dict(
            torch.load(classification_model_path))
        self._classification_model.to("cuda")
        self._classification_model.eval()

    def process_image(self, image_path: str, crops_path: str, augment: int = 0, with_embeddings=False):
        extracted_objs = self._segmentation_model(image_path=image_path,
                                                  output_dir=crops_path)
        pred_labels = []
        for obj in extracted_objs:
            pred_class, embeddings = self._classify_image(obj, augment)
            if with_embeddings:
                pred_labels.append([pred_class, embeddings])
            else:
                pred_labels.append(pred_class)
        return pred_labels

    def _classify_image(self, img: Image, augment: int):
        features, pred_distances = extract_features(img,
                                                    self._classification_model,
                                                    augment)
        vector = features[0]
        # classification by model
        scores = distance_to_scores(pred_distances)
        prediction = scores_to_class(scores, 7)
        predict_str = [self._prediction_to_class_name(l.item()
                                                      ) for l in prediction.cpu()]
        pred_class = predict_str[0]
        return pred_class, vector.cpu().numpy()
