import os


class DatasetPath():
    root = "Dataset/pollen_dataset_2025_02_05"
    raw_images = os.path.join(root, "samples")
    reference_images = os.path.join(root, "objects_clean")
    noise_bg_images = os.path.join(root, "Noise_40X")
    instance_segmentation = os.path.join(root, "instance_segmentation_test")

class ModelPath():
    root = "models"
    instance_segmentation = os.path.join(root, "instance_segmentation", "InSegModel_release.pt")

species_native = ['Acer',
                  'Alnus',
                  'Apiaceae',
                  'Artemisia',
                  'Betula',
                  'Chenopodiaceae',
                  'Corylus',
                  'Fraxinus',
                  'Picea',
                  'Pinus',
                  'Plantago',
                  'Poaceae',
                  'Populus tremula',
                  'Quercus',
                  'Rumex',
                  'Salix',
                  'Tilia',
                  'Urtica',
                  'Ulmus',
                  'Ambrosia']
species_invasive = ['Taraxacum', 'Malus']
