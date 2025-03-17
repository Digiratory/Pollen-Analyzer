import os


class DatasetPath():
    root = "Dataset/pollen_dataset_2025_02_05"
    raw_images = os.path.join(root, "samples")
    reference_images = os.path.join(root, "objects_clean")
    noise_bg_images = os.path.join(root, "Noise_40X")
    instance_segmentation = os.path.join(root, "instance_segmentation_test")
    real_historical_samples = os.path.join(root, "historical_samples")


class ModelPath():
    root = "models"
    instance_segmentation = os.path.join(
        root, "instance_segmentation", "InSegModel_release.pt")
    osc_classification = os.path.join(
        root, "osc_classification", "osc_classification_release.pt")


species_native = set(['Acer',
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
                  'Ambrosia'])
species_invasive = set(['Taraxacum', 'Malus'])

species_all = species_native | species_invasive

species_classes_map = {value: key for key, value in enumerate(species_native)}
species_classes_map["Unknown"] = -1

species_label_map = {v: k for k, v in species_classes_map.items()}

species_classes_map_all = {value: key for key, value in enumerate(species_all)}
species_classes_map_all["Unknown"] = -1
species_label_map_all = {v: k for k, v in species_classes_map_all.items()}