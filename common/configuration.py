import os


class DatasetPath():
    root = "Dataset/pollen_dataset_kfu_2025"
    raw_sample_images = os.path.join(root, "reference_sources", "samples")
    raw_sample_hist_images = os.path.join(root, "reference_sources", "historical_train")
    objects_clean = os.path.join(root, "objects_clean")
    reference_images = os.path.join(root, "reference_ds")
    noise_bg_images = os.path.join(root, "Noise_40X")
    instance_segmentation = os.path.join(root, "instance_segmentation_test")
    real_historical_samples = os.path.join(root, "historical_samples")
    real_historical_samples_classification = os.path.join(root, "historical_classification_test")
    


class ModelPath():
    root = "models"
    instance_segmentation = os.path.join(
        root, "instance_segmentation", "InSegModel_release.pt")
    osc_classification = os.path.join(
        root, "osc_classification", "osc_classification_release.pt")


species_native = ['Acer',
                  'Alnus',
                  'Apiaceae',
                  'Artemisia',
                  'Betula',
                  'Chenopodiaceae',
                  'Corylus',
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

species_by_ordo = {
    "Sapindales" : ["Acer"],
    "Fagales" : ["Alnus", "Betula", "Corylus", "Quercus"],
    "Rosales": ["Ulmus", "Malus","Urtica"],
    "Malpighiales": ["Salix", "Populus tremula"],
    "Malvales" : ["Tilia"],
    "Pinales":["Pinus", "Picea"],
    "Apiales":["Apiaceae"],
    "Asterales":["Artemisia", "Ambrosia", "Taraxacum"],
    "Poales": ["Poaceae"],
    "Caryophyllales": ["Chenopodiaceae", "Rumex"],
    "Lamiales": ["Plantago"],
}

species_by_familia = {
    "Sapindaceae": ["Acer"],
    "Betulaceae": ["Alnus", "Betula", "Corylus"],
    "Ulmaceae": ["Ulmus"],
    "Fagaceae":["Quercus"],
    "Salicaceae":["Salix", "Populus tremula"],
    "Malvaceae":["Tilia"],
    "Pinaceae":["Pinus", "Picea"],
    "Apiaceae":["Apiaceae"],
    "Asteraceae":["Artemisia", "Ambrosia", "Taraxacum"],
    "Rosaceae": ["Malus"],
    "Poaceae":["Poaceae"],
    "Chenopodiaceae":["Chenopodiaceae"],
    "Plantaginaceae":["Plantago"],
    "Polygonaceae": ["Rumex"],
    "Urticaceae": ["Urtica"],
}


genetic_order_classes = [
    'Acer',
    'Alnus',
    'Corylus',
    'Quercus',
    'Malus',
    'Ulmus',
    'Urtica',
    'Populus tremula',
    'Salix',
    'Ambrosia',
    'Artemisia',
    'Taraxacum', 
    'Apiaceae',
    'Plantago',
    'Betula',
    'Picea',
    'Pinus',
    'Poaceae',
    'Chenopodiaceae',
    'Rumex',
    'Tilia',
    'Unknown',
    'Not pollen'
]



species_all = species_native + species_invasive

species_classes_map = {value: key for key, value in enumerate(species_native)}
species_classes_map["Unknown"] = -1

species_label_map = {v: k for k, v in species_classes_map.items()}

species_classes_map_all = {value: key for key, value in enumerate(species_all)}
species_classes_map_all["Unknown"] = -1
species_label_map_all = {v: k for k, v in species_classes_map_all.items()}
