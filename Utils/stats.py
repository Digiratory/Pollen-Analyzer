import os


def count_files(directory):
    """Function for counting the amount of images."""
    total_image_count = 0
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        subdir_image_count = 0
        for species_dir in os.listdir(subdir_path):
            species_path = os.path.join(subdir_path, species_dir)
            if not os.path.isdir(species_path):
                continue
            species_count = sum(
                1 for file in os.listdir(species_path) if file.endswith(".png")
            )
            if species_count > 0:
                print(f"{species_path} images: {species_count} png")
                subdir_image_count += species_count
            species_count = sum(
                1 for file in os.listdir(species_path) if file.endswith(".jpg")
            )
            if species_count > 0:
                print(f"{species_path} images: {species_count} jpg")
                subdir_image_count += species_count
        print(f"{subdir_path} total images: {subdir_image_count}")
        print("-" * 10)
        total_image_count += subdir_image_count
    print(f"{directory} total images: {total_image_count}")


if __name__ == "__main__":
    ROOTDIR = "."
    count_files(ROOTDIR)
