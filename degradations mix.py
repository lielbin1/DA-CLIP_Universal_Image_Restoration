import numpy as np
import cv2
import os
import sys

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', 
    '.png', '.PNG', '.ppm', '.PPM', 
    '.bmp', '.BMP', '.tif'
]

def degrade(image, deg_type='jpeg', param=10):
    """Degrades the image based on the specified deg_type."""
    if deg_type == 'blur':
        ksize = (param, param)
        image_blur = cv2.GaussianBlur(image, ksize, 0)
        return image_blur

    elif deg_type == 'noisy':
        sigma = param / 255.0
        noisy_image = image + np.random.normal(0, sigma, image.shape)
        noisy_image = np.clip(noisy_image, 0, 1)
        return noisy_image

    elif deg_type == 'jpeg':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), param]
        _, enc = cv2.imencode('.jpg', (image * 255).astype(np.uint8), encode_param)
        jpeg_image = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        return jpeg_image

    else:
        return image

def is_image_file(filename):
    return any(filename.lower().endswith(extension.lower()) for extension in IMG_EXTENSIONS)

def generate_LQ(degradations):
    """
    Apply multiple degradations to the input images and save them in separate folders.

    degradations: List of tuples [(deg_type, param), ...]
    """
    home_dir = os.path.expanduser("~")
    sourcedir = os.path.join(home_dir, " ", "")
    savedir = os.path.join(home_dir, "", "")

    if not os.path.isdir(sourcedir):
        print(f"Error: source folder does not exist -> {sourcedir}")
        sys.exit(1)

    filepaths = [f for f in os.listdir(sourcedir) if is_image_file(f)]
    num_files = len(filepaths)
    print(f"Found {num_files} images in '{sourcedir}'")

    for deg_type, param in degradations:
        subfolder = f"{deg_type}_{param}"
        subfolder_path = os.path.join(savedir, subfolder)

        if not os.path.isdir(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Created folder: {subfolder_path}")

        for i, filename in enumerate(filepaths):
            print(f"[{deg_type}, {param}] No.{i+1} -- Processing: {filename}")

            full_path = os.path.join(sourcedir, filename)
            image = cv2.imread(full_path)

            if image is None:
                print(f"Warning: Could not read {full_path}, skipping.")
                continue

            image = image.astype(np.float32) / 255.0
            image_LQ = degrade(image, deg_type=deg_type, param=param)
            image_LQ = (image_LQ * 255).astype(np.uint8)

            save_path = os.path.join(subfolder_path, filename)
            cv2.imwrite(save_path, image_LQ)

    print("Finished!!!")

if __name__ == "__main__":
    degradations = [
        ('noisy', 50),
        ('blur', 25),
        ('jpeg', 10)
    ]
    generate_LQ(degradations)
