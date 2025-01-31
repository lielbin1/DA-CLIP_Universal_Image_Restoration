import os
import shutil

def reorganize_data(base_dir, output_dir):
    # Paths for GT, LQ, and LLQ directories
    gt_dir = os.path.join(base_dir, "GT")
    lq_dir = os.path.join(base_dir, "LQ")
    llq_dir = os.path.join(base_dir, "LLQ")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all noise types in GT and LQ
    noise_types = [d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))]

    # Iterate through each noise type
    for noise_type in noise_types:
        # Create the new noise_type folder in the output directory
        new_noise_dir = os.path.join(output_dir, noise_type)
        os.makedirs(os.path.join(new_noise_dir, "GT"), exist_ok=True)
        os.makedirs(os.path.join(new_noise_dir, "LQ"), exist_ok=True)
        os.makedirs(os.path.join(new_noise_dir, "LLQ"), exist_ok=True)

        # Copy GT and LQ images
        gt_noise_dir = os.path.join(gt_dir, noise_type)
        lq_noise_dir = os.path.join(lq_dir, noise_type)

        for img_name in os.listdir(gt_noise_dir):
            gt_src = os.path.join(gt_noise_dir, img_name)
            lq_src = os.path.join(lq_noise_dir, img_name)

            if os.path.isfile(gt_src):  # Only process files
                gt_dst = os.path.join(new_noise_dir, "GT", img_name)
                shutil.copy(gt_src, gt_dst)

            if os.path.isfile(lq_src):  # Only process files
                lq_dst = os.path.join(new_noise_dir, "LQ", img_name)
                shutil.copy(lq_src, lq_dst)

        # Organize LLQ images
        for llq_subfolder in os.listdir(llq_dir):
            if noise_type in llq_subfolder:  # Check if noise_type is in the LLQ folder name
                other_noise = llq_subfolder.replace(noise_type + "_", "").replace("_" + noise_type, "")

                llq_src_dir = os.path.join(llq_dir, llq_subfolder)
                llq_dst_dir = os.path.join(new_noise_dir, "LLQ", other_noise)
                os.makedirs(llq_dst_dir, exist_ok=True)

                for img_name in os.listdir(llq_src_dir):
                    llq_src = os.path.join(llq_src_dir, img_name)

                    if os.path.isfile(llq_src):  # Only process files
                        llq_dst = os.path.join(llq_dst_dir, img_name)

                        # Avoid duplicates
                        if not os.path.exists(llq_dst):
                            shutil.copy(llq_src, llq_dst)

if __name__ == "__main__":
    base_dir = "/sise/eliorsu-group/lielbin/Courses/Generative_Models_in_AI/daclip-uir-main/universal-image-restoration/config/daclip-sde/universal/val"
    output_dir = "/sise/eliorsu-group/lielbin/Courses/Generative_Models_in_AI/daclip-uir-main/universal-image-restoration/config/daclip-sde/universal/val_reorganized"
    reorganize_data(base_dir, output_dir)
    print("Data reorganization complete!")

