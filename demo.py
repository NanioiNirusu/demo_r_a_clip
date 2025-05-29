
import os
import csv
import time
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageFile

import torch
import torchvision as tv
import open_clip
from open_clip.clip_model_adapter import L2RCLIP

#############################################
# CONFIGURATION AND SETUP
#############################################

# Allow PIL to load truncated images if needed
ImageFile.LOAD_TRUNCATED_IMAGES = True


#############################################
# VISUALIZATION FUNCTIONS
#############################################

def render_figures_with_order(
        image_list,
        values,
        sup_values=None,
        figure_title=None,
        show_value=True,
):
    """
    Render images in a figure ordered by their quality scores.

    Args:
        image_list: List of image objects to display
        values: Quality scores for each image
        sup_values: Optional supplementary values (e.g., ground truth scores)
        figure_title: Optional title for the figure
        show_value: Whether to display score values as titles
    """
    if sup_values is None:
        compact = list(zip(image_list, values))
    else:
        compact = list(zip(image_list, values, sup_values))
    sorted_data = sorted(compact, key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(16, 12))
    for i, (image, value, *sup_value) in enumerate(sorted_data):
        plt.subplot(1, len(image_list), i + 1)
        plt.imshow(image)
        plt.axis('off')
        if show_value:
            if sup_values is None:
                v = f'score={value:.4f}'
            else:
                v = f'true/pred score = {sup_value[0]:.4f}/{value:.4f}'
            plt.title(v)
    plt.tight_layout()
    plt.show()


#############################################
# UTILITY FUNCTIONS
#############################################

def get_unique_filename(base_output_file):
    """
    Generate a unique output filename by incrementing the distortion number

    Args:
        base_output_file (str): Base filename to check and potentially modify

    Returns:
        str: A unique filename with incremented distortion number
    """

    def increment_distortion_number(filename):
        import re
        match = re.match(r'(D)(\d+)(_rankingclip_metrics\.csv)', filename)
        if match:
            prefix, number, suffix = match.groups()
            return f"{prefix}{int(number) + 1}{suffix}"
        return filename

    current_filename = base_output_file
    while os.path.exists(current_filename):
        current_filename = increment_distortion_number(current_filename)
    return current_filename


#############################################
# MAIN EVALUATION FUNCTION
#############################################

def main():
    """
    Main function to evaluate image quality using RankingCLIP.
    Loads the model, processes images, and saves results to CSV.
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model configuration
    config_path = './configs/model-config.json'
    model_name = 'hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup'

    with open(config_path, 'r') as f:
        model_cfg = json.load(f)

    # Initialize model
    model = L2RCLIP(
        **model_cfg,
        model_version='clip-adapter-v2',
    )
    model = model.to(device)

    tokenizer = open_clip.get_tokenizer(model_name)

    _, _, transform = open_clip.create_model_and_transforms(
        model_name=model_name,
    )
    transform = tv.transforms.Compose([
        tv.transforms.Resize((320, 320)),
        transform.transforms[2],
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    # Load checkpoint
    open_clip.load_checkpoint(
        model,
        checkpoint_path='./artifacts/cliprank-iqamos.pt',
    )

    # Path to your image directory
    image_dir = Path("J:/Masters/Datasets/AGIQA-1k-Database/file")

    # Get all image files in the directory
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(str(image_dir / ext)))

    # Output CSV file
    base_output_file = "D1_rankingclip_metrics.csv"
    output_file = get_unique_filename(base_output_file)

    # Define prompts for evaluation
    prompts = [
        "Rank the image quality from worst to best.",
        "Rank the images from low quality to high quality.",
        "Rank the visual appeal of these images."
    ]

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image", "RankingAware", "Prompt Used", "Evaluation Time (s)"
        ])

        # Process images in batches to avoid memory issues
        batch_size = 5
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]

            try:
                # Load and preprocess images
                image_array = torch.cat([
                    transform(Image.open(img).convert("RGB")).unsqueeze(0)
                    for img in batch_files
                ])

                # Process with each prompt and average the results
                all_scores = []
                for prompt in prompts:
                    start_time = time.time()

                    # Tokenize text
                    text_inputs = tokenizer([prompt] * len(batch_files))

                    # Run inference
                    with torch.no_grad():
                        output = model(
                            image_array.to(device),
                            text_inputs.to(device),
                        )

                    # Get predictions
                    predictions = output['adapter_logits'].cpu().numpy().ravel()
                    all_scores.append(predictions)

                    eval_time = time.time() - start_time

                # Average scores across prompts
                avg_scores = np.mean(all_scores, axis=0)

                # Write results to CSV
                for j, img_path in enumerate(batch_files):
                    writer.writerow([
                        img_path,
                        avg_scores[j],
                        "Average of quality prompts",
                        eval_time / len(prompts)
                    ])

                    print(f"Image: {img_path}")
                    print(f"  Quality Score: {avg_scores[j]}")
                    print(f"  Evaluation Time: {eval_time / len(prompts):.4f}s\n")

            except Exception as e:
                print(f"Error processing batch {i // batch_size}: {e}")

    print(f"RankingCLIP evaluation completed. Results saved to {output_file}")


#############################################
# MAIN SCRIPT EXECUTION
#############################################

if __name__ == "__main__":
    main()
