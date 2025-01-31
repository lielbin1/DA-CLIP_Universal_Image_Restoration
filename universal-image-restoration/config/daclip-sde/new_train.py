import sys

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

from PIL import Image
from matplotlib import pyplot as plt

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import os
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg16
from torchvision.transforms import ToPILImage
import neptune


# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required modules
import open_clip

from torchvision import transforms
import torch.nn.functional as F
# import neptune



NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNTVhYzkxZC1jOTc3LTQ4ZjYtOGFhZC00MzljZmVlOGFhYWEifQ=="

# Initialize Neptune run
run = neptune.init_run(
    project="nadavcherry/dp1",
    capture_hardware_metrics=True,
    tags=["da-clip_unet_new"],
    api_token=NEPTUNE_API_TOKEN
)

run["hyperparameters"] = {
    "learning_rate": 1e-5,
    "batch_size": 8,
    "num_epochs": 10,
}

# Load DaCLIP model
checkpoint = 'pretrained/daclip_ViT-B-32.pt'
model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("Loading DaCLIP model...")


# Define degradation categories
degradations = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy', 'shadowed',
                'snowy', 'uncompleted']
text = tokenizer(degradations)

class ImageNoiseDataset(Dataset):
    def __init__(self, llq_root, gt_root, transform=None):
        self.llq_root = llq_root
        self.gt_root = gt_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize all images to 224x224
            transforms.ToTensor()
        ])
        self.image_pairs = []

        for folder in os.listdir(llq_root):
            try:
                noise1, noise2 = folder.split("_")
                llq_folder = os.path.join(llq_root, folder)
                gt_folder = os.path.join(gt_root, noise1)

                if os.path.exists(gt_folder):
                    for img_name in os.listdir(llq_folder):
                        llq_path = os.path.join(llq_folder, img_name)
                        gt_path = os.path.join(gt_folder, img_name)
                        if os.path.exists(gt_path):
                            self.image_pairs.append((llq_path, gt_path))
            except ValueError:
                print(f"Skipping folder {folder} due to incorrect naming format.")
                continue

    def __len__(self):
        return 16

    def __getitem__(self, idx):
        llq_path, gt_path = self.image_pairs[idx]
        llq_image = Image.open(llq_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")

        # Apply the same transformation to both LLQ and GT images
        llq_image = self.transform(llq_image)
        gt_image = self.transform(gt_image)

        return llq_image, gt_image




def extract_embeddings(image):
    device = next(model.parameters()).device  # Get the model's device
    image = image.to(device)

    # Resize tensor if needed
    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

    with torch.no_grad(), torch.cuda.amp.autocast():
        # Get text features
        text_features = model.encode_text(text.to(device))  # Ensure text is on the same device
        # Get image features
        image_features, degra_features = model.encode_image(image, control=True)
        degra_features /= degra_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Identify the first noise type
        text_probs = (100.0 * degra_features @ text_features.T).softmax(dim=-1)
        index_1 = torch.argmax(text_probs, dim=-1)  # Shape: (batch_size,)

        # Mask the first noise and find the second noise
        masked_image = mask_noise(image, noise_index=index_1)
        _, degra_features_2 = model.encode_image(masked_image, control=True)
        text_probs_2 = (100.0 * degra_features_2 @ text_features.T).softmax(dim=-1)
        index_2 = torch.argmax(text_probs_2, dim=-1)  # Shape: (batch_size,)

        # Return embeddings with correct batch size
        return image_features, text_features[index_1], text_features[index_2]


def mask_noise(image, noise_index):
    """Apply a mask to the first noise"""
    mask = torch.ones_like(image)  # Placeholder mask logic
    masked_image = image * mask  # Apply the mask
    return masked_image

# Define the simplified UNet model
class SimplifiedUNet(nn.Module):
    def __init__(self, input_channels, embedding_dim, output_channels):
        super(SimplifiedUNet, self).__init__()

        self.enc1 = self.conv_block(input_channels + 3 * embedding_dim, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.up4 = self.up_conv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = self.up_conv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.up_conv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.up_conv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, vector_1, vector_2, vector_3, noisy_image):
        device = noisy_image.device

        # Ensure embeddings are on the correct device and reshape
        vector_1, vector_2, vector_3 = [v.to(device).view(noisy_image.size(0), -1) for v in (vector_1, vector_2, vector_3)]
        # print(f"vector_1 shape: {vector_1.shape}")
        # print(f"vector_2 shape: {vector_2.shape}")
        # print(f"vector_3 shape: {vector_3.shape}")
        # print(f"noisy_image shape: {noisy_image.shape}")

        # Concatenate embeddings and reshape
        joint_noise_embedding = torch.cat([vector_1, vector_2, vector_3], dim=-1)  # Shape: (batch_size, 1536)
        joint_noise_embedding = joint_noise_embedding.view(noisy_image.size(0), -1, 1, 1)  # Reshape to (batch_size, 1536, 1, 1)
        joint_noise_embedding = F.interpolate(joint_noise_embedding, size=noisy_image.shape[2:], mode="bilinear")

        # Concatenate embeddings with the noisy image
        x = torch.cat([noisy_image, joint_noise_embedding], dim=1)  # Shape: (batch_size, 1539, H, W)

        # Pass through the UNet
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.enc4(F.max_pool2d(e3, kernel_size=2))

        b = self.bottleneck(F.max_pool2d(e4, kernel_size=2))

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.vgg = vgg16(pretrained=True).features[:16].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def forward(self, output, target):
        # Ensure inputs are on the correct device
        output = output.to(self.device)
        target = target.to(self.device)

        # Extract VGG features and compute loss
        vgg_output = self.vgg(output)
        vgg_target = self.vgg(target)
        return F.mse_loss(vgg_output, vgg_target)


def train_unet(unet, dataloader, optimizer, criterion, num_epochs=10, checkpoint_dir="pretrained", save_every=100):
    """
    Train the UNet model and save checkpoints at intervals, logging progress to Neptune.
    """
    device = next(unet.parameters()).device
    unet.train()

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")  # Initialize best loss as infinity
    best_model_path = os.path.join(checkpoint_dir, "simplified_unet_best.pth")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for noisy_image, clean_image in pbar:
                noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)

                optimizer.zero_grad()

                # Extract embeddings
                vector_1, vector_2, vector_3 = extract_embeddings(noisy_image)

                # Forward pass
                output = unet(vector_1, vector_2, vector_3, noisy_image)

                # Compute loss
                loss = criterion(output, clean_image)

                # Backward pass
                loss.backward()

                # Gradient clipping (optional)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)

        # Save the best model based on the average loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(unet.state_dict(), best_model_path)
            print(f"New best model saved at {best_model_path}")
            try:
                run["model_checkpoint/best"].upload(best_model_path)
            except Exception as e:
                print(f"Failed to log best model to Neptune: {e}")

        # Save checkpoint at intervals (every `save_every` epochs)
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"simplified_unet_epoch_{epoch + 1}.pth")
            torch.save(unet.state_dict(), checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_path}")

            # Log checkpoint to Neptune
            try:
                run[f"model_checkpoint/epoch_{epoch + 1}"].upload(checkpoint_path)
            except Exception as e:
                print(f"Failed to log checkpoint to Neptune: {e}")

        # Log metrics to Neptune
        try:
            run["epoch_loss"].log(avg_loss)
        except Exception as e:
            print(f"Failed to log to Neptune: {e}")

        # Print epoch statistics
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "simplified_unet_final.pth")
    torch.save(unet.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    # Log final model checkpoint to Neptune
    try:
        run["model_checkpoint/final"].upload(final_model_path)
    except Exception as e:
        print(f"Failed to log final model to Neptune: {e}")




def save_image(tensor, path):
    """
    Save a PyTorch tensor as an image to a file.
    :param tensor: PyTorch tensor of shape (C, H, W).
    :param path: File path to save the image.
    """
    image = ToPILImage()(tensor.cpu().squeeze(0))
    image.save(path)



def test_unet_save(unet, dataloader, output_dir="save_images"):
    """
    Test the UNet and save results to a folder.
    :param unet: Trained UNet model.
    :param dataloader: DataLoader for the dataset.
    :param output_dir: Directory to save visualized images.
    """
    os.makedirs(output_dir, exist_ok=True)
    unet.eval()

    with torch.no_grad():
        for idx, (noisy_image, clean_image) in enumerate(dataloader):
            if idx >= 10:  # Test on only 10 examples
                break
            noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)
            vector_1, vector_2, vector_3 = extract_embeddings(noisy_image)
            output = unet(vector_1, vector_2, vector_3, noisy_image)
            save_image(noisy_image[0], os.path.join(output_dir, f"noisy_{idx}.png"))
            save_image(output[0], os.path.join(output_dir, f"output_{idx}.png"))
            save_image(clean_image[0], os.path.join(output_dir, f"ground_truth_{idx}.png"))

    print(f"Test images saved to {output_dir}")

def save_model(unet, path="pretrained/simplified_unet.pth"):
    """
    Save the trained UNet model weights.
    :param unet: The trained UNet model.
    :param path: The file path where to save the weights.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(unet.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    # Ensure the device is defined
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the UNet model
    simplified_unet = SimplifiedUNet(input_channels=3, embedding_dim=512, output_channels=3).to(device)

    # Define optimizer and combined loss
    optimizer = torch.optim.Adam(simplified_unet.parameters(), lr=1e-5)

    perceptual_loss = PerceptualLoss(device=device)
    
    def combined_loss(output, target):
        mse_loss = nn.MSELoss()(output, target)
        perceptual_loss_value = perceptual_loss(output, target)
        normalized_perceptual_loss = perceptual_loss_value / perceptual_loss_value.item()
        return mse_loss + 0.1 * normalized_perceptual_loss


    criterion = combined_loss

    # Dataset and DataLoader
    llq_root = "universal/train/LLQ"
    gt_root = "universal/train/GT"
    dataset = ImageNoiseDataset(llq_root, gt_root)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Train the model
    print("Training the UNet...")
    train_unet(
        unet=simplified_unet,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=1000,  # Total number of epochs
        checkpoint_dir="pretrained",
        save_every=100  # Save model every 100 epochs
    )



    save_model(simplified_unet, path="pretrained/daclip_simplified_unet.pth")

    print("Testing the UNet and saving images...")
    test_unet_save(simplified_unet, dataloader, output_dir="save_images")

