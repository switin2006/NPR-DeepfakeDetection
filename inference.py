import torch
import torchvision.transforms as transforms
from PIL import Image
from options.test_options import TestOptions
from networks.trainer import Trainer
import os

def preprocess_image(image_path, opt):
    """
    Loads and preprocesses a single image.

    Args:
        image_path (str): The path to the image file.
        opt (object): The options object containing preprocessing parameters.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except IOError:
        print(f"Error: Cannot open image file {image_path}")
        return None

    # Define the image transformations based on the training script
    transform = transforms.Compose([
        transforms.Resize((opt.loadSize, opt.loadSize)),
        transforms.CenterCrop(opt.cropSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations and add a batch dimension
    return transform(image).unsqueeze(0)

def main():
    """
    Main function to run the inference on a single image.
    """
    # --- 1. Parse Options and Load Model ---
    opt = TestOptions().parse()
    opt.isTrain = False

    # Load the model structure
    model = Trainer(opt)
    model.eval()

    # Load the trained weights
    if not os.path.exists(opt.resume):
        raise FileNotFoundError(f"FATAL: The specified model path does not exist: {opt.resume}")

    state_dict = torch.load(opt.resume, map_location=model.device)
    model.model.load_state_dict(state_dict)
    print(f"--- Model loaded from: {opt.resume} ---")

    # --- 2. Get Image Path and Preprocess ---
    # Replace 'path/to/your/image.jpg' with the actual path to your image
    image_path = 'path/to/your/image.jpg'
    image_tensor = preprocess_image(image_path, opt)

    if image_tensor is not None:
        # Move the tensor to the appropriate device (CPU or GPU)
        image_tensor = image_tensor.to(model.device)

        # --- 3. Perform Inference ---
        with torch.no_grad():
            output = model.model(image_tensor)
            # Apply sigmoid to get a probability
            prob = torch.sigmoid(output).item()
            # Classify as "Fake" if probability > 0.5, otherwise "Real"
            prediction = "Fake" if prob > 0.5 else "Real"

        # --- 4. Display the Result ---
        print(f"\n--- Inference Result ---")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {prediction} (Probability: {prob:.4f})")
        print("------------------------")

if __name__ == '__main__':
    main()