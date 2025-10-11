import torch
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
from networks.trainer import Trainer
from options.test_options import TestOptions

# opt is now passed as an argument, which is crucial
def preprocess_image(image_path, opt):
    try:
        image = Image.open(image_path).convert('RGB')
    except IOError:
        print(f"Error: Cannot open image file {image_path}")
        return None
    
    # This now uses the CORRECT loadSize and cropSize from the command line
    transform = transforms.Compose([
        transforms.Resize((opt.loadSize, opt.loadSize)),
        transforms.CenterCrop(opt.cropSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform(image).unsqueeze(0)

def predict_single_image(model, image_path, opt): # opt is also passed here
    model.eval()
    device = next(model.parameters()).device # Get device directly from model

    # Pass opt to the preprocessing function
    image_tensor = preprocess_image(image_path, opt)
    if image_tensor is None:
        return

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output_logits = model(image_tensor)
        prob = torch.sigmoid(output_logits).item()
        prediction = "Fake" if prob > 0.5 else "Real"

    print(f"\n--- Prediction ---")
    print(f"Image:      {os.path.basename(image_path)}")
    print(f"Prediction: {prediction}")
    print(f"Confidence of fake: {prob:.4f}")
    print("-----------------")
    return prediction, prob

if __name__ == '__main__':
    # --- THIS IS THE CORRECTED LOGIC ---
    # 1. Initialize TestOptions to get the full parser
    test_options = TestOptions()
    parser = test_options.initialize(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter))
    
    # 2. Add your custom argument
    parser.add_argument('--image_path', type=str, required=True, help='path to the input image')
    
    # 3. Parse ALL arguments at once
    opt = parser.parse_args()
    opt.isTrain = False

    # --- Load model ---
    # Make sure you have made the one-line fix in networks/trainer.py
    model_trainer = Trainer(opt)
    device = model_trainer.device # Use the device from the trainer
    model_trainer.model.to(device)
    model_trainer.model.eval()

    if not os.path.exists(opt.resume):
        raise FileNotFoundError(f"Model weights not found: {opt.resume}")
    
    state_dict = torch.load(opt.resume, map_location=device)
    model_trainer.model.load_state_dict(state_dict)
    print(f"Model loaded from: {opt.resume}")

    # --- Predict ---
    # Pass the fully loaded 'opt' to the prediction function
    predict_single_image(model_trainer.model, opt.image_path, opt)
