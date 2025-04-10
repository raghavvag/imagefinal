import subprocess
import Image
import torch
import argparse
def demo_cnn():
    subprocess.run(["python3", "models/ganimagedetection_image/predictor.py"])

# Modified function structure while maintaining functionality
def process_image(image_path, model, transform):
    """Process an image through the model and return predictions"""
    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        output = model(img_t)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Return top predictions
    return [(labels[idx], prob.item()) for prob, idx in zip(*torch.topk(probabilities, 5))]

# Reorganized main execution block
if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    model.eval()
    
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Image classification demo')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    
    # Run prediction and display results
    predictions = process_image(args.image, model, preprocess)
    print("\nTop 5 predictions:")
    for i, (label, prob) in enumerate(predictions, 1):
        print(f"{i}. {label}: {prob:.4f}")
(demo_cnn())