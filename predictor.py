

import os, torch
import glob
import time
from PIL import Image
from resnet50nodown import resnet50nodown
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.transforms as transforms

if __name__ == '__main__':
    
    weights_path = 'models/ganimagedetection_image/weights/gandetection_resnet50nodown_stylegan2.pth'
    input_folder = './temp'
    output_csv = 'models/ganimagedetection_image/result.txt'
    
    from torch.cuda import is_available as is_available_cuda
    device = 'cuda:0' if is_available_cuda() else 'cpu'
    net = resnet50nodown(device, weights_path)
    
    list_files = sorted(sum([glob.glob(os.path.join(input_folder,'*.'+x)) for x in ['jpg','JPG','jpeg','JPEG','png','PNG']], list()))
    num_files = len(list_files)
    
    # print('GAN IMAGE DETECTION')
    # print('START')
    
    with open(output_csv, 'w') as fid:
        # fid.write('filename,logit,probability,time\n')
        fid.flush()
        for index, filename in enumerate(list_files):
            # print('%5d/%d' % (index, num_files), end='\r')
            tic = time.time()
            img = Image.open(filename).convert('RGB')
            img.load()
            logit = net.apply(img)
            probability = torch.sigmoid(torch.tensor(logit)).item()  # For binary classification
            toc = time.time()
            
            # fid.write('%s,%f,%f,%f\n' % (filename, logit, probability, toc-tic))
            fid.write('%f' % (probability))
            fid.flush()

    # print('\nDONE')
    # print('OUTPUT: %s' % output_csv)


class ImagePredictor:
    """Class for making predictions with trained models"""
    def __init__(self, model_path=None, device=None):
        # Initialize with optional model path and device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model(model_path)
        self.transform = self._get_transforms()
        self.class_names = self._load_class_names()
    
    def _initialize_model(self, model_path):
        """Set up the model with weights"""
        model = resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self._load_class_names()))
        
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self):
        """Define image transformations"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=5):
        """Make prediction on an image"""
        # Process image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            
        # Get top k predictions
        top_probs, top_indices = probs.topk(top_k)
        
        # Convert to list of (class_name, probability) tuples
        results = []
        for i in range(top_k):
            results.append((
                self.class_names[top_indices[0][i].item()],
                top_probs[0][i].item()
            ))
        
        return results
