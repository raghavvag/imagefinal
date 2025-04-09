

import os
import glob
import time
import argparse
from PIL  import Image
from resnet50nodown import resnet50nodown

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="This script tests the network on an image folder and collects the results in a CSV file.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights_path', '-m', type=str , default='./weights/gandetection_resnet50nodown_stylegan2.pth', help='weights path of the network')
    parser.add_argument('--input_folder', '-i', type=str , default='./example_images', help='input folder with PNG and JPEG images')
    parser.add_argument('--output_csv'  , '-o', type=str , default=None, help='output CSV file')
    config = parser.parse_args()
    weights_path = config.weights_path
    input_folder = config.input_folder
    output_csv = config.output_csv
    
    from torch.cuda import is_available as is_available_cuda
    device = 'cuda:0' if is_available_cuda() else 'cpu'
    net = resnet50nodown(device, weights_path)
    
    if output_csv is None:
        output_csv = 'out.'+os.path.basename(input_folder)+'.csv'
    
    list_files = sorted(sum([glob.glob(os.path.join(input_folder,'*.'+x)) for x in ['jpg','JPG','jpeg','JPEG','png','PNG']], list()))
    num_files = len(list_files)
    
    print('GAN IMAGE DETECTION')
    print('START')
    
    with open(output_csv,'w') as fid:
        fid.write('filename,logit,time\n')
        fid.flush()
        for index, filename in enumerate(list_files):
            print('%5d/%d'%(index, num_files), end='\r')
            tic = time.time()
            img = Image.open(filename).convert('RGB')
            img.load()
            logit = net.apply(img)
            toc = time.time()
            
            fid.write('%s,%f,%f\n' %(filename, logit, toc-tic))
            fid.flush()

    print('\nDONE')
    print('OUTPUT: %s' % output_csv)


# Restructured training loop
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass and loss calculation
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

# Refactored validation function
def validate(model, dataloader, criterion, device):
    """Evaluate model on validation data"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return val_loss / len(dataloader.dataset), 100. * correct / total
    
