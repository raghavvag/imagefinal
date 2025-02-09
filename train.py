import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet50nodown import resnet50nodown
from tqdm import tqdm
import os

# Configuration
CFG = {
    'batch_size': 8,  # Reduce if OOM errors occur
    'epochs': 10,
    'lr': 1e-4,
    'input_size': 256,
    'num_workers': 4,
    'weight_decay': 1e-5,
    'checkpoint_dir': 'checkpoints/',
    'weights_path': 'weights/gandetection_resnet50nodown_stylegan2.pth'
}

# Setup Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((CFG['input_size'], CFG['input_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CFG['input_size'], CFG['input_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data Loaders
def create_loaders():
    train_ds = datasets.ImageFolder(r'C:\Users\aarya\Downloads\AI Model\Image\dataset\train', transform=train_transform)
    val_ds = datasets.ImageFolder(r'C:\Users\aarya\Downloads\AI Model\Image\dataset\val', transform=val_transform)
    
    return (
        DataLoader(train_ds, CFG['batch_size'], shuffle=True,
                  num_workers=CFG['num_workers'], pin_memory=True),
        DataLoader(val_ds, CFG['batch_size'], shuffle=False,
                 num_workers=CFG['num_workers'], pin_memory=True)
    )

# Initialize Model
def init_model(device):
    model = resnet50nodown(device, CFG['weights_path'], num_classes=1)
    model = model.to(device).train()
    return model

# Training Function
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG['checkpoint_dir'], exist_ok=True)
    
    model = init_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], 
                           weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    train_loader, val_loader = create_loaders()
    
    best_val_loss = float('inf')
    for epoch in range(CFG['epochs']):
        # Training Phase
        model.train()
        train_loss = 0
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        
        # Validation Phase
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                
                outputs = model(inputs)
                val_loss += criterion(outputs.squeeze(), labels).item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds.squeeze() == labels).sum().item()
                total += labels.size(0)
        
        # Epoch Statistics
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        val_acc = 100*correct/total
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{CFG['epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2f}%")
        
        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 
                      f"{CFG['checkpoint_dir']}/best_model.pth")
            print("Saved best model!")
            
        # Save Regular Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': avg_val_loss
        }, f"{CFG['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    train_model()