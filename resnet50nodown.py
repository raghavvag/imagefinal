

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Constants for image size limitations
MAX_IMAGE_DIM = 1536
PROCESSING_CHUNK = 1024

class LinearAcrossChannels(nn.Linear):
    """Custom linear layer that operates across channels"""
    
    def __init__(self, input_features: int, output_features: int, use_bias: bool = True) -> None:
        super(LinearAcrossChannels, self).__init__(input_features, output_features, use_bias)

    def forward(self, tensor):
        # Reshape for matrix multiplication
        output_dimensions = [tensor.shape[0], tensor.shape[2], tensor.shape[3], self.out_features]
        reshaped = tensor.permute(0, 2, 3, 1).reshape(-1, self.in_features)
        
        # Apply linear transformation
        transformed = reshaped.matmul(self.weight.t())
        if self.bias is not None:
            transformed = transformed + self.bias[None, :]
            
        # Reshape back to original format
        result = transformed.view(output_dimensions).permute(0, 3, 1, 2)
        return result

    
def create_3x3_conv(in_channels, out_channels, stride=1):
    """Creates a 3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=3, 
        stride=stride, 
        padding=1, 
        bias=False
    )

def create_1x1_conv(in_channels, out_channels, stride=1):
    """Creates a 1x1 convolution"""
    return nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=1, 
        stride=stride, 
        bias=False
    )

class ResidualBlock(nn.Module):
    """Bottleneck block for ResNet"""
    expansion_factor = 4

    def __init__(self, input_channels, internal_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        
        # First 1x1 convolution to reduce channels
        self.reduction_block = nn.Sequential(
            create_1x1_conv(input_channels, internal_channels),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution for spatial processing
        self.spatial_block = nn.Sequential(
            create_3x3_conv(internal_channels, internal_channels, stride),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 convolution to expand channels
        self.expansion_block = nn.Sequential(
            create_1x1_conv(internal_channels, internal_channels * self.expansion_factor),
            nn.BatchNorm2d(internal_channels * self.expansion_factor)
        )
        
        self.shortcut = shortcut
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        # Process through the three blocks
        out = self.reduction_block(x)
        out = self.spatial_block(out)
        out = self.expansion_block(out)

        # Apply shortcut connection if needed
        if self.shortcut is not None:
            residual = self.shortcut(x)

        # Add residual connection and apply activation
        out += residual
        out = self.activation(out)

        return out


class EnhancedResNet(nn.Module):
    """ResNet architecture with customizable parameters"""

    def __init__(self, block_type, block_counts, output_classes=1, initial_stride=2):
        super(EnhancedResNet, self).__init__()
        self.current_channels = 64
        
        # Initial processing block
        self.entry_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=initial_stride, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=initial_stride, padding=1)
        )
        
        # Residual blocks
        self.stage1 = self._construct_stage(block_type, 64, block_counts[0])
        self.stage2 = self._construct_stage(block_type, 128, block_counts[1], stride=2)
        self.stage3 = self._construct_stage(block_type, 256, block_counts[2], stride=2)
        self.stage4 = self._construct_stage(block_type, 512, block_counts[3], stride=2)

        # Output block
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dimension = 512 * block_type.expansion_factor
        self.classifier = LinearAcrossChannels(self.feature_dimension, output_classes)

        # Initialize weights
        self._initialize_parameters()

        # Image transformation pipeline
        self.preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _initialize_parameters(self):
        """Initialize model parameters"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _construct_stage(self, block_type, channels, num_blocks, stride=1):
        """Construct a stage of the network with multiple residual blocks"""
        shortcut = None
        if stride != 1 or self.current_channels != channels * block_type.expansion_factor:
            shortcut = nn.Sequential(
                create_1x1_conv(self.current_channels, channels * block_type.expansion_factor, stride),
                nn.BatchNorm2d(channels * block_type.expansion_factor),
            )

        layers = []
        # First block may have a stride and shortcut
        layers.append(block_type(self.current_channels, channels, stride, shortcut))
        
        # Update channel count for subsequent blocks
        self.current_channels = channels * block_type.expansion_factor
        
        # Add remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block_type(self.current_channels, channels))

        return nn.Sequential(*layers)

    def update_output_layer(self, output_classes):
        """Change the number of output classes"""
        self.classifier = LinearAcrossChannels(self.feature_dimension, output_classes)
        torch.nn.init.normal_(self.classifier.weight.data, 0.0, 0.02)
        return self
        
    def modify_input_channels(self, num_channels):
        """Change the number of input channels"""
        current_weights = self.entry_block[0].weight.data
        original_channels = int(current_weights.shape[1])
        
        if num_channels > original_channels:
            # Repeat weights to match new channel count
            repetitions = num_channels // original_channels
            if (repetitions * original_channels) < num_channels:
                repetitions += 1
            new_weights = current_weights.repeat(1, repetitions, 1, 1) / repetitions
        elif num_channels == original_channels:
            return self
        
        # Slice to get exact number of channels
        new_weights = new_weights[:, :num_channels, :, :]
        print(self.entry_block[0].weight.data.shape, '->', new_weights.shape)
        self.entry_block[0].weight.data = new_weights
        
        return self
    
    def extract_features(self, x):
        """Extract features without classification"""
        x = self.entry_block(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.extract_features(x)
        x = self.global_pooling(x)
        x = self.classifier(x)
        return x
            
    def process_image(self, pil_image):
        """Process a PIL image through the model"""
        device = self.entry_block[0].weight.device
        
        # Handle large images by processing in chunks
        if (pil_image.size[0] > MAX_IMAGE_DIM) and (pil_image.size[1] > MAX_IMAGE_DIM):
            import numpy as np
            print('Processing large image:', pil_image.size)
            
            with torch.no_grad():
                img = self.preprocessing(pil_image)
                predictions = []
                weights = []
                
                # Process image in chunks
                for y_pos in range(0, img.shape[-2], PROCESSING_CHUNK):
                    for x_pos in range(0, img.shape[-1], PROCESSING_CHUNK):
                        # Extract chunk
                        chunk = img[..., 
                                    y_pos:min(y_pos + PROCESSING_CHUNK, img.shape[-2]),
                                    x_pos:min(x_pos + PROCESSING_CHUNK, img.shape[-1])]
                        
                        # Process chunk
                        pred = torch.squeeze(self(chunk.to(device)[None, :, :, :])).cpu().numpy()
                        weight = chunk.shape[-2] * chunk.shape[-1]  # Weight by area
                        
                        predictions.append(pred)
                        weights.append(weight)
            
            # Weighted average of predictions
            final_prediction = np.mean(np.asarray(predictions) * np.asarray(weights)) / np.mean(weights)
        else:
            # Process normal-sized image
            with torch.no_grad():
                final_prediction = torch.squeeze(
                    self(self.preprocessing(pil_image).to(device)[None, :, :, :])
                ).cpu().numpy()
        
        return final_prediction

def create_resnet50_no_downsampling(device, weights_file, num_classes=1):
    """Create a ResNet-50 model without initial downsampling"""
    model = EnhancedResNet(ResidualBlock, [3, 4, 6, 3], output_classes=num_classes, initial_stride=1)
    model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu'))['model'])
    model = model.to(device).eval()
    return model

# For backward compatibility
resnet50nodown = create_resnet50_no_downsampling


