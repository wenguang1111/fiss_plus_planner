import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=64, use_resnet=True):
        super().__init__()
        
        if use_resnet:
            # Load ResNet18
            resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            
            # Remove the final classification layer
            self.features = nn.Sequential(*list(resnet18.children())[:-1])
            
            # ResNet18 outputs 512 channels
            resnet_output_dim = 512
            
            # Add projection layer to reduce to desired output_dim
            self.projection = nn.Sequential(
                nn.Linear(resnet_output_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )
        else:
            # Fallback to original CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),  # 128->64
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),  # 64->32
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),  # 32->16
            )
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.projection = nn.Linear(128, output_dim)
        
        self.use_resnet = use_resnet
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        
        if self.use_resnet:
            # ResNet already includes global avg pooling, but apply it again for consistency
            x = self.global_pool(x)
        else:
            x = self.global_pool(x)
        
        x = self.flatten(x)
        x = self.projection(x)
        return x
    
    
class CVAE(nn.Module):
    def __init__(self, X_dim, c_dim, z_dim, h_Q_dim=512, h_P_dim=512, cnn_feature_dim=64):
        super(CVAE, self).__init__()
        
        # CNN for image feature extraction with ResNet18 backbone
        self.cnn_extractor = CNNFeatureExtractor(output_dim=cnn_feature_dim, use_resnet=True)
        
        # Encoder (Q network) with BatchNorm and Dropout
        self.encoder = nn.Sequential(
            nn.Linear(X_dim + c_dim, h_Q_dim),
            nn.BatchNorm1d(h_Q_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h_Q_dim, h_Q_dim // 2),
            nn.BatchNorm1d(h_Q_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h_Q_dim // 2, h_Q_dim // 4),
            nn.BatchNorm1d(h_Q_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.encoder_fc_mu = nn.Linear(h_Q_dim // 4, z_dim)
        self.encoder_fc_logvar = nn.Linear(h_Q_dim // 4, z_dim)
        
        # Decoder (P network) - Symmetric to encoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + c_dim, h_P_dim // 4),
            nn.BatchNorm1d(h_P_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h_P_dim // 4, h_P_dim // 2),
            nn.BatchNorm1d(h_P_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h_P_dim // 2, h_P_dim),
            nn.BatchNorm1d(h_P_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h_P_dim, X_dim)  # No activation on output
        )

    def encode(self, x, c, img):
        img_features = self.cnn_extractor(img)
        c = torch.cat([c, img_features], dim=1)
        xc = torch.cat([x, c], dim=1)
        
        h = self.encoder(xc)
        mu = self.encoder_fc_mu(h)
        logvar = self.encoder_fc_logvar(h)

        return mu, logvar, img_features

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c, img_features):
        c = torch.cat([c, img_features], dim=1)
        zc = torch.cat([z, c], dim=1)
        y = self.decoder(zc)

        return y

    def forward(self, x, c, img):
        mu, logvar, img_features = self.encode(x, c, img)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z, c, img_features)
        return y, mu, logvar


def cvae_loss_function(y_pred, y_true, mu, logvar, kl_beta=1e-4):
    recon_loss = F.mse_loss(y_pred, y_true)

    # sum over latent dim
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # mean over batch
    kl_loss = kl_loss.mean()
    return recon_loss, kl_beta * kl_loss


if __name__ == "__main__":
    # Example usage
    X_dim, c_dim, z_dim = 3, 70, 64
    model = CVAE(X_dim=X_dim, c_dim=c_dim, z_dim=z_dim)

    # Test forward pass
    x = torch.randn(4, X_dim)
    c = torch.randn(4, 6)  # Original condition (without img features)
    img = torch.randn(4, 3, 128, 128)
    
    y_pred, mu, logvar = model(x, c, img)
    print(f"Input x shape: {x.shape}")
    print(f"Condition c shape: {c.shape}")
    print(f"Image shape: {img.shape}")
    print(f"Output y_pred shape: {y_pred.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print("Model initialized successfully with ResNet18 backbone!")