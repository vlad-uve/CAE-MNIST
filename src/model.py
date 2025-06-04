
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_channels, latent_dim, kernel, stride, padding, use_batch_norm, activation_func):
        super(Encoder, self).__init__()
        # store channel configuration and latent dimension
        self.n_channels = n_channels
        self.latent_dim = latent_dim

        # store conv layer parameters
        self.kernel = kernel      
        self.stride = stride
        self.padding = padding

        # store batch norm usage choice
        self.use_batch_norm = use_batch_norm

        # store activation function choice
        self.activation_func = activation_func

        # Build layers by calling method (can be overridden in child class)
        self._build_layers()


    def _build_layers(self):
        # 3-layer encoder architecture
        self.encn1 = nn.Conv2d(1, self.n_channels[0], self.kernel[0], self.stride, self.padding)
        self.encn2 = nn.Conv2d(self.n_channels[0], self.n_channels[1], self.kernel[1], self.stride, self.padding)
        self.encn3 = nn.Conv2d(self.n_channels[1], self.n_channels[2], self.kernel[2], self.stride, self.padding)

        # optional batch normalization layers after each conv
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(self.n_channels[0])
            self.bn2 = nn.BatchNorm2d(self.n_channels[1])
            self.bn3 = nn.BatchNorm2d(self.n_channels[2])
        else:
            self.bn1 = self.bn2 = self.bn3 = nn.Identity()

        # flatten and fully connected bottleneck layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.n_channels[2] * 4 * 4, self.latent_dim)


    def apply_activation(self, x):
        # safely apply activation function
        if self.activation_func == 'relu':
            return F.relu(x)
        elif self.activation_func == 'leaky_relu':
            return F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_func}")


    def forward(self, input_x):
        # pass through encoding layers with batch norm and activation
        h1 = self.apply_activation(self.bn1(self.encn1(input_x)))
        h2 = self.apply_activation(self.bn2(self.encn2(h1)))
        h3 = self.apply_activation(self.bn3(self.encn3(h2)))

        # return data encoded in latent space with flattening and fully connected layer
        encoded_x = self.fc1(self.flatten(h3))

        return encoded_x



class Decoder(nn.Module):
    def __init__(self, n_channels, latent_dim, kernel, stride, padding, use_batch_norm, activation_func):
        super(Decoder, self).__init__()
        # store channel configuration and latent dimension
        self.n_channels = n_channels
        self.latent_dim = latent_dim

        # store conv layer parameters
        self.kernel = kernel      
        self.stride = stride
        self.padding = padding

        # store batch norm usage choice
        self.use_batch_norm = use_batch_norm

        # store activation function choice
        self.activation_func = activation_func

        # Build layers by calling method (can be overridden in child class)
        self._build_layers()


    def _build_layers(self):
        # fully connected + unflatten to prepare for decoding
        self.fc1 = nn.Linear(self.latent_dim, self.n_channels[2] * 4 * 4)
        self.unflatten = nn.Unflatten(1, (self.n_channels[2], 4, 4))

        # define transposed conv layers for decoding
        self.decn1 = nn.ConvTranspose2d(self.n_channels[2], self.n_channels[1], self.kernel[2], self.stride, self.padding)
        self.decn2 = nn.ConvTranspose2d(self.n_channels[1], self.n_channels[0], self.kernel[1], self.stride, self.padding)
        self.decn3 = nn.ConvTranspose2d(self.n_channels[0], 1, self.kernel[0], self.stride, self.padding)        

        # optional batch normalization layers after each transposed conv
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(self.n_channels[1])
            self.bn2 = nn.BatchNorm2d(self.n_channels[0])
        else:
            self.bn1 = self.bn2 = nn.Identity()


    def apply_activation(self, x):
        # safely apply activation function
        if self.activation_func == 'relu':
            return F.relu(x)
        elif self.activation_func == 'leaky_relu':
            return F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_func}")


    def forward(self, encoded):
        #decoding data from latent space with unflattening of fully connected layer
        x = self.unflatten(self.fc1(encoded))

        # pass through transposed conv layers with batchnorm and activation
        h1 = self.apply_activation(self.bn1(self.decn1(x)))
        h2 = self.apply_activation(self.bn2(self.decn2(h1)))
        h3 = self.decn3(h2)
        decoded_x = F.sigmoid(h3)

        # return decoded data as an image with pixels of [0, 1] range
        return decoded_x


class AutoEncoder(nn.Module):
    def __init__(self, n_channels, latent_dim, use_batch_norm=False, activation_func='relu'):
        super(AutoEncoder, self).__init__()

        # fixed parameters for all conv layers
        kernel = [4, 4, 3]
        stride = 2
        padding = 1

        # define encoder and decoder blocks
        self.encoder = Encoder(n_channels, latent_dim, kernel, stride, padding, use_batch_norm, activation_func)
        self.decoder = Decoder(n_channels, latent_dim, kernel, stride, padding, use_batch_norm, activation_func)

    def forward(self, input_x):
        # encode and decode the input data
        encoded_x = self.encoder(input_x)
        decoded_x = self.decoder(encoded_x)

        return decoded_x, encoded_x
