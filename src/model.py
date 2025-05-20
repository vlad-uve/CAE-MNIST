
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_channels, stride, padding, latent_dim, use_batch_norm, activation_func):
        super(Encoder, self).__init__()

        # store activation function choice
        self.activation_func = activation_func

        # define convolutional layers for encoding
        self.encn1 = nn.Conv2d(1, n_channels[0], 4, stride=stride, padding=padding)
        self.encn2 = nn.Conv2d(n_channels[0], n_channels[1], 4, stride=stride, padding=padding)
        self.encn3 = nn.Conv2d(n_channels[1], n_channels[2], 3, stride=stride, padding=padding)

        # optional batch normalization layers after each conv
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(n_channels[0])
            self.bn2 = nn.BatchNorm2d(n_channels[1])
            self.bn3 = nn.BatchNorm2d(n_channels[2])
        else:
            self.bn1 = self.bn2 = self.bn3 = nn.Identity()

        # flatten and fully connected bottleneck layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_channels[2] * 4 * 4, latent_dim)

    def apply_activation(self, x):
        # safely apply activation function
        if self.activation_func == 'relu':
            return F.relu(x)
        elif self.activation_func == 'leaky_relu':
            return F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_func}")

    def forward(self, input_x):
        # pass through encoding layers with batchnorm and activation
        x = self.encn1(input_x)
        x = self.bn1(x)
        x = self.apply_activation(x)

        x = self.encn2(x)
        x = self.bn2(x)
        x = self.apply_activation(x)

        x = self.encn3(x)
        x = self.bn3(x)
        x = self.apply_activation(x)

        # return data encoded in latent space with flattening and fully connected layer
        x = self.flatten(x)
        encoded_x = self.fc1(x)

        return encoded_x


class Decoder(nn.Module):
    def __init__(self, n_channels, stride, padding, latent_dim, use_batch_norm, activation_func):
        super(Decoder, self).__init__()

        # store activation function choice
        self.activation_func = activation_func

        # fully connected + unflatten to prepare for decoding
        self.fc1 = nn.Linear(latent_dim, n_channels[2] * 4 * 4)
        self.unflatten = nn.Unflatten(1, (n_channels[2], 4, 4))

        # optional batch normalization layers after each transposed conv
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(n_channels[1])
            self.bn2 = nn.BatchNorm2d(n_channels[0])
        else:
            self.bn1 = self.bn2 = nn.Identity()

        # define transposed conv layers for decoding
        self.decn1 = nn.ConvTranspose2d(n_channels[2], n_channels[1], 3, stride=stride, padding=padding)
        self.decn2 = nn.ConvTranspose2d(n_channels[1], n_channels[0], 4, stride=stride, padding=padding)
        self.decn3 = nn.ConvTranspose2d(n_channels[0], 1, 4, stride=stride, padding=padding)

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
        x = self.fc1(encoded)
        x = self.unflatten(x)

        # pass through transposed conv layers with batchnorm and activation
        x = self.decn1(x)
        x = self.bn1(x)
        x = self.apply_activation(x)

        x = self.decn2(x)
        x = self.bn2(x)
        x = self.apply_activation(x)

        x = self.decn3(x)
        decoded_x = F.sigmoid(x)

        # return decoded data as an image with pixels of [0, 1] range
        return decoded_x


class AutoEncoder(nn.Module):
    def __init__(self, n_channels, latent_dim, use_batch_norm=False, activation_func='relu'):
        super(AutoEncoder, self).__init__()

        # fixed parameters for all conv layers
        stride = 2
        padding = 1

        # define encoder and decoder blocks
        self.encoder = Encoder(n_channels, stride, padding, latent_dim, use_batch_norm, activation_func)
        self.decoder = Decoder(n_channels, stride, padding, latent_dim, use_batch_norm, activation_func)

    def forward(self, input_x):
        # encode and decode the input data
        encoded_x = self.encoder(input_x)
        decoded_x = self.decoder(encoded_x)

        return decoded_x, encoded_x


class DeepEncoder(nn.Module):
    def __init__(self, n_channels, stride, padding, latent_dim, use_batch_norm, activation_func):
        super(DeepEncoder, self).__init__()

        # store activation function choice
        self.activation_func = activation_func

        # define convolutional layers for encoding
        self.encn1 = nn.Conv2d(1, n_channels[0], 4, stride=stride, padding=padding)
        self.encn2 = nn.Conv2d(n_channels[0], n_channels[1], 4, stride=stride, padding=padding)
        self.encn3 = nn.Conv2d(n_channels[1], n_channels[2], 3, stride=stride, padding=padding)
        self.encn4 = nn.Conv2d(n_channels[2], n_channels[3], 4, stride=stride, padding=padding)

        # optional batch normalization layers after each conv
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(n_channels[0])
            self.bn2 = nn.BatchNorm2d(n_channels[1])
            self.bn3 = nn.BatchNorm2d(n_channels[2])
            self.bn4 = nn.BatchNorm2d(n_channels[3])
        else:
            self.bn1 = self.bn2 = self.bn3 = self.bn4 = nn.Identity()

        # flatten and fully connected bottleneck layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_channels[3] * 2 * 2, latent_dim)

    def apply_activation(self, x):
        # safely apply activation function
        if self.activation_func == 'relu':
            return F.relu(x)
        elif self.activation_func == 'leaky_relu':
            return F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_func}")

    def forward(self, input_x):
        # pass through encoding layers with batchnorm and activation
        x = self.encn1(input_x)
        x = self.bn1(x)
        x = self.apply_activation(x)

        x = self.encn2(x)
        x = self.bn2(x)
        x = self.apply_activation(x)

        x = self.encn3(x)
        x = self.bn3(x)
        x = self.apply_activation(x)

        x = self.encn4(x)
        x = self.bn4(x)
        x = self.apply_activation(x)

        # flatten and pass through fully connected layer
        x = self.flatten(x)
        encoded_x = self.fc1(x)

        return encoded_x


class DeepDecoder(nn.Module):
    def __init__(self, n_channels, stride, padding, latent_dim, use_batch_norm, activation_func):
        super(DeepDecoder, self).__init__()

        # store activation function choice
        self.activation_func = activation_func

        # fully connected + unflatten to prepare for decoding
        self.fc1 = nn.Linear(latent_dim, n_channels[3] * 2 * 2)
        self.unflatten = nn.Unflatten(1, (n_channels[3], 2, 2))

        # define transposed conv layers for decoding
        self.decn1 = nn.ConvTranspose2d(n_channels[3], n_channels[2], 4, stride=stride, padding=padding)
        self.decn2 = nn.ConvTranspose2d(n_channels[2], n_channels[1], 3, stride=stride, padding=padding)
        self.decn3 = nn.ConvTranspose2d(n_channels[1], n_channels[0], 4, stride=stride, padding=padding)
        self.decn4 = nn.ConvTranspose2d(n_channels[0], 1, 4, stride=stride, padding=padding)

        # optional batch normalization layers after each transposed conv (except the last one)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(n_channels[2])
            self.bn2 = nn.BatchNorm2d(n_channels[1])
            self.bn3 = nn.BatchNorm2d(n_channels[0])
        else:
            self.bn1 = self.bn2 = self.bn3 = nn.Identity()

    def apply_activation(self, x):
        # safely apply activation function
        if self.activation_func == 'relu':
            return F.relu(x)
        elif self.activation_func == 'leaky_relu':
            return F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_func}")

    def forward(self, encoded):
        # decoding from latent space
        x = self.fc1(encoded)
        x = self.unflatten(x)

        x = self.decn1(x)
        x = self.bn1(x)
        x = self.apply_activation(x)

        x = self.decn2(x)
        x = self.bn2(x)
        x = self.apply_activation(x)

        x = self.decn3(x)
        x = self.bn3(x)
        x = self.apply_activation(x)

        x = self.decn4(x)
        decoded_x = F.sigmoid(x)

        return decoded_x


class DeepAutoEncoder(nn.Module):
    def __init__(self, n_channels, latent_dim, use_batch_norm=False, activation_func='relu'):
        super(DeepAutoEncoder, self).__init__()

        # fixed parameters for all conv layers
        stride = 2
        padding = 1

        # define encoder and decoder blocks
        self.encoder = DeepEncoder(n_channels, stride, padding, latent_dim, use_batch_norm, activation_func)
        self.decoder = DeepDecoder(n_channels, stride, padding, latent_dim, use_batch_norm, activation_func)

    def forward(self, input_x):
        # encode and decode the input data
        encoded_x = self.encoder(input_x)
        decoded_x = self.decoder(encoded_x)

        return decoded_x, encoded_x


class ShallowEncoder(nn.Module):
    def __init__(self, n_channels, stride, padding, latent_dim, use_batch_norm, activation_func):
        super(ShallowEncoder, self).__init__()

        # store activation function choice
        self.activation_func = activation_func

        # define convolutional layers for encoding
        self.encn1 = nn.Conv2d(1, n_channels[0], 4, stride=stride, padding=padding)
        self.encn2 = nn.Conv2d(n_channels[0], n_channels[1], 4, stride=stride, padding=padding)

        # optional batch normalization layers after each conv
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(n_channels[0])
            self.bn2 = nn.BatchNorm2d(n_channels[1])
        else:
            self.bn1 = self.bn2 = nn.Identity()

        # flatten and fully connected bottleneck layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_channels[1] * 7 * 7, latent_dim)

    def apply_activation(self, x):
        # safely apply activation function
        if self.activation_func == 'relu':
            return F.relu(x)
        elif self.activation_func == 'leaky_relu':
            return F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_func}")

    def forward(self, input_x):
        # pass through encoding layers with batchnorm and activation
        x = self.encn1(input_x)
        x = self.bn1(x)
        x = self.apply_activation(x)

        x = self.encn2(x)
        x = self.bn2(x)
        x = self.apply_activation(x)

        # flatten and pass through fully connected layer
        x = self.flatten(x)
        encoded_x = self.fc1(x)

        return encoded_x


class ShallowDecoder(nn.Module):
    def __init__(self, n_channels, stride, padding, latent_dim, use_batch_norm, activation_func):
        super(ShallowDecoder, self).__init__()

        # store activation function choice
        self.activation_func = activation_func

        # fully connected + unflatten to prepare for decoding
        self.fc1 = nn.Linear(latent_dim, n_channels[1] * 7 * 7)
        self.unflatten = nn.Unflatten(1, (n_channels[1], 7, 7))

        # define transposed conv layers for decoding
        self.decn1 = nn.ConvTranspose2d(n_channels[1], n_channels[0], 4, stride=stride, padding=padding)
        self.decn2 = nn.ConvTranspose2d(n_channels[0], 1, 4, stride=stride, padding=padding)

        # optional batch normalization layers (no batch norm after final layer)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(n_channels[0])
        else:
            self.bn1 = nn.Identity()

    def apply_activation(self, x):
        # safely apply activation function
        if self.activation_func == 'relu':
            return F.relu(x)
        elif self.activation_func == 'leaky_relu':
            return F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_func}")

    def forward(self, encoded):
        # decode from latent space
        x = self.fc1(encoded)
        x = self.unflatten(x)

        x = self.decn1(x)
        x = self.bn1(x)
        x = self.apply_activation(x)

        x = self.decn2(x)
        decoded_x = F.sigmoid(x)

        return decoded_x


class ShallowAutoEncoder(nn.Module):
    def __init__(self, n_channels, latent_dim, use_batch_norm=False, activation_func='relu'):
        super(ShallowAutoEncoder, self).__init__()

        # fixed parameters for all conv layers
        stride = 2
        padding = 1

        # define encoder and decoder blocks
        self.encoder = ShallowEncoder(n_channels, stride, padding, latent_dim, use_batch_norm, activation_func)
        self.decoder = ShallowDecoder(n_channels, stride, padding, latent_dim, use_batch_norm, activation_func)

    def forward(self, input_x):
        # encode and decode the input data
        encoded_x = self.encoder(input_x)
        decoded_x = self.decoder(encoded_x)

        return decoded_x, encoded_x
