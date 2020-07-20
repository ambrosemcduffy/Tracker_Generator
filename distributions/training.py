"""
This script is for training the Neural Network.
"""
import argparse
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from model import Network
from data_load import KeypointDataSet
from data_load import Rescale, RandomCrop, Normalize, ToTensor


def data_transform(npy_train, npy_test, batch_size=64, shuffle=True):
    # Setting up our image transforms
    DATA_TRANSFORM = transforms.Compose([Rescale(256),
                                         RandomCrop(224),
                                         Normalize(),
                                         ToTensor()])

    # importing in the Custome dataset
    TRANSFORMED_DATASET_TRAIN = KeypointDataSet(npy_file=npy_train,
                                                root_dir='data/',
                                                transform=DATA_TRANSFORM)
    TRANSFORMED_DATASET_TEST = KeypointDataSet(npy_file=npy_test,
                                               root_dir='data/',
                                               transform=DATA_TRANSFORM)

    # passing in the trainformed data into the dataLoader
    TRAIN_LOADER = DataLoader(TRANSFORMED_DATASET_TRAIN,
                              batch_size=64,
                              shuffle=True,
                              pin_memory=True)
    TEST_LOADER = DataLoader(TRANSFORMED_DATASET_TEST,
                             batch_size=64,
                             shuffle=True,
                             pin_memory=True)
    return TRAIN_LOADER, TEST_LOADER


NET = Network()
DEVICE_NAME = torch.cuda.get_device_name()
if torch.cuda.is_available():
    print('GPU DETECTED: {}'.format(DEVICE_NAME))
    NET = Network().cuda()
else:
    print('Using CPU')

print("Network Architecture\n{}".format(NET))


def train(train_loader, test_loader, epochs=10, learn_rate=0.001, print_every=100):
    """
    Args:
        epochs: 10
        learn_rate: 0.001
    Returns: prediction, N-loss, N-iterations
    """
    NET.train()
    optimizer = optim.Adam(NET.parameters(),
                           lr=learn_rate,
                           amsgrad=True,
                           weight_decay=0)
    criterion = torch.nn.MSELoss()
    error_l = []
    epochs_l = []
    error_l_test = []
    epochs_l_test = []
    epochs = epochs
    print('Training the network')
    for epoch in range(epochs):
        train_running_loss = 0.0
        test_running_loss = 0.0
        for i, data in enumerate(train_loader):
            images = data['image']
            key_pts = data['keypoints']
            key_pts = key_pts.view(key_pts.size(0), -1)
            # pylint: disable=maybe-no-member
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)
            if torch.cuda.is_available():
                images = images.cuda()
                key_pts = key_pts.cuda()
            output = NET(images)
            loss = torch.sqrt(criterion(output, key_pts) + 1e-6)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item() * images.size(0)
        for i, data_test in enumerate(test_loader):
            images_test = data_test['image']
            key_pts_test = data_test['keypoints']
            key_pts_test = key_pts_test.view(key_pts_test.size(0), -1)
            # pylint: disable=maybe-no-member
            key_pts_test = key_pts_test.type(torch.FloatTensor)
            images_test = images_test.type(torch.FloatTensor)
            if torch.cuda.is_available():
                images_test = images_test.cuda()
                key_pts_test = key_pts_test.cuda()
            output_test = NET(images_test)
            loss_test = torch.sqrt(criterion(output_test, key_pts_test) + 1e-6)
            test_running_loss += loss_test.item() * images_test.size(0)
        if i % print_every == 0:
            print("Epochs-{} Batch {} Loss_train: {} loss_test: {}".format(epoch, i+1,train_running_loss/len(train_loader), test_running_loss/len(test_loader)))
            error_l.append(train_running_loss/print_every)
            error_l_test.append(test_running_loss/print_every)
            epochs_l.append(epoch+1)
            epochs_l_test.append(epoch+1)
    print('Training End')
    print('Saving out model...\n\n')
    torch.save(NET.state_dict(), 'saved_models/_model_save.pt')
    return output, error_l, error_l_test, epochs_l, epochs_l_test, NET


def plot_loss(epochs_list, loss_list):
    """
    Args:
        epochs_list: number of epochs
        loss_list: number of erros over time
    Return: None v
    """
    print('plotting loss')
    plt.plot(epochs_list, loss_list)
    plt.title('LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.tight_layout()
    plt.show()
    return None


def inference_test(train_loader, idx, Net):
    """
    This function is used for inference to see if the model works
    Args:
        idx: index Number
        Returns: None
    """
    NET.eval()
    data = iter(train_loader).next()
    inputs = data['image']
    img = inputs[idx]
    img = np.expand_dims(img, axis=0)
    # pylint: disable=maybe-no-member
    img_tensor = torch.FloatTensor(img)
    img_tensor = img_tensor.reshape(img.shape[0], 1, 224, 224)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    output = NET(img_tensor)
    output = output * 50. + 100.
    output = output.reshape(output.shape[0], 5, 2)
    output = output.cpu()
    output = output.detach().numpy()
    img_tensor = img_tensor.reshape(img.shape[0], 224, 224, 1)
    img_tensor = img_tensor.cpu()
    img_tensor = img_tensor.detach().numpy()
    img_tensor = img_tensor
    plt.imshow(np.squeeze(img_tensor), origin='lower ', cmap='gray')
    plt.scatter(output[0][:, 0], output[0][:, 1], marker='x')
    return None


train_l, test_l = data_transform('data/train_data/data.npz',
                                 'data/train_data/data_test.npz',
                                 batch_size=128 ,
                                 shuffle=True)

output, error_l, error_l_test, epochs_l, epochs_l_test, NET = train(train_l,
                                                                    test_l,
                                                                    700,
                                                                    learn_rate=0.001,
                                                                    print_every=100)

'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='file path to data')
    parser.add_argument('-b', type=int, help='batch size', default=64)
    parser.add_argument('-s', type=bool, help='shuffle True or False',
                        default=True)
    parser.add_argument('-e', type=int, help=' number of epochs', default=10)
    parser.add_argument('-lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('-p', type=int, help='print every num', default=100)
    args = parser.parse_args()
    train_loader, testloader = data_transform(args.f, args.b, args.s)
    pred, loss_list, epoch_list, net = train(train_loader,
                                             args.e,
                                             args.lr,
                                             args.p)


if __name__ == '__main__':
    main()
'''