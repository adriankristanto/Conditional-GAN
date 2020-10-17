import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import models.DCGAN as DCGAN
import os

MAIN_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = 'cgan-model-bce-epoch10.pth'
EPOCH = 200
SAVE_INTERVAL = 20
# for generation
SAMPLE_INTERVAL = 469
SAMPLE_SIZE = 64
# shape of a single image
INPUT_IMG_SHAPE = (1, 32, 32)

# Hyperparameters
BATCH_SIZE = 128
Z_DIM = 100
GENERATOR_LR = 0.0002
DISCRIMINATOR_LR = 0.0002
BETAS = (0.5, 0.999)

if __name__ == "__main__":
    
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}\n", flush=True)

    # directory setup
    if 'data' not in os.listdir(MAIN_DIR):
        print('creating data directory...', flush=True)
        os.mkdir(MAIN_DIR + 'data')
    if 'generated_images' not in os.listdir(MAIN_DIR):
        print('creating generated_images directory...', flush=True)
        os.mkdir(MAIN_DIR + 'generated_images')
    if 'logs' not in os.listdir(MAIN_DIR):
        print('creating logs directory...', flush=True)
        os.mkdir(MAIN_DIR + 'logs')
    if 'saved_models' not in os.listdir(MAIN_DIR):
        print('creating saved_models directory...', flush=True)
        os.mkdir(MAIN_DIR + 'saved_models')
    
    # 1. load the dataset
    DATA_PATH = MAIN_DIR + 'data/'
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = datasets.MNIST(root=DATA_PATH, download=True, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # total unique classes in MNIST dataset
    num_classes = len(trainset.classes)

    print(f"""
Total training data: {len(trainset)}
Total unique classes: {num_classes}
    """, flush=True)

    # input image sample
    # data_iter = iter(trainset)
    # img, label = next(data_iter)
    # torchvision.utils.save_image(img, 'sample.png')

    # 2. instantiate the model
    # Generator G
    INPUT_DIM = Z_DIM + num_classes 
    # we will append a one-hot vector to the z_dim
    # where the one-hot vector is of num_classes dimension
    G = DCGAN.Generator(
        channels=[INPUT_DIM, 256, 128, 64, 1],
        kernel_sizes=[None, 4, 4, 4, 4],
        strides=[None, 1, 2, 2, 2],
        paddings=[None, 0, 1, 1, 1],
        batch_norm=True,
        activations=[nn.ReLU(), nn.Tanh()]
    )

    # Discriminator D
    INPUT_CHANNEL = INPUT_IMG_SHAPE[0] + num_classes
    # we will append (batch_size, num_classes, 28, 28), which is similar to the one-hot vector
    # to (batch_size, 1, 28, 28) which is the input batch of MNIST
    # for example, if dimension 3 of the one-hot vector has the value 1, then we create a tensor of size (1, 28, 28) full on 1s for index 3
    # and the other 9 tensor will be filled with 0s
    D = DCGAN.Discriminator(
        channels=[INPUT_CHANNEL, 64, 128, 256, 1],
        kernel_sizes=[None, 4, 4, 4, 4],
        strides=[None, 2, 2, 2, 1],
        paddings=[None, 1, 1, 1, 0],
        batch_norm=False,
        activation=nn.LeakyReLU(0.2),
    )

    print(f"""
Generator G:
{G}
Critic D:
{D}
""", flush=True)

    # setting for training using multiple GPUs
    multigpu = False
    if torch.cuda.device_count() > 1:
        print(f'Number of GPUs: {torch.cuda.device_count()}\n', flush=True)
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
        multigpu = True
    
    # move the models to the device that we use for training
    G.to(device)
    D.to(device)

    # 3. define the loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # 4. define the optimisers
    g_optim = optim.Adam(G.parameters(), lr=GENERATOR_LR, betas=BETAS)
    d_optim = optim.Adam(D.parameters(), lr=DISCRIMINATOR_LR, betas=BETAS)

    # 5. train the model
    MODEL_DIRPATH = MAIN_DIR + 'saved_models/'
    GENERATED_DIRPATH = MAIN_DIR + 'generated_images/'
    SAVED_MODEL_NAME = MODEL_DIRPATH + CONTINUE_TRAIN_NAME

    next_epoch = 0
    if CONTINUE_TRAIN:
        checkpoint = torch.load(SAVED_MODEL_NAME)
        G.load_state_dict(checkpoint.get('G_state_dict'))
        D.load_state_dict(checkpoint.get('D_state_dict'))
        g_optim.load_state_dict(checkpoint.get('g_optim_state_dict'))
        d_optim.load_state_dict(checkpoint.get('d_optim_state_dict'))
        next_epoch = checkpoint.get('epoch')
    
    def save_training_progress(G, D, g_optimizer, d_optimizer, epoch, target_dir):
        torch.save({
            'epoch' : epoch + 1,
            'G_state_dict' : G.state_dict(),
            'D_state_dict' : D.state_dict(),
            'g_optim_state_dict' : g_optimizer.state_dict(),
            'd_optim_state_dict' : d_optimizer.state_dict()
        }, target_dir)

    for epoch in range(next_epoch, EPOCH):

        trainloader = tqdm(trainloader)

        for i, train_data in enumerate(trainloader):
            D.train()
            G.train()

            reals, labels = train_data[0].to(device), train_data[1].to(device)
            batch_size = len(reals)
            # convert the labels to one-hot vectors to be concatenated to the noise vector
            # first, expand the one-hot vectors to shape (batch_size, 10, 1, 1) using [:, :, None, None]
            one_hot_labels = F.one_hot(labels, num_classes=num_classes)[:, :, None, None]
            # expand the one-hot vectors to one-hot images of size (1, 28, 28)
            # next, repeat (1, 1, 28, 28)
            one_hot_images = one_hot_labels.repeat((1, *INPUT_IMG_SHAPE))

            # 1. zeros the gradients
            d_optim.zero_grad()
            # 2. generate noise vectors
            noise_1 = torch.randn((batch_size, Z_DIM, 1, 1)).to(device)
            # 3. concate the noise vectors with the one-hot vectors
            # do not concatenate on the batch size dimension
            # NOTE: one_hot_labels is not float, we need to turn it to float
            noise_1_onehot = torch.cat([noise_1.float(), one_hot_labels.float()], dim=1)
            # 4. pass the noise vectors to the generator
            fakes = G(noise_1_onehot).detach()
            # 5. concatenate the fakes with the one-hot images
            fakes_onehot = torch.cat([fakes.float(), one_hot_images.float()], dim=1)
            # 6. predict the fakes
            fakes_preds = D(fakes_onehot)
            # 7. concatenate the reals with the one-hot images
            reals_onehot = torch.cat([reals.float(), one_hot_images.float()], dim=1)
            # 8. predict the reals
            reals_preds = D(reals_onehot)
            # 9. compute the loss
            discriminator_loss = (criterion(fakes_preds, torch.zeros_like(fakes_preds)) + criterion(reals_preds, torch.ones_like(reals_preds))) / 2
            # 10. backward propagation
            discriminator_loss.backward(retain_graph=True)
            # 11. optimiser update
            d_optim.step()

            ### train generator
            # 1. zeros the gradients
            g_optim.zero_grad()
            # 2. generate noise vectors
            noise_2 = torch.randn((batch_size, Z_DIM, 1, 1)).to(device)
            # 3. concatenate the noise vectors with the one-hot vectors
            noise_2_onehot = torch.cat([noise_2.float(), one_hot_labels.float()], dim=1)
            # 4. pass the noise vectors to the generator
            fakes = G(noise_2_onehot)
            # 5. concatenate the fakes with one-hot images
            fakes_onehot = torch.cat([fakes.float(), one_hot_images.float()], dim=1)
            # 6. predict the fakes
            fakes_preds = D(fakes_onehot)
            # 7. compute the loss
            # we want to maximise the prediction of the critic on the fake samples
            generator_loss = criterion(fakes_preds, torch.ones_like(fakes_preds))
            # 8. backward propagation
            generator_loss.backward()
            # 9. optimiser update step
            g_optim.step()

            trainloader.set_description((
                f"epoch: {epoch+1}/{EPOCH}; "
                f"generator loss: {generator_loss.item():.5f}; "
                f"critic loss: {discriminator_loss.item():.5f}"
            ))

            if i % SAMPLE_INTERVAL == 0:
                print(labels[:SAMPLE_SIZE])
                torchvision.utils.save_image(fakes[:SAMPLE_SIZE], GENERATED_DIRPATH + f"cgan_bce_{epoch+1}_{i}.png")
        
        # save the model
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_training_progress(G, D, g_optim, d_optim, epoch, MODEL_DIRPATH + f'cgan-model-bce-epoch{epoch + 1}.pth')
    
    # save the model at the end of training
    save_training_progress(G, D, g_optim, d_optim, epoch, MODEL_DIRPATH + f'cgan-model-bce-epoch{epoch + 1}.pth')