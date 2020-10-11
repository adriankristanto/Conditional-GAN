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
CONTINUE_TRAIN_NAME = 'cgan-model-epoch10.pth'
EPOCH = 400
SAVE_INTERVAL = 20
# for generation
SAMPLE_SIZE = 64

# Hyperparameters
BATCH_SIZE = 128
Z_DIM = 100
GENERATOR_LR = 0.0002
DISCRIMINATOR_LR = 0.0002
BETAS = (0.5, 0.999)

# WGAN-GP global variables
# the recommended ratio used is 5 critic updates to 1 generator update
CRITIC_ITER = 5
# the following is the penalty coefficient
LAMBDA = 10

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
    DATA_PATH = MAIN_DIR + 'data'
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5))
    ])

    trainset = datasets.MNIST(root=DATA_PATH, download=True, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    # total unique classes in MNIST dataset
    num_classes = len(trainset.classes)
    # shape of a single MNIST image
    MNIST_IMG_SHAPE = (1, 28, 28)

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
        kernel_sizes=[None, 7, 5, 4, 4],
        strides=[None, 1, 1, 2, 2],
        paddings=[None, 0, 2, 1, 1],
        batch_norm=True,
        activations=[nn.ReLU(), nn.Tanh()]
    )

    # Discriminator D
    INPUT_CHANNEL = MNIST_IMG_SHAPE[0] + num_classes
    # we will append (batch_size, num_classes, 28, 28), which is similar to the one-hot vector
    # to (batch_size, 1, 28, 28) which is the input batch of MNIST
    # for example, if dimension 3 of the one-hot vector has the value 1, then we create a tensor of size (1, 28, 28) full on 1s for index 3
    # and the other 9 tensor will be filled with 0s
    D = DCGAN.Discriminator(
        channels=[INPUT_CHANNEL, 64, 128, 256, 1],
        kernel_sizes=[None, 4, 4, 5, 7],
        strides=[None, 2, 2, 1, 1],
        paddings=[None, 1, 1, 2, 0],
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
    def GradientPenaltyLoss(D, real_samples, fake_samples, reduction='mean'):
        batch_size = len(real_samples)
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(device)

        inputs = epsilon * real_samples + (1 - epsilon) * fake_samples
        inputs.requires_grad_(True)
        inputs = inputs.to(device)

        outputs = D(inputs)

        gradients = torch.autograd.grad(
            inputs=inputs,
            outputs=outputs,
            grad_outputs=torch.ones_like(outputs).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2

        reduction_func = None
        if reduction == 'mean':
            reduction_func = torch.mean
        elif reduction == 'sum':
            reduction_func = torch.sum
    
        return reduction_func(gradient_penalty)
    
    # 4. define the optimisers
    g_optim = optim.Adam(G.parameters(), lr=GENERATOR_LR, betas=BETAS)
    d_optim = optim.Adam(D.parameters(), lr=DISCRIMINATOR_LR, betas=BETAS)

    # 5. train the model
    MODEL_DIRPATH = MAIN_DIR + 'saved_models'
    GENERATED_DIRPATH = MAIN_DIR + 'generated_images'
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
            one_hot_labels = F.one_hot(labels, num_classes=num_classes)
            # expand the one-hot vectors to one-hot images of size (1, 28, 28)
            # first, expand the one-hot vectors to shape (batch_size, 10, 1, 1) using [:, :, None, None]
            # next, repeat (1, 1, 28, 28)
            one_hot_images = one_hot_labels[:, :, None, None].repeat((1, *MNIST_IMG_SHAPE))

            discriminator_loss_mean = 0

            ### train critic
            for _ in range(CRITIC_ITER):
                # 1. zeros the gradients
                d_optim.zero_grad()
                # 2. generate noise vectors
                noise = torch.randn((batch_size, Z_DIM, 1, 1)).to(device)
                # 3. concate the noise vectors with the one-hot vectors
                # do not concatenate on the batch size dimension
                noise = torch.cat([noise, one_hot_labels], dim=1)
                # 4. pass the noise vectors to the generator
                fakes = G(noise)
                # 5. concatenate the fakes with the one-hot images
                fakes = torch.cat([fakes, one_hot_images], dim=1)
                # 6. predict the fakes
                fakes_preds = D(fakes.detach())
                # 7. concatenate the reals with the one-hot images
                reals = torch.cat([reals, one_hot_images], dim=1)
                # 8. predict the reals
                reals_preds = D(reals)
                # 9. compute the loss
                # the higher the score of fake predictions, the higher the loss -> because we want to predict as low as possible for fakes
                # the higher the score of real predictions, the lower the loss -> we want to predict as high as possible for fakes
                discriminator_loss = fakes_preds.mean() - reals_preds.mean() + LAMBDA * GradientPenaltyLoss(D, reals, fakes)
                # 10. backward propagation
                discriminator_loss.backward()
                # 11. optimiser update
                d_optim.step()

                discriminator_loss_mean += discriminator_loss
            discriminator_loss_mean = discriminator_loss_mean / CRITIC_ITER

            ### train generator
            # 1. zeros the gradients
            g_optim.zero_grad()
            # 2. generate noise vectors
            noise = torch.randn((batch_size, Z_DIM, 1, 1)).to(device)
            # 3. concatenate the noise vectors with the one-hot vectors
            noise = torch.cat([noise, one_hot_labels], dim=1)
            # 4. pass the noise vectors to the generator
            fakes = G(noise)
            # 5. concatenate the fakes with one-hot images
            fakes = torch.cat([fakes, one_hot_images], dim=1)
            # 6. predict the fakes
            fakes_preds = D(fakes)
            # 7. compute the loss
            # we want to maximise the prediction of the critic on the fake samples
            generator_loss = -1 * fakes_preds.mean()
            # 8. backward propagation
            generator_loss.backward()
            # 9. optimiser update step
            g_optim.step()
            break 
        break