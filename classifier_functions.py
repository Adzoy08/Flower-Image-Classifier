# Imports here
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

from torchvision import datasets, transforms, models
from workspace_utils import active_session
import os, random, torch, time, copy, json
from torch.optim import lr_scheduler
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, optim
from PIL import Image 
import numpy as np


def load_data(args):
    '''FUNCTION TO LOAD DATA, TRANSFORM IT AND CREATE DATALOADERS'''
    data_dir = args
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transform the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    image_valid_sets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    image_test_sets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Create dataloaers with the image datasets and the trainforms
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(image_valid_sets, batch_size=32)
    test_loaders = torch.utils.data.DataLoader(image_test_sets, batch_size=32)
    
    print(f'{data_dir} loaded successfully.')
    print()
    return image_datasets, image_valid_sets, image_test_sets, dataloaders, valid_loaders, test_loaders


def train_model(dataloaders, valid_loaders, image_datasets, image_valid_sets, model, criterion, optimizer, scheduler, epochs=20):
    '''Credit: PYTORCH TRANSFER LEARNING FOR COMPUTER VISION TUTORIAL'''
    start_time = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for e in range(epochs):
        print('Epoch {}/{}'.format(e + 1, epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [dataloaders, valid_loaders]:
            running_loss = 0.0
            running_corrects = 0

            if phase == dataloaders:
                scheduler.step()
                phase_name = 'Training phase'
                size = len(image_datasets)
                model.train()  # Set model to training mode            
            else:
                model.eval()   # Set model to evaluate mode
                size = len(image_valid_sets)
                phase_name = 'Valiation phase'

            # Iterate over data and get the inputs.
            for inputs, labels in phase:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == dataloaders:
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            
            epoch_loss = running_loss / size
            epoch_acc = running_corrects.double() / size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase_name, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == valid_loaders and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
       
        print('-' * 60)
        
    end_time = time.time()
    time_spent = end_time - start_time
    hours = int(time_spent / 3600)
    minutes = int((time_spent % 3600) / 60)
    seconds = int((time_spent % 3600) % 60)
    print('Training completed in ' + str(hours) + ':' + str(minutes) + ':' + str(seconds))
    
    print('Best val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def check_validation(model, test_loaders):
    '''FUNCTION TO CHECK VALIDATION ON THE TEST SET'''    
    accuracy = 0
    test_loss = 0
    criterion = nn.NLLLoss()
    
    for inputs, labels in test_loaders:
        # Set volatile to True so we don't save the history
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        with torch.no_grad():
            model.eval()
            log_ps = model(inputs)
            test_loss += criterion(log_ps, labels)

    ## Calculating the accuracy 
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    print("Test Accuracy: {:.2f}%".format((accuracy/len(test_loaders) * 100)))

def build_model(dataloaders, valid_loaders, image_datasets, image_valid_sets, hidden_units, lr, epochs):
    '''FUNCTION TO BUILD AND TRAIN THE NETWORK WITH A PRE-TRAINED NETWORK  '''
    
    # Pre-trained network
    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # A new, untrained feed-forward network classifier with ReLU activations and dropout
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 1024)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.2)),
                              ('fc2', nn.Linear(1024, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.2)),
                              ('fc3', nn.Linear(hidden_units, 256)),
                              ('relu', nn.ReLU()),
                              ('fc4', nn.Linear(256, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    model.classifier = classifier
    
    # Train the classifier layers with the pre-trained network.
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    model = model.cuda()

    # For every 2 epochs,decay lr by a factor of 0.1
    model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    with active_session():
        model = train_model(dataloaders, valid_loaders, image_datasets, image_valid_sets, model, criterion, optimizer, model_lr_scheduler, epochs)
    
    # Save the checkpoint 
    model.class_to_idx = image_datasets.class_to_idx
    model.epochs = epochs
    checkpoint = {'input_size': 25088,
                  'batch_size': dataloaders.batch_size,
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': model.epochs}
    
    torch.save(checkpoint, 'ImageClassifier/my_checkpoint.pth')
    return model 


def load_checkpoint(directory):
    ''' FUNCTION TO LOAD A CHECKPOINT AND REBUILD MODEL '''
    checkpoint = torch.load(directory)
    model = models.vgg16()
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024)),
                          ('relu', nn.ReLU()),
                          ('drpot', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(1024, 512)),
                          ('relu', nn.ReLU()),
                          ('drpot', nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(512, 256)),
                          ('relu', nn.ReLU()),
                          ('fc4', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']


def process_image(image_path):
    ''' Process a PIL image for use in a PyTorch model
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Resize image
    image = Image.open(image_path)
    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        image.thumbnail(size=[256**600, 256])
    else:
        image.thumbnail(size=[256, 256**600])
    
    # Crop out the centre
    image = image.crop((16, 16, 240, 240))
    
    # Color channels
    np_image = np.array(image)
    np_image = np_image/255.
    
    # Normalize the color channels
    mean = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225] 
    np_image = (np_image - mean) / std_dev
    
    # Transpose
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, index_to_class, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Predict the class from an image file 
    model.eval();
    
    #'Process' the image then convert to tensor 
    image = torch.FloatTensor([process_image(image_path)])
    
    output = model.forward(Variable(image))
    ps = torch.exp(output).data.numpy()[0]
    
    top_index = np.argsort(ps)[-topk:][::-1] 
    probabilities = ps[top_index]
    classes = [index_to_class[x] for x in top_index]

    return probabilities, classes


def predict_image(image_path, model, topk):
    ''' Function for predicting an image.
    '''    
    loaded_model, class_to_idx = load_checkpoint(model)
    index_to_class = { v : k for k,v in class_to_idx.items()}
    print('model loaded successfully')
    image_name = image_path.split('/')[-2]
    print(cat_to_name[image_name].title())
    
    probabilities, classes = predict(image_path, loaded_model, index_to_class, topk)
    
    print()
    print('Prediction of flower name: ')
    print('-' * 50)
    for i in range(len(probabilities)):
        print('{}. Flower:({}) {}, Probability: {:.2f}%'.format(i+1, classes[i], cat_to_name[classes[i]].title(), probabilities[i]*100))
                  
    print('-' * 50)
        
with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)