import argparse
import torch
    
def data_augmentation(args):
    # define tranformations, ImageFolder & DataLoader
    # returns DataLoader objects, for training and validation respectively, then a class_to_idx dict
    train_dir = os.path.join(args.data_directory, "train")
    valid_dir = os.path.join(args.data_directory, "valid")

    # validate paths before doing anything else
    if not os.path.exists(args.data_directory):
        print("Data Directory doesn't exist: {}".format(args.data_directory))
        raise FileNotFoundError
    if not os.path.exists(args.save_directory):
        print("Save Directory doesn't exist: {}".format(args.save_directory))
        raise FileNotFoundError

    if not os.path.exists(train_dir):
        print("Train folder doesn't exist: {}".format(train_dir))
        raise FileNotFoundError
    if not os.path.exists(valid_dir):
        print("Valid folder doesn't exist: {}".format(valid_dir))
        raise FileNotFoundError


        train_transforms = transforms.Compose([
                                       
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomCrop(size=(224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
                                       

        valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


        test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


    train_data = ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = ImageFolder(root=valid_dir, transform=valid_transforms)

    train_data_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data_loader = data.DataLoader(valid_data, batch_size=64, shuffle=True)

    return train_data_loader, valid_data_loader, train_data.class_to_idx    
    
def train_classifier():
    


    args = parser.parse_args()
    data = args.data_dir
    path = args.save_dir
    lr = args.learning_rate
    model = args.arch
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu  = args.gpu
    
    model = models.vgg19_bn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
                          nn.Linear(25088, 2048),
                          nn.ReLU(),
                          nn.Linear(2048, 256),
                          nn.ReLU(),
                          nn.Linear(256, 102),
                          nn.LogSoftmax(dim=1)
                          )
    model = model.to('cuda')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    epochs = 5
    printFor = 20
    steps = 0

    for e in range(epochs):
        running_train_loss = 0
        for inputs, labels in train_loader:
            steps += 1
        
            inputs = Variable(inputs.float().cuda())
            labels = Variable(labels.long().cuda())
        
            optimizer.zero_grad()
        
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_train_loss += loss.item()
        
            if steps % printFor == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        validation_loss += batch_loss.item()
                    
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epochs}.. "
                    f"Loss: {running_train_loss/printFor:.3f}.. "
                    f"Validation Loss: {validation_loss/len(valid_loader):.3f}.. "
                    f"Accuracy: {accuracy/len(valid_loader):.3f}")
                running_train_loss = 0
                model.train()
                
                
                checkpoint_save = {
                'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'epochs': epochs,
                'optim_stat_dict': optimizer.state_dict(),
                'class_to_idx': train_data.class_to_idx,
                'input_size': 25088,
                'output_size': 102,
                'structure': 'vgg16_bn',
                'learning_rate': 0.001,
                 }

                torch.save(checkpoint_save, 'checkpoint_file_vgg19_bn.pth')
    
    
    
    
def main():
    
    
    
    parser = argparse.ArgumentParser(
    
        description = 'train Image application'
    )
    
    
    parser.add_argument('data_dir', action="store", default="./flowers/")
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', action="store", default="vgg19_bn")
    parser.add_argument('--learning_rate', action="store", type=float,default=0.01)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
    parser.add_argument('--epochs', action="store", default=3, type=int)
    parser.add_argument('--gpu', action="store", default="gpu")
    
    
    args = parser.parse_args()
    if torch.cuda.is_available() and args.gpu == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # load and transform data
    train_data_loader, valid_data_loader, class_to_idx = data_augmentation(args)

    # train and save model
    train_classifier(args, train_data_loader, valid_data_loader, class_to_idx)
    
    
    
    
    
    



    
    
    
        
    
    
    
    
    
    
    