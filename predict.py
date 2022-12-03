
import argparse
import torch
import matplotlib.pyplot as plt
def load_trained_model(args):
    checkpoint = torch.load('checkpoint_file_vgg19_bn.pth')
    structure = checkpoint['structure']
    model,_,_ = nn_model()
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    pil_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = pil_transforms(img)
    
    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to('cuda')
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        logps = model.forward(img.cuda())
        
    probability = torch.exp(logps).data
    
    return probability.topk(topk)

def predict_image(args):
    
    plt.rcdefaults()
    fig, ax = plt.subplots()

    index = 1
    path = args.image_filepath
    ps = predict(path, model)
    image = process_image(path)

    ax1 = imshow(image, ax = plt)
    ax1.axis('off')
    ax1.title(cat_to_name[str(index)])


    a = np.array(ps[0][0])
    b = [cat_to_name[str(index+1)] for index in np.array(ps[1][0])]
    print(ps[1][0])
    fig,ax2 = plt.subplots(figsize=(5,5))


    y_pos = np.arange(5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(b)
    ax2.set_xlabel('Probability')
    ax2.invert_yaxis()
    ax2.barh(y_pos, a)

    plt.show()
    
if __name__ == '__main__':
    
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(dest='image_filepath', help="Path to the image that you want to classify")
    parser.add_argument(dest='model_filepath', help="path to the checkpoint")
    parser.add_argument('--json_filepath', dest='json_filepath', help="file path to json", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="This returns top 5 classes", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="To use gpu for training use this", action='store_true')

    # Parse and print the results
    args = parser.parse_args()

    predict_image(args)    