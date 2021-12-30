import classifier_functions as cf
import os, random, argparse, torch

def main():
    category = random.sample(range(1, 99), 1)[0]
    image_path = 'ImageClassifier/flowers/test/' + str(category) +'/' + random.choice(os.listdir('ImageClassifier/flowers/test/' + str(category) +'/'))
    
    parser = argparse.ArgumentParser(description='PREDICT THE FLOWER CLASSIFICATION')
    parser.add_argument('--image_path', type=str, default=image_path, help='Path of image for classification')
    parser.add_argument('--checkpoint' , type=str, default='ImageClassifier/my_checkpoint.pth', help='Path of saved model')
    parser.add_argument('--gpu', type=bool, default=False, help='IS GPU ENABLED? TRUE/FALSE')
    parser.add_argument('--topk', type=int, default=3, help='Display top k probabilities')
    args = parser.parse_args()
        
    if args.gpu == True:  
        if torch.cuda.is_available():
            print ("GPU ENABLED")
            saved_model = args.checkpoint
            image_path = args.image_path
            top_k = args.topk
          
            cf.predict_image(image_path, saved_model, top_k)
            
        else:
            print("GPU is not available. Enable GPU and try again")
    else:
        print("Enable GPU and add this argument '--gpu True'")
        
if __name__ == "__main__":
    main()