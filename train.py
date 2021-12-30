import classifier_functions as cf
import argparse, torch

def main():
    parser = argparse.ArgumentParser(description='BUILD AND TRAIN THE NETWORK FOR FLOWER CLASSIFICATION')
    parser.add_argument('--data_dir', type=str, default='ImageClassifier/flowers', help='Path to flower folder')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--gpu', type=bool, default=False, help='IS GPU ENABLED? TRUE/FALSE')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units for fc layer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    args = parser.parse_args()
    
    if args.gpu:
        if torch.cuda.is_available():
            print ("GPU ENABLED")
            hidden_units = args.hidden_units
            data_pack = args.data_dir
            epochs = args.epochs
            lr = args.lr
            
            image_datasets, image_valid_sets, image_test_sets, dataloaders, valid_loaders, test_loaders = cf.load_data(data_pack)
            model = cf.build_model(dataloaders, valid_loaders, image_datasets, image_valid_sets, hidden_units, lr, epochs)
            cf.check_validation(model, test_loaders)
            
        else:
            print("GPU is not available. Enable GPU and try again")
    else:
        print("Enable GPU and add this argument '--gpu True'")
        
if __name__ == "__main__":
    main()