# Image Classification for City Dog Show

Our city is organizing a city-wide dog show, and I volunteered to assist the organizing committee in registering contestants. My task involves using machine learning models to accurately classify a contestant’s photo as either a dog or not, and if it is a dog, identifying its breed.

To accomplish this, I trained three distinct convolutional neural networks (CNNs) on the ImageNet dataset: AlexNet, VGG, and ResNet. I assessed their performance and the time required to make inferences for a given set of images.

`check_images.py` is the entry point for the program.

`get_input_args.py` defines command-line arguments for different inputs such as pet folder, architecture, and dog file.

`get_pet_labels.py` extracts pet name (true label) from file name, formats it, and puts it in a dictionary. Its key is file name, and its value is a list at the first index with the true pet name.

`classify_images.py` makes predictions on each image, then stores the result in a dictionary. At the 1st index, the predicted label; at the 2nd index, 1 if the true label and predicted label match, otherwise 0.

`adjust_results4_isadog.py` reads given list of dog names from dog file then if true label is in the dog names put 1 at dictionary 3rd index otherwise 0. if predicted label is the dog list put 1 in the dictionary's 4 th index otherwise 0.

`calculates_results_stats.py` calculates simple statistics to compare the performance of each of the CNN models.

`print_results.py` formats the results to output on the console.

`classifier.py`: This module contains all the CNN model-related code.

`test_classifier.py`: This is used to test the classifier function.

`print_functions_for_lab_checks.py`: This module contains all the tests for the above functions.


`alexnet_pet-images.txt`
`alexnet_uploaded-images.txt`

`check_images.txt`
`dognames.txt`
`imagenet1000_clsid_to_human.txt`

`resnet_pet-images.txt`
`resnet_uploaded-images.txt`

`vgg_pet_images.txt`
`vgg_uploaded-images.txt`

`run_models_batch.sh`
`run_models_batch_uploaded.sh`
