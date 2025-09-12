# Train Your First PyTorch Model: Card Classifier

This project walks through a beginner-friendly image classification pipeline in the notebook `train-your-first-pytorch-model-card-classifier.ipynb`.

The goal is simple: given a picture of a playing card, train a model that predicts which card it is, such as `ace of spades` or `five of diamonds`.

The notebook teaches the three building blocks you will see in almost every PyTorch project:

1. A `Dataset` to load images and labels
2. A `Model` to turn images into predictions
3. A `Training Loop` to improve the model over time

## What This Model Does

The model learns from folders of card images. Each folder name is treated as the label for that class, and the model tries to predict the correct class from the image pixels.

In this notebook, the final setup uses:

- 53 card classes
- images resized to `128 x 128`
- batch size `32`
- a pretrained `EfficientNet-B0` backbone from `timm`
- `CrossEntropyLoss` as the loss function
- `Adam` optimizer with learning rate `0.001`
- `5` training epochs

## The Big Picture

Here is the full flow of the notebook:

1. Import PyTorch and helper libraries
2. Create a custom dataset class
3. Load card images from folders
4. Resize images and convert them to tensors
5. Group images into batches with a `DataLoader`
6. Build a neural network for 53 card classes
7. Send data through the model to get predictions
8. Compute a loss to measure mistakes
9. Update the model with backpropagation
10. Check performance on validation data
11. Plot training and validation loss
12. Run prediction on new test images

## Step-by-Step Explanation

### 1. Import the libraries

The notebook starts by importing the tools it needs:

- `torch`, `torch.nn`, `torch.optim`: core PyTorch components
- `Dataset`, `DataLoader`: for loading data
- `torchvision.transforms`: for image preprocessing
- `ImageFolder`: for reading images from class-named folders
- `timm`: for loading a pretrained vision model
- `matplotlib`, `numpy`, `pandas`, `tqdm`: for plotting, array handling, and progress bars

Why this matters:

- PyTorch handles the model and training
- `torchvision` handles image preparation
- `timm` gives you a strong pretrained model so you do not start from scratch

### 2. Create the dataset class

The notebook defines a custom class called `PlayingCardDataset`, which wraps `ImageFolder`.

What it does:

- reads images from a directory
- uses folder names as labels
- returns one image and one label at a time
- exposes the list of class names through `classes`

Why this matters:

- PyTorch expects datasets to answer three questions:
  - how many samples are there?
  - what is sample `i`?
  - what are the class names?

Even though this class is small, it introduces the standard PyTorch dataset pattern.

### 3. Load the card dataset

The notebook first creates a dataset from the training folder:

- `/kaggle/input/cards-image-datasetclassification/train`

Later, it sets up three folders:

- `train/`
- `valid/`
- `test/`

These serve different purposes:

- `train`: used to teach the model
- `valid`: used to check progress during training
- `test`: meant for final evaluation or demo predictions

Why this matters:

- if you only test on training data, the model can look better than it really is
- validation data helps you measure how well the model generalizes to unseen images

### 4. Apply image transforms

Before images can go into a neural network, they must be turned into a consistent format.

The notebook uses:

- `Resize((128, 128))`
- `ToTensor()`

What these do:

- `Resize((128, 128))`: makes every image the same size
- `ToTensor()`: converts the image into numbers PyTorch can use

After this step, a single image has shape:

- `[3, 128, 128]`

This means:

- `3` color channels: red, green, blue
- `128 x 128` pixels

Why this matters:

- neural networks need fixed-size numeric input
- transforms make the data clean and consistent

### 5. Use a DataLoader for batching

The notebook wraps the dataset in a `DataLoader`:

- `batch_size=32`
- `shuffle=True` for training

Instead of feeding one image at a time, the model sees 32 images at once.

The notebook shows a batch shape of:

- images: `[32, 3, 128, 128]`
- labels: `[32]`

Why batching helps:

- training is faster
- GPUs work better with batches
- gradients are usually more stable than single-image training

Why shuffling helps:

- it prevents the model from seeing the data in the same order every epoch
- it reduces the chance of learning order-specific patterns

### 6. Build the model

The notebook defines `SimpleCardClassifer`, which uses a pretrained `EfficientNet-B0`.

The model has two main parts:

1. `features`
   - this is the pretrained image feature extractor
2. `classifier`
   - this is a final linear layer that maps extracted features to 53 card classes

In plain language:

- the pretrained backbone learns useful visual patterns such as edges, textures, and shapes
- the last layer learns how to use those patterns to tell one card apart from another

Why this is a strong beginner approach:

- training a vision model from scratch needs much more data and compute
- transfer learning lets you reuse knowledge from a model already trained on a huge image dataset

### 7. Understand the model output

When a batch goes through the model, the output shape is:

- `[32, 53]`

That means:

- `32` predictions, one for each image in the batch
- `53` numbers for each image, one score per class

These raw scores are often called `logits`.

Important beginner note:

- the model does not directly output probabilities at this stage
- it outputs scores, and the loss function knows how to work with them

### 8. Choose the loss function and optimizer

The notebook uses:

- `nn.CrossEntropyLoss()`
- `optim.Adam(model.parameters(), lr=0.001)`

### Loss function: `CrossEntropyLoss`

This tells the model how wrong it is.

If the model gives a high score to the correct class, the loss becomes smaller.
If it gives a high score to the wrong class, the loss becomes larger.

For multi-class image classification, this is the standard starting choice.

### Optimizer: `Adam`

The optimizer decides how the model weights should change after each batch.

Why `Adam` is used here:

- it usually works well out of the box
- it is a very common beginner-friendly default

### 9. Set up train, validation, and test datasets

The notebook creates three datasets using the same `PlayingCardDataset` class and the same transform:

- `train_dataset`
- `val_dataset`
- `test_dataset`

Then it creates loaders:

- `train_loader`
- `val_loader`
- `test_loader`

The intended purpose is:

- `train_loader`: training
- `val_loader`: validation during training
- `test_loader`: later evaluation

Note:

- in the notebook, `test_loader` is created from `val_dataset` instead of `test_dataset`
- if you want a true test loader, that line should use `test_dataset`

### 10. Train the model

This is the core of the notebook.

The training loop runs for `5` epochs.

### What happens in one training epoch

For each batch in `train_loader`, the notebook does this:

1. Move images and labels to the device (`GPU` if available, otherwise `CPU`)
2. Reset old gradients with `optimizer.zero_grad()`
3. Run the images through the model
4. Compute the loss
5. Call `loss.backward()` to calculate gradients
6. Call `optimizer.step()` to update the model weights
7. Add the batch loss to a running total

This is the standard PyTorch training pattern.

### What backpropagation means

Backpropagation is how the model learns.

It measures how each model weight contributed to the error, then nudges the weights in a direction that should reduce future error.

You do not have to calculate this manually. PyTorch does it for you when you call:

- `loss.backward()`

### 11. Validate after each epoch

After the training pass, the notebook switches to validation mode:

- `model.eval()`
- `with torch.no_grad():`

Why this matters:

- `model.eval()` tells the model to behave correctly for evaluation
- `torch.no_grad()` saves memory and skips gradient tracking because we are not updating weights

The validation loop:

1. runs the model on validation batches
2. computes validation loss
3. stores the value in `val_losses`

This lets you compare:

- training loss: how well the model fits training data
- validation loss: how well it performs on unseen validation data

The final notebook output shows the model reaching a low validation loss by the end of training, which is a good sign that it is learning useful patterns.

### 12. Plot the loss curves

The notebook then plots:

- `train_losses`
- `val_losses`

Why this is useful:

- if both losses go down, training is usually improving
- if training loss drops but validation loss rises, the model may be overfitting

This is one of the easiest ways to visually check whether training is healthy.

### 13. Make predictions on new images

The last part of the notebook shows how to use the trained model on a single image.

It defines three helper steps:

### `preprocess_image`

- opens an image with `PIL`
- converts it to RGB
- applies the same transform used during training
- adds a batch dimension with `unsqueeze(0)`

That last part is important:

- the model expects a batch, even if there is only one image

### `predict`

- switches the model to evaluation mode
- disables gradients
- runs the image through the model
- applies `softmax` to turn raw scores into probabilities

Why `softmax` is used here:

- during inference, probabilities are easier for humans to understand than logits

### `visualize_predictions`

- shows the original image
- shows a horizontal bar chart of class probabilities

This makes the result easier to inspect than only printing a class index.

## Key Beginner Terms

- `Dataset`: a PyTorch object that knows how to load samples
- `DataLoader`: creates batches from a dataset
- `Transform`: preprocessing applied to each image
- `Batch`: a small group of samples processed together
- `Epoch`: one full pass through the training dataset
- `Model`: the neural network
- `Loss`: a number showing how wrong the model is
- `Optimizer`: the algorithm that updates the weights
- `Backpropagation`: the process used to compute how weights should change
- `Validation`: checking the model on unseen data during training
- `Inference`: using the trained model to make predictions

## Why This Notebook Is Good for Beginners

This notebook is a strong first project because it introduces the full machine learning workflow without too much abstraction.

You learn:

- how image folders become labeled training data
- how tensors move through a model
- how training and validation differ
- how loss guides learning
- how to turn a trained model into actual predictions

It also uses transfer learning, which is one of the most practical ways to build a useful computer vision model quickly.

## Possible Improvements After This Notebook

Once you understand this version, the next good steps are:

1. calculate accuracy on validation and test data
2. save the trained model with `torch.save`
3. add data augmentation such as random flips or rotations
4. use early stopping or learning-rate scheduling
5. show only the top 3 or top 5 predictions instead of all 53 classes
6. fix `test_loader` so it uses `test_dataset`

## How to Read the Notebook

If you are new to PyTorch, read the notebook in this order:

1. dataset class
2. transforms
3. dataloaders
4. model definition
5. loss and optimizer
6. training loop
7. validation loop
8. prediction functions

If you understand those pieces, you understand the core of a real PyTorch training pipeline.
