
## Usage
```python
model1 = ... # Some model, casted to GPU
model2 = ... # Another model, casted to GPU
dataloader = ... # Your dataloader

calculator = CKACalculator(model1, model2, dataloader)
cka_matrix = calculator.calculate_cka_matrix()
```

Rather than caching intermediate feature representations, this code computes CKA on-the-fly (simultaneously with the model forward pass) by using the mini-batch CKA, as described in the [paper by Nguyen et. al.](https://openreview.net/pdf?id=KJNcAkY8tY4)
By leveraging GPU superiority, **this implementation runs much faster than any Numpy implementation.**

## Setup
I haven't added a `requirements.txt` since the exact version of each package is not that important :man_shrugging:

#### Required packages to use the class/functions:
* python3.7+
* torch (any relatively recent version should be O.K.)
* torchvision 
* tqdm
* torchmetrics

#### To run the `example.ipynb`:
* jupyter
* matplotlib
* numpy

## Example notebook
Try out the example notebook in `example.ipynb`.

## Other
* If you found this repo helpful, please give it a :star:
* If you find any bugs/improvements, feel free to create a new issue.
* This code is mostly tested on ResNets

### TODO (when I feel like it)
* Ditch hooks; change to `torch.fx` implementation
