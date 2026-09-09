# Experiments/Tests

Various things that I'm playing around with. Lowkey just hyperparameter tuning

## E1: Convolution & linear layers

| Test Accuracy | Conv layers | Linear Layers | Notes |
| --- | --- | --- | --- |
| .7377 | 3 -> 32 -> 64 -> 128 | 2048->60->10 | |
| .7302 | 3 -> 32 -> 64 -> 128 | 2048->256->64->10 | Jagged training accuracy and loss curves |
| .7634 | 3 -> 32 -> 64 -> 128 | 2048->512->64->10 | Smoothened out curves + accuracy increase |
| .7169 | 3 -> 16 -> 32 -> 64 | 2048->512->64->10 | |
| .7556 | 3 -> 16 -> 64 -> 256 | 2048->512->64->10 | Very large peaks and dips in curves |
| .784 | 3 -> 64 -> 128 -> 256 | 2048->512->64->10 | One interesting peak. Will rerun |
| .7672 | 3 -> 64 -> 128 -> 256 | 2048->512->64->10 | Same layers, different accuracy |
| .7899 | 3 -> 64 -> 128 -> 256 | 2048->512->64->10 | Doubled epochs, overfitting signs due to rising test loss |

## E2: Adam

### Goal: Use Adam and see what changes**  

Stack:  
Conv: 3 -> 64 -> 128 -> 256  
linear: 2048 -> 512 -> 64 -> 10  
Epochs: 10  

SGD accuracy(lr=0.1, gamma=0.9): ~.7672  
Adam accuracy(lr=0.001): .759  

### Notes

Adam seems to converge much quicker (72% accuracy by epoch 3). Epoch 10 test loss was greater than epoch 1, showing signs of overfitting too.


## E3: Data aug

Goal: Try data aug. See if that increases generalizability 

Stack:  
Conv: 3 -> 64 -> 128 -> 256  
linear: 2048 -> 512 -> 64 -> 10  
Epochs: 10  
Adam optimizer

Base test accuracy: ~.7672

| Test Accuracy | Change(s) |
| --- | --- |
| .7872 | Random horiz flip(p=0.5) |
| .7646 | RandomResizeCrop(32x32, (0.8, 1.0)) |
| .784 | random horiz + random resize(0.8,1.0) |
| .7921 | random horiz + random resize(0.8,1.0) + random rotation(-15,15) |

Ending here, but I know there's more I could try.

## E4: Other architecture changes

Going to try a few things

| Change | Accuracy |
|---|---|
|Batch norm| .8002 |
|much larger arch: 3 -> 32x32x32 -> 64x16x16 -> 128x16x16 -> 256x8x8 -> 512 -> 64 -> 10| .8125 | 

Best result?
.8603