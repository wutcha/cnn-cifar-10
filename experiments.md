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



 