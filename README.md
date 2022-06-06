# ML_Final_Project
## About This Project

- realtime segmentation network
## Usage
- default Unet
    ```python
    python3 train.py --batch_size 8 --cuda_device 1 --model_name default_unet --depth 5 --wandb
    ```
- deeplabv3+
    ```python
    python3 train.py --batch_size 8 --cuda_device 1 --model_name deeplabv3+ --backbone MobileNetV2 --wandb
    ```
## Reference
- [DeepLabV3+ exmaple](https://keras.io/examples/vision/deeplabv3_plus/)
- [DeepLabV3 benchmarks](https://www.tensorflow.org/lite/examples/segmentation/overview)
- [Keras Backbone Networks](https://keras.io/api/applications/)
