## 人脸检测模型

## 环境搭建

```
pip install onnxruntime
pip install onnxruntime-gpu
pip install opencv-python
```

## 快速开始

```
python start.py
```

### Input
Input tensor is `1 x 3 x height x width` with mean values `127, 127, 127` and scale factor `1.0 / 128`. 

Input image `RGB`  resized to `320 x 240`  `640 x 480` 

### Preprocessing
 `image_path`
```python
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (320, 240))
image_mean = np.array([127, 127, 127])
image = (image - image_mean) / 128
image = np.transpose(image, [2, 0, 1])
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32)
```

### Output
The model outputs two arrays `(1 x 4420 x 2)` and `(1 x 4420 x 4)` of scores and boxes.

### Postprocessing
In postprocessing, threshold filtration and [NMS](box_utils.py) are applied to the scores and boxes arrays.

## License
MIT
