# banana-ripeness-classification
Banana ripeness classification with Neural Networks.

```text
data        - Manually labeled data (255 images of green, ripe and overripe bananas).
docs        - Web UI for inference (includes pretrained model).
notebooks   - Keras code for training the model.
sources     - URLs for the images in the data (many broken links).
```

Use the notebook to train the model and then convert using `tensorflowjs_converter`:

```text
$ tensorflowjs_converter --input_format=keras ./keras_model.h5 ./tfjs_model
```

## References
https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e

https://hackernoon.com/classifying-images-using-tensorflow-js-keras-58431c4df04

https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet
