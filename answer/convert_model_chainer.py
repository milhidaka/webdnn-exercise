"""
Converts the trained CNN model for WebDNN
"""

import numpy as np
import chainer

from webdnn.backend import generate_descriptor
from webdnn.frontend.chainer import ChainerConverter

from train_mnist_chainer import CNN  # import model definition


def main():
    model = chainer.links.Classifier(CNN())
    chainer.serializers.load_npz('chainer_output/chainer_model.npz', model)
    predictor = model.predictor
    input_variable = chainer.Variable(np.zeros((1, 1, 28, 28), dtype=np.float32))
    prediction_raw_variable = predictor(input_variable)
    prediction_with_softmax_variable = chainer.functions.softmax(prediction_raw_variable)

    graph = ChainerConverter().convert([input_variable], [
        prediction_with_softmax_variable])  # convert graph to intermediate representation
    backend = 'webgl'
    exec_info = generate_descriptor(backend, graph)
    exec_info.save('webdnn_graph_descriptor')


if __name__ == '__main__':
    main()
