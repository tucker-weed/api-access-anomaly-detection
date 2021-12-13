from __future__ import absolute_import
from preprocess import get_data

import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
    def __init__(self):
        """
        Model architecture.
        """
        super(Model, self).__init__()

        self.batch_size = 100
        self.num_classes = 2

        # Hyperparameters
        self.learning_rate = 0.001
        self.num_epochs = 10

        # Trainable parameters
        self.D1 = tf.keras.layers.Dense(self.batch_size, activation="relu")
        self.D2 = tf.keras.layers.Dense(50, activation="relu")
        self.D3 = tf.keras.layers.Dense(25, activation="relu")
        self.D4 = tf.keras.layers.Dense(self.num_classes)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of api data.
        """
        layer1_output = self.D1(inputs)
        layer2_output = self.D2(layer1_output)
        layer3_output = self.D3(layer2_output)

        logits = self.D4(layer3_output)

        return logits

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        """
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing logits to correct labels.
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    """
    Trains the model on all of the inputs and labels for one epoch.
    """
    BATCH_SZ = model.batch_size

    inds = np.arange(0, np.shape(train_inputs)[0])
    np.random.shuffle(inds)
    train_inputs = train_inputs[inds]
    train_labels = train_labels[inds]

    steps = 0
    for i in range(0, np.shape(train_inputs)[0], BATCH_SZ):
        steps += 1
        image = train_inputs[i:i + BATCH_SZ]
        label = train_labels[i:i + BATCH_SZ]
        loss = None
        with tf.GradientTape() as tape:
            predictions = model.call(image)
            loss = model.loss(predictions, label)

        train_acc = model.accuracy(predictions, label)
        print("Loss: {} | Accuracy on training set after {} steps: {}".format(str(loss.numpy())[: 6], steps, train_acc))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels, setType):
    """
    Tests the model on the test inputs and labels.
    """
    BATCH_SZ = model.batch_size
    accs = []

    steps = 0
    for i in range(0, np.shape(test_inputs)[0], BATCH_SZ):
        steps += 1
        image = test_inputs[i:i + BATCH_SZ]
        label = test_labels[i:i + BATCH_SZ]
        predictions = model.call(image, is_testing=True)
        loss = model.loss(predictions, label)
        acc = model.accuracy(predictions, label)
        print("Loss: {} | Accuracy on {} set after {} steps: {}".format(str(loss.numpy())[: 6], setType, steps, acc))
        accs.append(acc)
    return tf.reduce_mean(tf.convert_to_tensor(accs))


def main():
    """
    Executes training and testing steps.
    """
    model = Model()
    (inp_train, lab_train, inp_test, lab_test) = get_data("./data/remaining_behavior_ext.csv")

    for epoch in range(model.num_epochs):
        print("\nEPOCH: {}\n".format(epoch + 1))
        train(model, inp_train, lab_train)
        if epoch < model.num_epochs - 1:
            print("\nCURRENT TEST ACCURACY\n")
            acc = test(model, inp_test, lab_test, setType="test")
            print("Aggregate accuracy: {}".format(acc))

    print("\nTEST SET\n")
    acc = test(model, inp_test, lab_test, setType="test")
    print("\nFINAL TEST ACCURACY: {}\n".format(acc))


if __name__ == '__main__':
    main()
