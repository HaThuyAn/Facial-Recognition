import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.saving import register_keras_serializable

IMG_WIDTH = 200
IMG_HEIGHT = 200


@register_keras_serializable(package="Custom")
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Calculate two squared-L2 distances:
    # Distance between anchor and positive
    # Distance between anchor and negative
    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

    def get_config(self):
        config = super(DistanceLayer, self).get_config()
        return config


@register_keras_serializable(package="Custom")
class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5, **kwargs):
        super(SiameseModel, self).__init__(**kwargs)
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # we do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Update and return the training loss metric
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Update and return the loss metric
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically
        return [self.loss_tracker]

    def get_config(self):
        config = super(SiameseModel, self).get_config()
        config.update({
            "margin": self.margin,
            "siamese_network": tf.keras.models.clone_model(self.siamese_network).get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        siamese_network_config = config.pop("siamese_network")
        siamese_network = tf.keras.models.Model.from_config(siamese_network_config)
        margin = config.pop("margin", 0.5)
        return cls(siamese_network=siamese_network, margin=margin, **config)


def build_siamese():
    anchor_input = layers.Input(name="anchor", shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    positive_input = layers.Input(name="positive", shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    negative_input = layers.Input(name="negative", shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Step 2: Base CNN (ResNet50)
    base_cnn = resnet.ResNet50(
        weights="imagenet",
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
        include_top=False
    )

    # Start by freezing all layers
    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    # Step 3: Embedding Network
    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    bn1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(bn1)
    bn2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(bn2)

    # Step 4: Embedding Model
    embedding = Model(base_cnn.input, output, name="Embedding")

    # Step 5: Distance Layer
    distances = DistanceLayer()(
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )

    # Step 6: Siamese Network
    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    # Step 7: Siamese Model
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.0001))

    return siamese_model