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
    This layer computes the L2 distance between two embeddings.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input1, input2):
        # Calculate squared L2 distance
        distance = tf.reduce_sum(tf.square(input1 - input2), axis=-1, keepdims=True)
        return distance

    def get_config(self):
        config = super(DistanceLayer, self).get_config()
        return config


@register_keras_serializable(package="Custom")
class SiameseModel(Model):
    """Siamese Network with Contrastive Loss"""

    def __init__(self, siamese_network, margin=0.5, **kwargs):
        super(SiameseModel, self).__init__(**kwargs)
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        (input1, input2), labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            distances = self.siamese_network([input1, input2])
            loss = self._compute_loss(labels, distances)

        # Compute gradients
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        # Update loss metric
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        (input1, input2), labels = data
        distances = self.siamese_network([input1, input2])
        loss = self._compute_loss(labels, distances)

        # Update loss metric
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, labels, distances):
        # Contrastive Loss calculation
        positive_loss = (1 - labels) * tf.square(distances)  # Similar pair (label 1)
        negative_loss = labels * tf.square(tf.maximum(self.margin - distances, 0))  # Dissimilar pair (label 0)
        loss = positive_loss + negative_loss
        loss = tf.reduce_mean(loss)
        return loss

    @property
    def metrics(self):
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
    # Create input placeholders for pairs of images
    input1 = layers.Input(name="input1", shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    input2 = layers.Input(name="input2", shape=(IMG_WIDTH, IMG_HEIGHT, 3))

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
    distance = DistanceLayer()(
        embedding(resnet.preprocess_input(input1)),
        embedding(resnet.preprocess_input(input2)),
    )

    # Step 6: Siamese Network
    siamese_network = Model(inputs=[input1, input2], outputs=distance)

    # Step 7: Siamese Model
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.0001))

    return siamese_model