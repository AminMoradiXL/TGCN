import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from custom_functions import func_create_tf_dataset, func_preprocess, func_compute_adjacency_matrix

test_number = 33
in_feat = 1
batch_size = 64
epochs = 20
input_sequence_length = 12
forecast_horizon = 10
multi_horizon = True
out_feat = 10
lstm_units = 128
patience = 10
learning_rate=0.0002
adj_select = 0 # adj matrix selection : 0 for orig, 1 for rand, 2 for identity

url = "https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/data_loader/PeMS-M.zip"
data_dir = keras.utils.get_file(origin=url, extract=True, archive_format="zip")
data_dir = data_dir.rstrip(".zip")

route_distances = pd.read_csv(
    os.path.join(data_dir, "W_228.csv"), header=None).to_numpy()

speeds_array = pd.read_csv(
    os.path.join(data_dir, "V_228.csv"), header=None).to_numpy()

print(f"route_distances shape={route_distances.shape}")
print(f"speeds_array shape={speeds_array.shape}")

# sample_routes = [
#     0,
#     1,
#     4,
#     7,
#     8,
#     11,
#     15,
#     108,
#     109,
#     114,
#     115,
#     118,
#     120,
#     123,
#     124,
#     126,
#     127,
#     129,
#     130,
#     132,
#     133,
#     136,
#     139,
#     144,
#     147,
#     216,
# ]

# route_distances = route_distances[np.ix_(sample_routes, sample_routes)]
# speeds_array = speeds_array[:, sample_routes]

# print(f"route_distances shape={route_distances.shape}")
# print(f"speeds_array shape={speeds_array.shape}")

plt.figure(figsize=(18, 6))
plt.plot(speeds_array[:, [0, -1]])
plt.legend(["route_0", "route_25"])

fig = plt.figure(figsize=(8, 8))
plt.matshow(np.corrcoef(speeds_array.T), 0)
plt.colorbar()
plt.clim(-1,1)
plt.xlabel("road number")
plt.ylabel("road number")
fig.savefig('traffic_correlation',dpi=600)


train_size, val_size = 0.5, 0.2

train_array, val_array, test_array = func_preprocess(speeds_array, train_size, val_size)


train_dataset, val_dataset = (func_create_tf_dataset(
    data_array, 
    input_sequence_length, 
    forecast_horizon, 
    batch_size)
    for data_array in [train_array, val_array]
)

test_dataset = func_create_tf_dataset(
    test_array,
    input_sequence_length,
    forecast_horizon,
    batch_size=test_array.shape[0],
    shuffle=False,
    multi_horizon=multi_horizon,
)

print(f"train set size: {train_array.shape}")
print(f"validation set size: {val_array.shape}")
print(f"test set size: {test_array.shape}")

class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes


sigma2 = 0.1
epsilon = 0.5

if adj_select == 0: # original
    adjacency_matrix = func_compute_adjacency_matrix(route_distances, sigma2, epsilon)
elif adj_select == 1: # random
    adjacency_matrix = np.random.rand(len(route_distances),len(route_distances))
    adjacency_matrix  = adjacency_matrix * np.transpose(adjacency_matrix)
    for i in range(len(adjacency_matrix)):
        adjacency_matrix[i][i]=1
elif adj_select == 2: # identity        
    adjacency_matrix = np.identity(len(route_distances))

node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")


class GraphConv(layers.Layer):
    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tf.Tensor):
        """Computes each node's representation.

        The nodes' representations are obtained by multiplying the features tensor with
        `self.weight`. Note that
        `self.weight` has shape `(in_feat, out_feat)`.

        Args:
            features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)

    def call(self, features: tf.Tensor):
        """Forward pass.

        Args:
            features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)


class LSTMGC(layers.Layer):
    """Layer comprising a convolution layer followed by LSTM and dense layers."""

    def __init__(
        self,
        in_feat,
        out_feat,
        lstm_units: int,
        input_seq_len: int,
        output_seq_len: int,
        graph_info: GraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # graph conv layer
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

        self.lstm = layers.LSTM(lstm_units, activation="relu")
        self.dense = layers.Dense(output_seq_len)

        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs):
        """Forward pass.

        Args:
            inputs: tf.Tensor of shape `(batch_size, input_seq_len, num_nodes, in_feat)`

        Returns:
            A tensor of shape `(batch_size, output_seq_len, num_nodes)`.
        """

        # convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(
            inputs
        )  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_feat)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )

        # LSTM takes only 3D tensors as input
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out = self.lstm(
            gcn_out
        )  # lstm_out has shape: (batch_size * num_nodes, lstm_units)

        dense_output = self.dense(
            lstm_out
        )  # dense_output has shape: (batch_size * num_nodes, output_seq_len)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(
            output, [1, 2, 0]
        )  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)


graph_conv_params = {
    "aggregation_type": "mean",
    "combination_type": "concat",
    "activation": None,
}

st_gcn = LSTMGC(
    in_feat,
    out_feat,
    lstm_units,
    input_sequence_length,
    forecast_horizon,
    graph,
    graph_conv_params,
)
inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
outputs = st_gcn(inputs)

model = keras.models.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
    loss=keras.losses.MeanSquaredError(),
)
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[keras.callbacks.EarlyStopping(patience=patience)],
)

### 

x_test, y_test = next(test_dataset.as_numpy_iterator())
y_pred = model.predict(x_test)

if multi_horizon == False:
    naive_mse, model_mse = (
        np.square(x_test[:, -1, :, 0] - y_test[:, 0, :]).mean(),
        np.square(y_pred[:, 0, :] - y_test[:, 0, :]).mean(),
    )
elif multi_horizon == True:
    naive_mse, model_mse = (
        np.square(x_test[:, -forecast_horizon:, :, 0] - y_test).mean(),
        np.square((y_pred - y_test).mean(axis=1)).mean(),
    )
    
print(f"naive MSE: {naive_mse}, model MSE: {model_mse}")

file_name = f'Results/test_number_{test_number}/MSE_log.txt'
os.makedirs(os.path.dirname(file_name), exist_ok=True)
with open(file_name,"w+") as f:
    f.write(f'batch_size = {batch_size }\n')
    f.write(f'epochs = {epochs}\n')
    f.write(f'input_sequence_length = {input_sequence_length}\n')
    f.write(f'forecast_horizon = {forecast_horizon}\n')
    f.write(f'multi_horizon = {multi_horizon}\n')
    f.write(f'out_feat = {out_feat}\n')
    f.write(f'lstm_units = {lstm_units}\n')
    f.write(f'patience = {patience}\n')
    f.write(f'learning_rate={learning_rate}\n')
    f.write(f'adj_select = {adj_select} # adj matrix selection : 0 for orig, 1 for rand, 2 for identity\n') 
    f.write(f'naive MSE = {naive_mse} and model MSE = {model_mse}')
    f.close()

for i in range(np.shape(y_test)[2]):
    plt.figure(figsize=(18, 6))
    plt.plot(y_test[:, 0, i])
    plt.plot(y_pred[:, 0, i])
    plt.legend(["actual", "forecast"])
    fig_name = f'Results/test_number_{test_number}/feature_{i}.png'
    plt.savefig(fig_name)

