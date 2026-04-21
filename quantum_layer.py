import pennylane as qml
import torch
import torch.nn as nn


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=10, n_layers=4):
        super().__init__()

        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):

            # DATA RE-UPLOADING
            for l in range(n_layers):

                qml.AngleEmbedding(inputs, wires=range(n_qubits))

                for i in range(n_qubits):
                    qml.RY(weights[l, i], wires=i)

                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)