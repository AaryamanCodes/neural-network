import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.network import NeuralNetwork
from src.optim import SGD

SEED = 0
HIDDEN_SIZE = 32
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.3


def load_data():
    digits = load_digits()
    X, y = digits.data, digits.target
    X = X / 16.0
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def accuracy(net, X, y):
    preds = net.predict(X)
    return (preds == y).mean()


def iterate_minibatches(X, y, batch_size, rng):
    n = X.shape[0]
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def main():
    rng = np.random.default_rng(SEED)
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    net = NeuralNetwork(in_features=64, hidden_features=HIDDEN_SIZE, out_features=10, seed=SEED)
    optimizer = SGD(learning_rate=LEARNING_RATE)

    train_losses = []
    test_losses = []

    for epoch in range(1, EPOCHS + 1):
        epoch_losses = []
        for X_batch, y_batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, rng):
            loss, _ = net.forward(X_batch, y_batch)
            net.backward()
            optimizer.step(net.parameters())
            epoch_losses.append(loss)

        train_loss = np.mean(epoch_losses)
        test_loss, _ = net.forward(X_test, y_test)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 20 == 0 or epoch == 1:
            train_acc = accuracy(net, X_train, y_train)
            test_acc = accuracy(net, X_test, y_test)
            print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | test loss {test_loss:.4f} "
                  f"| train acc {train_acc:.3f} | test acc {test_acc:.3f}")

    final_train_acc = accuracy(net, X_train, y_train)
    final_test_acc = accuracy(net, X_test, y_test)
    print(f"\nFinal: train acc {final_train_acc:.3f}, test acc {final_test_acc:.3f}")
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training loss (manual backprop, from scratch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/loss_curve.png", dpi=120)
    print("Saved outputs/loss_curve.png")

    with open("outputs/results.txt", "w") as f:
        f.write(f"Final train accuracy: {final_train_acc:.4f}\n")
        f.write(f"Final test accuracy:  {final_test_acc:.4f}\n")
        f.write(f"Final train loss:     {train_losses[-1]:.4f}\n")
        f.write(f"Final test loss:      {test_losses[-1]:.4f}\n")
    print("Saved outputs/results.txt")


if __name__ == "__main__":
    main()
