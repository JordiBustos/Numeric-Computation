def update_w_and_b(feature1, feature2, w, b, alpha):
    dl_dw = 0
    dl_db = 0
    N = len(feature1)

    for i in range(N):
        dl_dw += -2 * feature1[i] * (feature2[i] - (w * feature1[i] + b))
        dl_db = -2 * (feature2[i] - (w * feature1[i] + b))

    w = w - (1 / N) * dl_dw * alpha
    b = b - (1 / N) * dl_db * alpha

    return w, b


def train(feature1, feature2, w, b, alpha, epochs):
    for e in range(epochs):
        w, b = update_w_and_b(feature1, feature2, w, b, alpha)

        if e % 100 == 0:
            print(
                "Epoch %d: w = %.2f, b = %.2f loss: "
                % (e, w, b, avg_loss(feature1, feature2, w, b))
            )

    return w, b


def avg_loss(feature1, feature2, w, b):
    N = len(feature1)
    total_error = 0.0

    for i in range(N):
        total_error += (feature2[i] - (w * feature1[i] + b)) ** 2

    return total_error / N


def predict(x, w, b):
    return w * x + b
