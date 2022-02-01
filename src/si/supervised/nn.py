from src.si.util.activation import *
from src.si.util.metrics import mse, mse_prime
from src.si.supervised.Model import Model
from src.si.util.im2col import pad2D, im2col, col2im


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error, lr):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, inputsize, outputsize):
        super().__init__()
        self.weights = np.random.rand(inputsize, outputsize) - 0.5
        self.bias = np.zeros((1, outputsize))

    def setWeights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        # dE/dW ? X.T * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # dE/dB = dE/dY
        bias_error = np.sum(output_error, axis=0)
        # dE/dX
        input_error = np.dot(output_error, self.weights.T)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error


class Activation(Layer):
    def __init__(self, activation):
        super().__init__()
        self.function = activation

    def forward(self, input):
        self.input = input
        self.output = self.function(input)
        return self.output

    def backward(self, output_error, lr):
        return np.multiply(self.function.prime(self.input), output_error)


class NN(Model):
    def __init__(self, epochs=1000, lr=0.001, verbose=True):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    def add(self, layer):
        self.layers.append(layer)

    def useLoss(self, loss=None, loss_prime=None):
        self.loss = loss
        self.loss_prime = loss_prime

    def fit(self, dataset):
        X, Y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = X
            for layer in self.layers:
                output = layer.forward(output)

            # backward propagation
            error = self.loss_prime(Y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            err = self.loss(Y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f"epoch {epoch+1}/{self.epochs} error={err}")
        print("\r", f"epoch {epoch + 1}/{self.epochs} error = {err}")
        self.is_fited = True

    def predict(self, input_data):
        assert self.is_fited
        output = input_data
        for layer in self.layers:
            output= layer.forward(output)
        return output

    def cost(self, X=None, y=None):
        assert self.is_fited, 'Model must be fitted'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y, output)


class Flatten(Layer):
    def forward(self, input):
        self.input_shape= input.shape
        output = input.reshape(input.shape[0], -1)
        return output

    def backward(self, output_error, lr):
        return output_error.reshape(self.input_shape)


class Conv2D(Layer):
    def __init__(self, input_shape, kernel_shape, layer_depth, stride=1, padding=0):
        super().__init__()
        self.input_shape = input_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride
        self.padding = padding

        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1],
                                      self.in_ch, self.out_ch) - 0.5

        self.bias = np.zeros((self.out_ch,1))

    def forward(self, input_data):
        s = self.stride
        self.X_shape = input_data.shape
        _, p = pad2D(input_data, self.padding, self.weights.shape[:2], s)

        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = input_data.shape

        # compute the dimensions of the convolution output
        out_rows = int((in_rows + pr1 + pr2-fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 -fc) / s + 1)

        # convert X and w into the appropriate 2D matrices and take their product
        self.X_col, _ = im2col(input_data, self.weights.shape, p, s)
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)

        output_data = (W_col @ self.X_col + self.bias).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)
        return output_data

    def backward(self, output_error, learning_rate):
        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding

        db = np.sum(output_error, axis=(0, 1, 2))
        db = db.reshape(out_ch,)

        dout_reshaped = output_error.transpose(1, 2, 3, 0).reshape(out_ch, -1)
        dW = dout_reshaped @ self.X_col.T
        dW = dW.reshape(self.weights.shape)

        W_reshape = self.weights.reshape(out_ch, -1)
        dx_col = W_reshape. T @ dout_reshaped
        input_error = col2im(dx_col, self.X_shape, self.weights.shape, (p, p, p, p), self.stride)

        self.weights -= learning_rate*dW
        self.bias -= learning_rate*db

        return input_error


class MaxPooling(Layer):
    def __init__(self, pool_size, stride=2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
        self.X_copy = None
        self.X_shape = None

    def pool(self, x_col):
        raise NotImplementedError

    def dpool(self, dx_col, dout_col, cache):
        raise NotImplementedError

    def forward(self, input):
        # n, h, w, d = input.shape  # comprimento, altura e numero das imagens
        # h_out = (h - self.size) / self.stride + 1
        # w_out = (w - self.size) / self.stride + 1
        # if not w_out.is_intiger() or not h_out.is_intiger():
        #     raise Exception("Invalid output dimension")
        # h_out, w_out = int(h_out), int(w_out)
        # X_reshaped = input.reshape()
        # self.X_col = im2col(X_reshaped, self.size, pad=0, stride=self.stride)

        # out, self.max_idx = self.pool(self.X_col)
        # out = out.reshape(h_out, w_out, n, d)
        # out = out.transpose(3, 2, 0, 1)
        # return out
        self.X_copy = np.array(input, copy=True)
        self.X_shape = input.shape
        n, h, w, d = input.shape
        height_pool, width_pool = self.pool_size
        h_out = 1 + (h - height_pool) // self.stride
        w_out = 1 + (w - width_pool) // self.stride
        # if not w_out.is_intiger() or not h_out.is_intiger():
        #    raise Exception("Invalid output dimension")
        h_out, w_out = int(h_out), int(w_out)
        output = np.zeros((n, h_out, w_out, d))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + height_pool
                w_start = j * self.stride
                w_end = w_start + width_pool
                a_prev_slice = input[:, h_start:h_end, w_start:w_end, :]
                self.save_mask(x=a_prev_slice, cords=(i, j))
                output[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))
        return output

    def backward(self, output_error, lr):
        # n, w, h, d = self.X_shape
        # dX_col = np.zeros_like(self.X_col)
        # dout_col = output_error.transpose(2, 3, 0, 1).ravel()
        # dX = self.dpool(dX_col, dout_col, self.max_idx)
        # dX = col2im(dX_col, (n*d, h, w, 1), self.pool_size, pad=0, stride=self.stride)
        # dX = dX.reshape(self.X_shape)
        # return dX
        output = np.zeros_like(self.X_copy)
        _, h_out, w_out, _ = output_error.shape
        h_pool, w_pool = self.pool_size

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                output[:, h_start:h_end, w_start:w_end, :] += output_error[:, i:i + 1, j:j + 1, :] * self.cache[(i, j)]
        return output

    def save_mask(self, x, cords):
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self.cache[cords] = mask


class AvgPooling(MaxPooling):
    def __init__(self, pool_size, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
        self.X_copy = None
        self.X_shape = None

    def forward(self, input):
        self.X_shape = input.shape
        n, h, w, d = input.shape
        height_pool, width_pool = self.pool_size
        h_out = 1 + (h - height_pool) // self.stride
        w_out = 1 + (w - width_pool) // self.stride
        h_out, w_out = int(h_out), int(w_out)
        output = np.zeros((n, h_out, w_out, d))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + height_pool
                w_start = j * self.stride
                w_end = w_start + width_pool
                a_prev_slice = input[:, h_start:h_end, w_start:w_end, :]
                self.save_mask(x=a_prev_slice, cords=(i, j))
                output[:, i, j, :] = np.mean(a_prev_slice, axis=(1, 2))
        return output