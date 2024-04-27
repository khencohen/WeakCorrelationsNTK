import matplotlib.pyplot as plt
from jax import jit
from jax import grad
from jax import random

import jax.numpy as jnp
from jax.nn import log_softmax
from jax.example_libraries import optimizers

import tensorflow_datasets as tfds

import neural_tangents as nt
from neural_tangents import stax
from neural_tangents import taylor_expand
from tqdm import tqdm


# https://github.com/google/neural-tangents/blob/main/notebooks/function_space_linearization.ipynb
# https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/function_space_linearization.ipynb#scrollTo=J-8i_4KD7o5s
# Using https://github.com/google/neural-tangents/blob/main/notebooks/function_space_linearization.ipynb


def process_data(data_chunk, architecture='Dense'):
    """Flatten the images and one-hot encode the labels."""
    image, label = data_chunk['image'], data_chunk['label']
    samples = image.shape[0]

    if architecture == 'Dense':
        image = jnp.array(jnp.reshape(image, (samples, -1)), dtype=jnp.float32)
        # If color images, convert to grayscale
        if image.shape[-1] == 3:
            image = jnp.dot(image, jnp.array([0.299, 0.587, 0.114]))
    elif architecture == 'Conv2D':
        image = jnp.array(image, dtype=jnp.float32)
    image = (image - jnp.mean(image)) / jnp.std(image)
    label = jnp.eye(10)[label]

    return {'image': image, 'label': label}


@optimizers.optimizer
def momentum(learning_rate, momentum=0.9):
    """A standard momentum optimizer for testing.

    Different from `jax.example_libraries.optimizers.momentum` (Nesterov).
    """
    learning_rate = optimizers.make_schedule(learning_rate)

    def init_fn(x0):
        v0 = jnp.zeros_like(x0)
        return x0, v0

    def update_fn(i, g, state):
        x, velocity = state
        velocity = momentum * velocity + g
        x = x - learning_rate(i) * velocity
        return x, velocity

    def get_params(state):
        x, _ = state
        return x

    return init_fn, update_fn, get_params


def get_emp_ntk(input_size=784, hidden_size=128, output_size=10,
                num_layers=3, activation='relu', architecture='Dense'):
    act = {'relu': stax.Relu(),
           'erf': stax.Erf(),
           'sigmoid': stax.Sigmoid_like(),
           'gelu': stax.Gelu(),
           'leaky_relu': stax.LeakyRelu(0.1)}[activation]

    if architecture == 'Dense':
        layers_list = [stax.Dense(hidden_size, W_std=1.0, b_std=0.05), act] * num_layers \
                      + [stax.Dense(output_size, W_std=1.0, b_std=0.05)]
    elif architecture == 'Conv2D':
        layers_list = [stax.Conv(hidden_size, (3, 3), W_std=1.0, b_std=0.05), act] * num_layers \
                      + [stax.Dense(output_size, W_std=1.0, b_std=0.05)]

    init_fn, f, _ = stax.serial(*layers_list)

    key = random.PRNGKey(0)
    if architecture == 'Dense':
        _, params = init_fn(key, (-1, input_size))
    elif architecture == 'Conv2D':
        _, params = init_fn(key, (-1, 28, 28, 1))


    # Construct the NTK
    ntk = nt.batch(nt.empirical_ntk_fn(f, vmap_axes=0),
                   batch_size=64, device_count=0)

    return ntk, params, f


def get_db(dataset_size=64, architecture='Dense', dataset='MNIST'):
    if dataset == 'MNIST':
        ds_train, ds_test = tfds.as_numpy(
            tfds.load('mnist:3.*.*', split=['train[:%d]' % dataset_size,
                                            'test[:%d]' % dataset_size],
                      batch_size=-1)
        )
    elif dataset == 'FashionMNIST':
        ds_train, ds_test = tfds.as_numpy(
            tfds.load('fashion_mnist:3.*.*', split=['train[:%d]' % dataset_size,
                                                    'test[:%d]' % dataset_size],
                      batch_size=-1)
        )
    elif dataset == 'CIFAR10':
        ds_train, ds_test = tfds.as_numpy(
            tfds.load('cifar10:3.*.*', split=['train[:%d]' % dataset_size,
                                              'test[:%d]' % dataset_size],
                      batch_size=-1)
        )

    train = process_data(ds_train, architecture=architecture)
    test = process_data(ds_test, architecture=architecture)
    return train, test


def run_ntk_training(dataset_size=64, hidden_size=128,
                     output_size=10, num_layers=3, activation='relu',
                     learning_rate=1e-1, num_of_epochs=1000, architecture='Dense',
                     dataset='MNIST'):

    train, test = get_db(dataset_size=dataset_size, architecture=architecture, dataset=dataset)
    input_size = train['image'].shape[-1]

    ntk, params, f = get_emp_ntk(input_size=input_size, hidden_size=hidden_size,
                                 output_size=output_size, num_layers=num_layers,
                                 activation=activation, architecture=architecture)

    g_dd = ntk(train['image'], None, params)
    g_td = ntk(test['image'], train['image'], params)

    # Create a optimizer and initialize it
    opt_init, opt_apply, get_params = optimizers.sgd(learning_rate)
    state = opt_init(params)

    # Create an MSE loss and a gradient
    loss = lambda fx, y_hat: 0.5 * jnp.mean((fx - y_hat) ** 2)
    grad_loss = jit(grad(lambda params, x, y: loss(f(params, x), y)))

    # Create an MSE predictor and compute the function space values of the network at initialization.
    predictor = nt.predict.gradient_descent_mse(g_dd, train['label'],
                                                learning_rate=learning_rate)
    fx_train = f(params, train['image'])

    predictor_test = nt.predict.gradient_descent_mse(g_td, train['label'],
                                                     learning_rate=learning_rate)
    fx_test = f(params, test['image'])

    # Create second order approximation for the ntk

    # init_params = params.copy()
    # lin_plus_quad = lambda x, W: taylor_expand(lambda params: f(params, x), init_params, 2)(W)

    # Train the network.
    # print ('Time\tLoss\tLinear Loss\tQuad Loss')

    X, Y = train['image'], train['label']
    XT, YT = test['image'], test['label']
    training_steps = jnp.arange(num_of_epochs)
    predictions = predictor(training_steps, fx_train)
    predictions_test = predictor_test(training_steps, fx_test)

    print_every = 1
    loop = tqdm(training_steps)
    list_of_ld = []
    list_of_epochs_ld = []
    list_of_lin_loss = []
    list_of_nn_lin_loss = []
    C = lambda x, y: jnp.sqrt(jnp.mean((x - y) ** 2))
    for i in loop:
        params = get_params(state)
        state = opt_apply(i, grad_loss(params, X, Y), state)

        if i % print_every == 0:
            # exact_loss = loss(f(params, X), Y)
            # linear_loss = loss(predictions[i], Y)

            exact_loss = loss(f(params, XT), YT)
            linear_loss = loss(predictions_test[i], YT)

            # quad_loss = loss(lin_plus_quad(X, params), Y)
            loop.set_description('Exact Loss: {:.4f}, Linear Loss: {:.4f}'.format(exact_loss, linear_loss))

            list_of_ld.append(C(exact_loss, linear_loss))
            list_of_nn_lin_loss.append(exact_loss)
            list_of_lin_loss.append(linear_loss)
            list_of_epochs_ld.append(i)

    return list_of_ld, list_of_nn_lin_loss, list_of_lin_loss, list_of_epochs_ld


def run_single_experiment():
    learning_rate = 1e0  # Learning rate for the optimizer
    num_of_epochs = 3000  # Number of epochs to train the network
    dataset_size = 256  # Number of samples to use from the dataset
    hidden_size = 128  # Number of neurons in the hidden layer
    output_size = 10  # Number of classes
    num_layers = 3  # Number of hidden layers
    activation = 'relu'  # Activation function
    architecture = 'Dense'  # Architecture of the network
    dataset = 'MNIST'       # Dataset to use

    results = run_ntk_training(
        dataset_size=dataset_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        activation=activation,
        learning_rate=learning_rate,
        num_of_epochs=num_of_epochs,
        architecture=architecture,
        dataset=dataset
    )

    list_of_ld, list_of_nn_lin_loss, list_of_lin_loss, list_of_epochs_ld = results

    plt.figure(figsize=(9, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list_of_epochs_ld, list_of_nn_lin_loss, label='Exact Loss')
    plt.plot(list_of_epochs_ld, list_of_lin_loss, label='Linear Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(list_of_epochs_ld, list_of_ld)
    plt.title(r'$||F - F_{lin}||^2$')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()


def run_nol_experiment(list_of_nol):
    learning_rate = 1e0  # Learning rate for the optimizer
    num_of_epochs = 3000  # Number of epochs to train the network
    dataset_size = 256  # Number of samples to use from the dataset
    hidden_size = 128  # Number of neurons in the hidden layer
    output_size = 10  # Number of classes
    activation = 'relu'  # Activation function
    architecture = 'Dense'  # Architecture of the network
    dataset = 'MNIST'       # Dataset to use

    list_of_final_C = []
    for num_layers in list_of_nol:
        results = run_ntk_training(
            dataset_size=dataset_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            activation=activation,
            learning_rate=learning_rate,
            num_of_epochs=num_of_epochs,
            architecture=architecture,
            dataset=dataset
        )

        list_of_ld, list_of_nn_lin_loss, list_of_lin_loss, list_of_epochs_ld = results
        list_of_final_C.append(list_of_ld[-1])

    plt.figure()
    plt.plot(list_of_nol, list_of_final_C)
    plt.title(r'$||F - F_{lin}||^2$ vs. Number of Layers')
    plt.xlabel('Number of Layers')
    plt.ylabel('Loss')
    plt.show()


def run_activation_experiment(list_of_activations):
    learning_rate = 1e0  # Learning rate for the optimizer
    num_of_epochs = 3000  # Number of epochs to train the network
    dataset_size = 256  # Number of samples to use from the dataset
    hidden_size = 128  # Number of neurons in the hidden layer
    output_size = 10  # Number of classes
    num_layers = 3  # Number of hidden layers
    architecture = 'Dense'  # Architecture of the network
    dataset = 'MNIST'       # Dataset to use

    list_of_final_C = []
    for activation in list_of_activations:
        results = run_ntk_training(
            dataset_size=dataset_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            activation=activation,
            learning_rate=learning_rate,
            num_of_epochs=num_of_epochs,
            architecture=architecture,
            dataset=dataset
        )

        list_of_ld, list_of_nn_lin_loss, list_of_lin_loss, list_of_epochs_ld = results
        list_of_final_C.append(list_of_ld[-1])

    plt.figure()
    # Show bars
    plt.bar(list_of_activations, list_of_final_C)
    plt.title(r'$||F - F_{lin}||^2$ vs. Activation Function')
    plt.xlabel('Activation Function')
    plt.ylabel('Loss')
    plt.show()


def run_architecture_experiment(list_of_architectures):
    learning_rate = 1e0  # Learning rate for the optimizer
    num_of_epochs = 3000  # Number of epochs to train the network
    dataset_size = 64  # Number of samples to use from the dataset
    output_size = 10  # Number of classes
    num_layers = 3  # Number of hidden layers
    activation = 'relu'  # Activation function
    dataset = 'MNIST'

    list_of_final_C = []
    for architecture in list_of_architectures:
        hidden_size = 128 if architecture == 'Dense' else 32
        results = run_ntk_training(
            dataset_size=dataset_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            activation=activation,
            learning_rate=learning_rate,
            num_of_epochs=num_of_epochs,
            architecture=architecture,
            dataset=dataset
        )

        list_of_ld, list_of_nn_lin_loss, list_of_lin_loss, list_of_epochs_ld = results
        list_of_final_C.append(list_of_ld[-1])

    plt.figure()
    # Show bars
    plt.bar(list_of_architectures, list_of_final_C)
    plt.title(r'$||F - F_{lin}||^2$ vs. Architecture')
    plt.xlabel('Architecture')
    plt.ylabel('Loss')
    plt.show()


def run_dataset_experiment(list_of_datasets):
    learning_rate = 1e0  # Learning rate for the optimizer
    num_of_epochs = 3000  # Number of epochs to train the network
    dataset_size = 64  # Number of samples to use from the dataset
    hidden_size = 128  # Number of neurons in the hidden layer
    output_size = 10  # Number of classes
    num_layers = 3  # Number of hidden layers
    activation = 'relu'  # Activation function
    architecture = 'Dense'  # Architecture of the network

    list_of_final_C = []
    for dataset in list_of_datasets:
        results = run_ntk_training(
            dataset_size=dataset_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            activation=activation,
            learning_rate=learning_rate,
            num_of_epochs=num_of_epochs,
            architecture=architecture,
            dataset=dataset
        )

        list_of_ld, list_of_nn_lin_loss, list_of_lin_loss, list_of_epochs_ld = results
        list_of_final_C.append(list_of_ld[-1])

    plt.figure()
    # Show bars
    plt.bar(list_of_datasets, list_of_final_C)
    plt.title(r'$||F - F_{lin}||^2$ vs. Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    # run_single_experiment()

    # run_nol_experiment(
    #     list_of_nol=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # )

    # run_activation_experiment(
    #     list_of_activations=['relu', 'erf', 'sigmoid', 'gelu', 'leaky_relu']
    # )

    run_dataset_experiment(
        list_of_datasets=['CIFAR10', 'FashionMNIST', 'MNIST']
    )

    ### Warning! CNN Too slow for some reason: ##
    # run_architecture_experiment(
    #     list_of_architectures=['Conv2D', 'Dense']
    # )
