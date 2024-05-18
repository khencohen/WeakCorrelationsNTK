import pickle
import os

import matplotlib.pyplot as plt
from jax import jit
from jax import grad
from jax import random
import jax, jax.numpy as jnp

from jax import jacfwd, jacrev, grad
from jax import grad, hessian

import jax.numpy as jnp
from jax.nn import log_softmax
from jax.example_libraries import optimizers

import tensorflow_datasets as tfds

import neural_tangents as nt
from neural_tangents import stax
from neural_tangents import taylor_expand
from tqdm import tqdm

from div_corr_utils import DivCorr


# https://github.com/google/neural-tangents/blob/main/notebooks/function_space_linearization.ipynb
# https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/function_space_linearization.ipynb#scrollTo=J-8i_4KD7o5s
# Using https://github.com/google/neural-tangents/blob/main/notebooks/function_space_linearization.ipynb


def process_data(data_chunk, architecture='Dense', twoclass=True):
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

    if twoclass:
        label = jnp.eye(2)[label]
    else:
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


def get_emp_ntk(input_size=784, hidden_size=128, output_size=10, batch_size=64,
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
                   batch_size=batch_size, device_count=0)

    return ntk, params, f


def get_db(dataset_size=64, architecture='Dense', dataset='MNIST', twoclass=True):
    if dataset == 'MNIST':
        ds_train = tfds.as_numpy(
            tfds.load('mnist', split=['train'], batch_size=-1))[0]
    elif dataset == 'FashionMNIST':
        ds_train = tfds.as_numpy(
            tfds.load('fashion_mnist', split=['train'], batch_size=-1))[0]
    elif dataset == 'CIFAR10':
        ds_train = tfds.as_numpy(
            tfds.load('cifar10', split=['train'], batch_size=-1))[0]

    # Filter the dataset
    if twoclass:
        # Take only two classes - 0 and somthing else
        db_filter = (ds_train['label'] == 0) | (ds_train['label'] == 7)
    else:
        db_filter = jnp.ones(ds_train['label'].shape, dtype=bool)

    filtered_db = {}
    filtered_db['image'] = ds_train['image'][db_filter]
    filtered_db['label'] = ds_train['label'][db_filter]

    # Shuffle the dataset and take the first dataset_size samples with 80% for training
    # and 20% for testing
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    perm = random.permutation(subkey, filtered_db['image'].shape[0])
    filtered_db['image'] = filtered_db['image'][perm]
    filtered_db['label'] = filtered_db['label'][perm]

    train_size = int(0.8 * dataset_size)
    total_size = dataset_size

    train_db = {'image': filtered_db['image'][:train_size],
                'label': filtered_db['label'][:train_size]}
    test_db = {'image': filtered_db['image'][train_size:total_size],
               'label': filtered_db['label'][train_size:total_size]}

    train = process_data(train_db, architecture=architecture, twoclass=twoclass)
    test = process_data(test_db, architecture=architecture, twoclass=twoclass)
    return train, test


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def run_ntk_training(dataset_size=64, hidden_size=128,
                     output_size=10, num_layers=3, activation='relu',
                     learning_rate=1e-1, num_of_epochs=1000, architecture='Dense',
                     dataset='MNIST', num_of_corr_calculations=5,
                     batch_size = 64,
                     num_params_per_layer=10):
    train, test = get_db(dataset_size=dataset_size,
                         architecture=architecture,
                         dataset=dataset,
                         twoclass=(output_size == 2))
    input_size = train['image'].shape[-1]

    ntk, params, f = get_emp_ntk(input_size=input_size,
                                 hidden_size=hidden_size,
                                 output_size=output_size,
                                 num_layers=num_layers,
                                 batch_size=batch_size,
                                 activation=activation,
                                 architecture=architecture)

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

    # predictor_test = nt.predict.gradient_descent_mse(g_td, test['label'],
    #                                                  learning_rate=learning_rate)
    # fx_test = f(params, test['image'])

    # Create second order approximation for the ntk

    # init_params = params.copy()
    # lin_plus_quad = lambda x, W: taylor_expand(lambda params: f(params, x), init_params, 2)(W)

    # Train the network.
    # print ('Time\tLoss\tLinear Loss\tQuad Loss')

    X, Y = train['image'], train['label']
    # XT, YT = test['image'], test['label']
    training_steps = jnp.arange(num_of_epochs)
    predictions = predictor(training_steps, fx_train)
    # predictions_test = predictor_test(training_steps, fx_test)

    print_every = 1
    loop = tqdm(training_steps)

    list_of_epochs_ld = []

    list_of_train_ld = []
    list_of_train_nn_lin_loss = []
    list_of_train_lin_loss = []

    list_of_test_ld = []
    list_of_test_lin_loss = []
    list_of_test_nn_lin_loss = []

    lis_of_df_epochs = []
    lis_of_df_corr12 = []
    lis_of_df_corr13 = []
    C = lambda x, y: jnp.sqrt(jnp.mean((x - y) ** 2))
    # C = lambda x, y: jnp.abs(x - y).mean()
    for i in loop:
        params = get_params(state)
        state = opt_apply(i, grad_loss(params, X, Y), state)

        if i % print_every == 0:
            ### Train Loss ###
            train_exact_loss = loss(f(params, X), Y)
            train_linear_loss = loss(predictions[i], Y)
            loop.set_description('Exact Loss: {:.4f}, Linear Loss: {:.4f}'.format(train_exact_loss, train_linear_loss))

            list_of_train_ld.append(C(train_exact_loss, train_linear_loss))
            list_of_train_nn_lin_loss.append(train_exact_loss)
            list_of_train_lin_loss.append(train_linear_loss)

            ### Test Loss ###
            # Calculate the output probabilities
            # test_exact_output = f(params, XT)
            # test_lin_output = predictions_test[i]
            # test_exact_prob = jnp.exp(log_softmax(test_exact_output))
            # test_lin_prob = jnp.exp(log_softmax(test_lin_output))
            # test_exact_loss = jnp.mean(jnp.sum(test_exact_prob * YT, axis=1))
            # test_linear_loss = jnp.mean(jnp.sum(test_lin_prob * YT, axis=1))
            # # quad_loss = loss(lin_plus_quad(X, params), Y)
            # loop.set_description('Exact Loss: {:.4f}, Linear Loss: {:.4f}'.format(test_exact_loss, test_linear_loss))



            # test_exact_loss = loss(f(params, XT), YT)
            # test_linear_loss = loss(predictions_test[i], YT)
            #
            # list_of_test_ld.append(C(test_exact_loss, test_linear_loss))
            # list_of_test_nn_lin_loss.append(test_exact_loss)
            # list_of_test_lin_loss.append(test_linear_loss)

            list_of_epochs_ld.append(i)


        if i % (num_of_epochs // num_of_corr_calculations) == 0 or i == num_of_epochs - 1:
            divcorr = DivCorr(f, params, X, num_params_per_layer=num_params_per_layer)
            dfdidx = divcorr.get_df_didx(dx=1e-3)
            d2fdidx2 = divcorr.get_d2f_didx2(dx=1e-3)
            df_corr12 = divcorr.get_derivatives_correlation_2order(dfdidx, d2fdidx2)  # * (learning_rate**2)
            lis_of_df_epochs.append(i)
            lis_of_df_corr12.append(df_corr12)

            # d3fdidx3 = divcorr.get_d3f_didx3(dx=1e-3)
            # df_corr13 = divcorr.get_derivatives_correlation_3order(dfdidx, d3fdidx3)
            # lis_of_df_corr13.append(df_corr13)

    # Print the gradient of the network with respect to the weights
    # first_derivative = grad(lambda params: loss(f(params, X), Y))(params)

    # Calculate the Hessian of the network with respect to the weights
    # def hessian(f):
    #     return jacfwd(jacrev(f))
    # H = hessian(f)(params)

    # def unflatten(flat, tree):
    #     shapes = [x.shape for x in flat]
    #     sizes = [x.size for x in flat]
    #     sizes = jnp.array(sizes)
    #     idxs = jnp.cumsum(sizes)[:-1]
    #     parts = [flat[idxs[i]:idxs[i + 1]].reshape(shape) for i, shape in enumerate(shapes)]
    #     return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tree), parts)

    # create_folder('data')
    # # Save divcorr for later use
    # with open('data/divcorr.pkl', 'wb') as f:
    #     pickle.dump(divcorr, f)
    #
    # del divcorr
    # divcorr = DivCorr(None, [],None, num_params_per_layer=6)
    # # Load divcorr
    # with open('data/divcorr.pkl', 'rb') as f:
    #     divcorr = pickle.load(f)

    # d3fdidx3 = divcorr.get_d3f_didx3(dx=1e-1)

    # Calculate correlations:

    # Calculate div f with finite differences

    # H = hessian(lambda params: loss(f(params, X), Y))(params)
    # # Convert to numpy
    # first_derivative = flatten(first_derivative)[0]
    # H = flatten(H)[0]

    return list_of_train_ld, list_of_train_nn_lin_loss, list_of_train_lin_loss, list_of_epochs_ld, \
        list_of_test_ld, list_of_test_nn_lin_loss, list_of_test_lin_loss, \
        lis_of_df_epochs, lis_of_df_corr12, lis_of_df_corr13


def run_single_experiment(dataset='MNIST'):
    learning_rate = 1e-3  # Learning rate for the optimizer
    num_of_epochs = 20000  # Number of epochs to train the network
    dataset_size = int(512 / 0.8)  # Number of samples to use from the dataset
    # dataset_size = int(128 / 0.8)  # Number of samples to use from the dataset
    # hidden_size = 32  # Number of neurons in the hidden layer
    hidden_size = 128  # Number of neurons in the hidden layer
    # output_size = 10  # Number of classes
    output_size = 2  # Number of classes
    num_layers = 2  # Number of hidden layers
    activation = 'relu'  # Activation function
    architecture = 'Dense'  # Architecture of the network
    num_params_per_layer = 15

    results = run_ntk_training(
        dataset_size=dataset_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        activation=activation,
        learning_rate=learning_rate,
        num_of_epochs=num_of_epochs,
        architecture=architecture,
        dataset=dataset,
        num_of_corr_calculations=1,  # Calculating once in the end
        num_params_per_layer=num_params_per_layer
    )

    list_of_train_ld, list_of_train_nn_lin_loss, list_of_train_lin_loss, list_of_epochs_ld, \
        list_of_test_ld, list_of_test_nn_lin_loss, list_of_test_lin_loss, \
        list_of_df_epochs, list_of_df_corr12, list_of_df_corr13 = results

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(list_of_epochs_ld, list_of_train_nn_lin_loss, label='Train Exact Loss')
    plt.plot(list_of_epochs_ld, list_of_train_lin_loss, label='Train Linear Loss')
    # plt.plot(list_of_epochs_ld, list_of_test_nn_lin_loss, label='Test Exact Loss')
    # plt.plot(list_of_epochs_ld, list_of_test_lin_loss, label='Test Linear Loss')
    plt.title('Train Losses')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(list_of_epochs_ld, list_of_train_ld, label='Train Loss Difference')
    # plt.plot(list_of_epochs_ld, list_of_test_ld, label='Test Loss Difference')
    plt.title(r'$C(F, F_{lin})$')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plt.subplot(1, 3, 3)
    # plt.plot(list_of_df_epochs, list_of_df_corr12, '-o')
    # plt.title(r'$\rho(\partial_{i} f, \partial_{i} \partial_{j} f)$')
    # plt.xlabel('Epochs')
    # plt.ylabel('Correlation')
    # plt.show()



def run_width_experiment(list_of_widths, dataset='MNIST',
                         dataset_size=256, num_of_epochs=20000,
                         avg_factor=3, num_layers=3,
                         num_params_per_layer=10):
    lr0 = 1e-3  # Learning rate for the optimizer
    output_size = 2  # Number of classes
    activation = 'relu'  # Activation function
    architecture = 'Dense'  # Architecture of the network

    # learning_rate_fn = lambda h: lr0 * (128/jnp.sqrt(h))
    learning_rate_fn = lambda h: lr0

    # tot_list_of_test_final_C = []
    tot_list_of_train_final_C = []
    tot_list_of_df_corr12 = []
    for hidden_size in list_of_widths:
        # avg_ld_test = 0.0
        avg_corr12 = 0.0
        avg_ld_train = 0.0
        for _ in range(avg_factor):
            results = run_ntk_training(
                dataset_size=dataset_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                activation=activation,
                learning_rate=learning_rate_fn(hidden_size),
                num_of_epochs=num_of_epochs,
                architecture=architecture,
                dataset=dataset,
                num_of_corr_calculations=1,  # Calculating once in the end
                num_params_per_layer=num_params_per_layer
            )

            list_of_train_ld, list_of_train_nn_lin_loss, list_of_train_lin_loss, list_of_epochs_ld, \
                list_of_test_ld, list_of_test_nn_lin_loss, list_of_test_lin_loss, \
                list_of_df_epochs, list_of_df_corr12, list_of_df_corr13 = results

            avg_ld_train += list_of_train_ld[-1]
            # avg_ld_test += list_of_test_ld[-1]
            avg_corr12 += list_of_df_corr12[-1]

        # tot_list_of_test_final_C.append(avg_ld_test)
        tot_list_of_train_final_C.append(avg_ld_train)
        tot_list_of_df_corr12.append(avg_corr12)

    plt.figure(figsize=(12, 6))
    plt.suptitle('Width Experiment - {}'.format(dataset))
    plt.subplot(1, 2, 1)
    # plt.plot(list_of_widths, tot_list_of_test_final_C, label='Test Loss')
    plt.plot(list_of_widths, tot_list_of_train_final_C, label='Train Loss')
    plt.title(r'$C(F, F_{lin})$ vs. Width')
    plt.xlabel('Width')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(list_of_widths, tot_list_of_df_corr12)
    plt.title(r'$\rho(\partial_{i} f, \partial_{i} \partial_{j} f)$ vs. Width')
    plt.xlabel('Width')
    plt.ylabel('Correlation')

    plt.savefig('results/width_experiment_{}.png'.format(dataset), dpi=600)
    plt.show()

    return


if __name__ == '__main__':
    create_folder('results')

    # run_single_experiment(dataset='MNIST')
    # exit()

    for dataset in ['CIFAR10']:  # , 'FashionMNIST', 'MNIST']:
        run_width_experiment(
            list_of_widths=[32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192],
            dataset=dataset,
            dataset_size=int(512 / 0.8),
            num_of_epochs=5_000,
            avg_factor=1,
            num_layers=2,
            num_params_per_layer=15
        )

#
#
# def run_activation_experiment(list_of_activations):
#     learning_rate = 1e0  # Learning rate for the optimizer
#     num_of_epochs = 3000  # Number of epochs to train the network
#     dataset_size = 256  # Number of samples to use from the dataset
#     hidden_size = 128  # Number of neurons in the hidden layer
#     output_size = 10  # Number of classes
#     num_layers = 3  # Number of hidden layers
#     architecture = 'Dense'  # Architecture of the network
#     dataset = 'MNIST'  # Dataset to use
#
#     list_of_final_C = []
#     for activation in list_of_activations:
#         results = run_ntk_training(
#             dataset_size=dataset_size,
#             hidden_size=hidden_size,
#             output_size=output_size,
#             num_layers=num_layers,
#             activation=activation,
#             learning_rate=learning_rate,
#             num_of_epochs=num_of_epochs,
#             architecture=architecture,
#             dataset=dataset
#         )
#
#         list_of_ld, list_of_nn_lin_loss, list_of_lin_loss, list_of_epochs_ld = results
#         list_of_final_C.append(list_of_ld[-1])
#
#     plt.figure()
#     # Show bars
#     plt.bar(list_of_activations, list_of_final_C)
#     plt.title(r'$C(F, F_{lin})$ vs. Activation Function')
#     plt.xlabel('Activation Function')
#     plt.ylabel('Loss')
#     plt.show()
#
#
# def run_architecture_experiment(list_of_architectures):
#     learning_rate = 1e0  # Learning rate for the optimizer
#     num_of_epochs = 3000  # Number of epochs to train the network
#     dataset_size = 64  # Number of samples to use from the dataset
#     output_size = 10  # Number of classes
#     num_layers = 3  # Number of hidden layers
#     activation = 'relu'  # Activation function
#     dataset = 'MNIST'
#
#     list_of_final_C = []
#     for architecture in list_of_architectures:
#         hidden_size = 128 if architecture == 'Dense' else 32
#         results = run_ntk_training(
#             dataset_size=dataset_size,
#             hidden_size=hidden_size,
#             output_size=output_size,
#             num_layers=num_layers,
#             activation=activation,
#             learning_rate=learning_rate,
#             num_of_epochs=num_of_epochs,
#             architecture=architecture,
#             dataset=dataset
#         )
#
#         list_of_ld, list_of_nn_lin_loss, list_of_lin_loss, list_of_epochs_ld = results
#         list_of_final_C.append(list_of_ld[-1])
#
#     plt.figure()
#     # Show bars
#     plt.bar(list_of_architectures, list_of_final_C)
#     plt.title(r'$C(F, F_{lin})$ vs. Architecture')
#     plt.xlabel('Architecture')
#     plt.ylabel('Loss')
#     plt.show()

#
# def run_dataset_experiment(list_of_datasets):
#     learning_rate = 1e0  # Learning rate for the optimizer
#     num_of_epochs = 3000  # Number of epochs to train the network
#     dataset_size = 64  # Number of samples to use from the dataset
#     hidden_size = 128  # Number of neurons in the hidden layer
#     output_size = 10  # Number of classes
#     num_layers = 3  # Number of hidden layers
#     activation = 'relu'  # Activation function
#     architecture = 'Dense'  # Architecture of the network
#
#     list_of_final_C = []
#     for dataset in list_of_datasets:
#         results = run_ntk_training(
#             dataset_size=dataset_size,
#             hidden_size=hidden_size,
#             output_size=output_size,
#             num_layers=num_layers,
#             activation=activation,
#             learning_rate=learning_rate,
#             num_of_epochs=num_of_epochs,
#             architecture=architecture,
#             dataset=dataset
#         )
#
#         list_of_ld, list_of_nn_lin_loss, list_of_lin_loss, list_of_epochs_ld = results
#         list_of_final_C.append(list_of_ld[-1])
#
#     plt.figure()
#     # Show bars
#     plt.bar(list_of_datasets, list_of_final_C)
#     plt.title(r'$C(F, F_{lin})$ vs. Dataset')
#     plt.xlabel('Dataset')
#     plt.ylabel('Loss')
#     plt.show()
#


# def run_nol_experiment(list_of_nol, dataset='MNIST', dataset_size=256):
#     learning_rate = 1e0  # Learning rate for the optimizer
#     num_of_epochs = 3000  # Number of epochs to train the network
#     hidden_size = 128  # Number of neurons in the hidden layer
#     output_size = 10  # Number of classes
#     activation = 'relu'  # Activation function
#     architecture = 'Dense'  # Architecture of the network
#
#     list_of_final_C = []
#     for num_layers in list_of_nol:
#         results = run_ntk_training(
#             dataset_size=dataset_size,
#             hidden_size=hidden_size,
#             output_size=output_size,
#             num_layers=num_layers,
#             activation=activation,
#             learning_rate=learning_rate,
#             num_of_epochs=num_of_epochs,
#             architecture=architecture,
#             dataset=dataset
#         )
#
#         list_of_ld, list_of_nn_lin_loss, list_of_lin_loss, list_of_epochs_ld = results
#         list_of_final_C.append(list_of_ld[-1])
#
#     plt.figure()
#     plt.plot(list_of_nol, list_of_final_C)
#     plt.title(r'$C(F, F_{lin})$ vs. Number of Layers')
#     plt.xlabel('Number of Layers')
#     plt.ylabel('Loss')
#     plt.show()
