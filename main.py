"""
File: main.py
Author: Khen Cohen
Email: khencohen@mail.tau.ac.il
Description: This code is a realization of the paper "Weak Correlations as the Underlying Principle for Linearization of Gradient-Based Learning Systems" by Khen Cohen et al.

This program is distributed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
from jax import random
from jax import grad
import jax.numpy as jnp
from jax.example_libraries import optimizers
import tensorflow_datasets as tfds
import neural_tangents as nt
from neural_tangents import stax
from tqdm import tqdm
from div_corr_utils import DivCorr


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
                     batch_size=64,
                     num_params_per_layer=10, corr_order=2):
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

    C = lambda x, y: jnp.sqrt(0.5 * jnp.mean((x - y) ** 2))
    for i in loop:
        params = get_params(state)
        state = opt_apply(i, grad_loss(params, X, Y), state)

        if i % print_every == 0:
            ### Train Loss ###
            train_exact_loss = loss(f(params, X), Y)
            train_linear_loss = loss(predictions[i], Y)
            loop.set_description('Exact Loss: {:.4f}, Linear Loss: {:.4f}'.format(train_exact_loss, train_linear_loss))

            # list_of_train_ld.append(loss(train_exact_loss, train_linear_loss))
            list_of_train_ld.append(C(train_exact_loss, train_linear_loss))
            list_of_train_nn_lin_loss.append(train_exact_loss)
            list_of_train_lin_loss.append(train_linear_loss)

            ### Test Loss ###
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
            df_corr12 = divcorr.get_derivatives_correlation_2order(dfdidx, d2fdidx2)
            lis_of_df_epochs.append(i)
            lis_of_df_corr12.append(df_corr12)

            if corr_order == 3:
                d3fdidx3 = divcorr.get_d3f_didx3(dx=1e-3)
                df_corr13 = divcorr.get_derivatives_correlation_3order(dfdidx, d3fdidx3)
                lis_of_df_corr13.append(df_corr13)

    return list_of_train_ld, list_of_train_nn_lin_loss, list_of_train_lin_loss, list_of_epochs_ld, \
        list_of_test_ld, list_of_test_nn_lin_loss, list_of_test_lin_loss, \
        lis_of_df_epochs, lis_of_df_corr12, lis_of_df_corr13


def run_single_experiment(dataset='MNIST'):
    learning_rate = 1e0  # Learning rate for the optimizer
    num_of_epochs = 1_000  # Number of epochs to train the network
    dataset_size = int(512 / 0.8)  # Number of samples to use from the dataset
    # dataset_size = int(128 / 0.8)  # Number of samples to use from the dataset
    # hidden_size = 32  # Number of neurons in the hidden layer
    hidden_size = 256  # Number of neurons in the hidden layer
    # output_size = 10  # Number of classes
    output_size = 10  # Number of classes
    num_layers = 1  # Number of hidden layers
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

    plt.figure(figsize=(10, 4))
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

    plt.savefig('results/ntk_train_{}_{}_{}_{}_{}_{}.pdf'.format(dataset, hidden_size, num_layers, activation, num_params_per_layer, output_size), dpi=600)
    plt.show()


def run_width_experiment(list_of_widths, dataset='MNIST',
                         dataset_size=256, num_of_epochs=20000,
                         avg_factor=3, num_layers=3,
                         num_params_per_layer=10, lr0=1e-3,
                         activation='relu', corr_order=2,
                         output_size=2):
    architecture = 'Dense'  # Architecture of the network

    learning_rate_fn = lambda h: lr0

    tot_list_of_train_final_C = []
    tot_list_of_df_corr12 = []
    tot_list_of_df_corr13 = []
    for hidden_size in list_of_widths:
        avg_corr12 = 0.0
        avg_corr13 = 0.0
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
                num_params_per_layer=num_params_per_layer,
                corr_order=corr_order
            )

            list_of_train_ld, list_of_train_nn_lin_loss, list_of_train_lin_loss, list_of_epochs_ld, \
                list_of_test_ld, list_of_test_nn_lin_loss, list_of_test_lin_loss, \
                list_of_df_epochs, list_of_df_corr12, list_of_df_corr13 = results

            avg_ld_train += list_of_train_ld[-1]
            avg_corr12 += list_of_df_corr12[-1]
            if corr_order == 3:
                avg_corr13 += list_of_df_corr13[-1]

        tot_list_of_train_final_C.append(avg_ld_train)
        tot_list_of_df_corr12.append(avg_corr12)
        if corr_order == 3:
            tot_list_of_df_corr13.append(avg_corr13)

    list_of_widths = np.array(list_of_widths)
    tot_list_of_train_final_C = np.array(tot_list_of_train_final_C)
    tot_list_of_df_corr12 = np.array(tot_list_of_df_corr12)

    # Save all data
    np.save('data/widths_{}_{}_{}_{}.npy'.format(dataset, num_layers, activation, corr_order), list_of_widths)
    np.save('data/train_final_C_{}_{}_{}_{}.npy'.format(dataset, num_layers, activation, corr_order),
            tot_list_of_train_final_C)
    np.save('data/df_corr12_{}_{}_{}_{}.npy'.format(dataset, num_layers, activation, corr_order), tot_list_of_df_corr12)

    if corr_order == 3:
        tot_list_of_df_corr13 = np.array(tot_list_of_df_corr13)
        np.save('data/df_corr13_{}_{}_{}_{}.npy'.format(dataset, num_layers, activation, corr_order),
                tot_list_of_df_corr13)

    return list_of_widths, tot_list_of_train_final_C, tot_list_of_df_corr12, tot_list_of_df_corr13


def plot_results(dataset='MNIST', num_layers=3,
                 activation='relu', corr_order=2):
    list_of_widths = np.load('data/widths_{}_{}_{}_{}.npy'.format(dataset, num_layers, activation, corr_order))
    tot_list_of_train_final_C = np.load(
        'data/train_final_C_{}_{}_{}_{}.npy'.format(dataset, num_layers, activation, corr_order))
    tot_list_of_df_corr12 = np.load(
        'data/df_corr12_{}_{}_{}_{}.npy'.format(dataset, num_layers, activation, corr_order))

    plt.plot(list_of_widths, tot_list_of_train_final_C)
    plt.title(r'$\mathcal{C}^{0,2}(F, F_{lin})$' + \
              ' vs. Width (dataset={} layers={} activation={})'.format(dataset, num_layers, activation))
    plt.xlabel('Width')
    plt.ylabel('Loss')
    plt.savefig('results/Cfflin_{}_{}_{}.pdf'.format(dataset, num_layers, activation), dpi=600)
    plt.show()

    plt.plot(list_of_widths, tot_list_of_df_corr12)
    plt.title(r'$\mathcal{C}^{0,2}$' + ' vs. Width (dataset={} layers={} activation={})'.format(dataset, num_layers,
                                                                                                activation))
    plt.xlabel('Width')
    plt.ylabel('Correlation')
    plt.savefig('results/corr12_{}_{}_{}.pdf'.format(dataset, num_layers, activation), dpi=600)
    plt.show()

    if corr_order == 3:
        tot_list_of_df_corr13 = np.load(
            'data/df_corr13_{}_{}_{}_{}.npy'.format(dataset, num_layers, activation, corr_order))

        plt.plot(list_of_widths, tot_list_of_df_corr13)
        plt.title(r'$\mathcal{C}^{0,3}$' + ' vs. Width (dataset={} layers={} activation={})'.format(dataset, num_layers,
                                                                                                    activation))
        plt.xlabel('Width')
        plt.ylabel('Correlation')
        plt.savefig('results/corr13_{}_{}_{}_corr13.pdf'.format(dataset, num_layers, activation), dpi=600)
        plt.show()

    return



if __name__ == '__main__':
    create_folder('results')
    create_folder('data')

    # Run a single simulation
    run_single_experiment(dataset='CIFAR10')

    # Run the width experiment
    for dataset in ['MNIST', 'FashionMNIST', 'CIFAR10']:
        run_width_experiment(
            list_of_widths=[8, 16, 32, 64, 128, 256, 512, 1024],
            dataset=dataset,
            dataset_size=int(128 / 0.8),
            num_of_epochs=1_000,
            avg_factor=1,
            num_layers=1,
            num_params_per_layer=10,
            lr0=1e0,
            output_size=10,
        )
        plot_results(dataset=dataset,
                     num_layers=1,
                     activation='relu',
                     corr_order=2)
