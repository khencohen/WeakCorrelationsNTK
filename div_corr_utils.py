import numpy as np
import jax, jax.numpy as jnp


def flatten(v):
    def f(v):
        leaves, _ = jax.tree_util.tree_flatten(v)
        return jnp.concatenate([x.ravel() for x in leaves])

    out, pullback = jax.vjp(f, v)
    return out, lambda x: pullback(x)[0]


class DivCorr:
    '''
    This class is used to calculate the derivatives of
    a function f with respect to the parameters of the function.
    And then calculate the correlation between the derivatives.
    '''

    def __init__(self, f, params, X, num_params_per_layer=100):
        self.params_idx = []
        self.f = f
        self.params = params
        self.params_as_list = []
        self.X = X
        self.num_layers = len(params)
        self.num_params_per_layer = num_params_per_layer

        # Set params as a list
        self.set_params_as_list()

        # Set self.params_idx:
        self.sample_params()

    def set_params_as_list(self):
        self.params_as_list = []
        for i in range(len(self.params)):
            if len(self.params[i]) <= 1:
                self.params_as_list.append(tuple([]))
                # self.params_as_list.append(self.params[i])
                continue

            p = [self.params[i][0], self.params[i][1]]
            self.params_as_list.append(p)

    def sample_params(self):
        # Hold the indices of the parameters to calculate the derivatives
        self.params_idx = []
        for l in range(self.num_layers):
            # For activation functions
            if len(self.params[l]) <= 1:
                continue

            weights, bias = self.params[l]
            # Choose randomly num_params_per_layer parameters from each layer
            widx = np.random.choice(weights.size, self.num_params_per_layer)
            bidx = np.random.choice(bias.size, self.num_params_per_layer)

            self.params_idx += list(np.array([np.full_like(widx, l), np.zeros_like(widx), widx]).T)
            self.params_idx += list(np.array([np.full_like(bidx, l), np.ones_like(bidx), bidx]).T)

    def from_list2tuples_params(self, params_list):
        # Convert a list of parameters to a list of tuples
        for i in range(len(params_list)):
            if len(params_list[i]) <= 1:
                continue
            params_list[i] = tuple(params_list[i])
        return params_list

    def get_params_plus_dx(self, idx_list, dx=1e-3):
        # This function returns the parameters with a small dx added to the parameters
        # First copy the paramerters
        params_plus_dx = self.params_as_list.copy()
        for l, matbias, idx in idx_list:
            w, h = params_plus_dx[l][matbias].shape
            val = params_plus_dx[l][matbias][(idx // h, idx % h)]
            params_plus_dx[l][matbias] = params_plus_dx[l][matbias].at[(idx // h, idx % h)].set(val + dx)

        return self.from_list2tuples_params(params_plus_dx)

    def get_df_didx(self, dx = 1e-3):
        # Running over all the parameters and calculating the derivative of f with respect to the parameters
        div_f_list = []
        for idx in self.params_idx:
            params_plus_dx = self.get_params_plus_dx([idx], dx=dx)
            params_minus_dx = self.get_params_plus_dx([idx], dx=-dx)
            f_plus_dx = self.f(params_plus_dx, self.X)
            f_minus_dx = self.f(params_minus_dx, self.X)
            div_f = (f_plus_dx - f_minus_dx) / (2 * dx)
            div_f_list.append(div_f)

        div_f_list = jnp.array(div_f_list)
        return div_f_list

    def get_d2f_didx2(self, dx=1e-3):
        # Running over all the parameters and calculating the
        # derivative of f with respect to the parameters idx1 idx2
        div_f_list1 = []
        for idx1 in self.params_idx:
            div_f_list2 = []
            for idx2 in self.params_idx:
                params_plus_dx = self.get_params_plus_dx(
                    [idx1, idx2], dx=dx)
                params_minus_dx = self.get_params_plus_dx(
                    [idx1, idx2], dx=-dx)
                f_plus_dx = self.f(params_plus_dx, self.X)
                f_minus_dx = self.f(params_minus_dx, self.X)
                div_f = (f_plus_dx - 2 * self.f(self.params, self.X) + f_minus_dx) / (dx ** 2)
                div_f_list2.append(div_f)
            div_f_list1.append(div_f_list2)

        div_f_list1 = jnp.array(div_f_list1)
        return div_f_list1

    def get_d3f_didx3(self, dx=1e-3):
        # Running over all the parameters and calculating the
        # derivative of f with respect to the parameters idx1 idx2 idx3
        div_f_list1 = []
        for idx1 in self.params_idx:
            div_f_list2 = []
            for idx2 in self.params_idx:
                div_f_list3 = []
                for idx3 in self.params_idx:
                    params_plus_dx = self.get_params_plus_dx(
                        [idx1, idx2, idx3], dx=dx)
                    params_minus_dx = self.get_params_plus_dx(
                        [idx1, idx2, idx3], dx=-dx)
                    f_plus_dx = self.f(params_plus_dx, self.X)
                    f_minus_dx = self.f(params_minus_dx, self.X)
                    div_f = (f_plus_dx - 3 * self.f(self.params, self.X) + 3 * f_minus_dx - f_minus_dx) / (dx ** 3)
                    div_f_list3.append(div_f)
                div_f_list2.append(div_f_list3)
            div_f_list1.append(div_f_list2)

        div_f_list1 = jnp.array(div_f_list1)
        return div_f_list1


    def get_derivatives_correlation_2order(self, dfdx, d2fdx2):
        # dfdx shape = (d, S, Output)
        # d2fdx2 shape = (d, d, S, Output)
        d = dfdx.shape[0]
        s = dfdx.shape[1]
        dfdx_r = dfdx.reshape(d, s, -1).T
        d2fdx2_r = d2fdx2.reshape(d, d, s, -1).T

        # dfdxA indecies (l, k, i)
        # dfdxB indecies (l, k, j)
        # d2fdx2 indecies (l, k, i, j)
        # mean_corr = sum over k (dfdxA[k, i] * dfdxB[k, j] * d2fdx2[k, i, j]) with same k
        mean_corr = jnp.einsum('lki,lkj,lkij->lk', dfdx_r, dfdx_r, d2fdx2_r)
        mean_corr /= (d ** 2)
        mean_corr = mean_corr.mean(axis=0)
        mean_corr = jnp.sqrt(jnp.mean(mean_corr**2)).item()
        return mean_corr

    def get_derivatives_correlation_3order(self, dfdx, d3fdx3):
        # dfdx shape = (d, S, Output)
        # d3fdx3 shape = (d, d, d, S, Output)
        d = dfdx.shape[0]
        s = dfdx.shape[1]
        dfdx_r = dfdx.reshape(d, s, -1).T
        d3fdx3_r = d3fdx3.reshape(d, d, d, s, -1).T

        # dfdxA indecies (l, k, i)
        # dfdxB indecies (l, k, j)
        # dfdxC indecies (l, k, l)
        # d3fdx3 indecies (l, k, i, j, r)
        # mean_corr = sum over k
        # (dfdxA[k, i] * dfdxB[k, l] * dfdxC[k, l] * d3fdx3[k, i, j, l]) with same k
        mean_corr = jnp.einsum('lki,lkj,lkr,lkijr->lk', dfdx_r, dfdx_r, dfdx_r, d3fdx3_r)
        mean_corr /= (d ** 3)
        mean_corr = mean_corr.mean(axis=0)
        mean_corr = jnp.sqrt(jnp.mean(mean_corr**2)).item()
        return mean_corr
