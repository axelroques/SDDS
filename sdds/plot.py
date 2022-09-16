
import matplotlib.pyplot as plt
import numpy as np


def plot_atoms(D, k=5):
    """
    Plot the k first atoms in the dictionary.
    """

    # _, axes = plt.subplots(k, 1, figsize=(12, k*2))
    _, ax = plt.subplots(1, 1, figsize=(12, 5))

    for i, atom in enumerate(D.T):
        if i < k:
            ax.plot(atom)

    plt.show()

    return


def plot_activations(SDDS):
    """
    Reconstruct the signal using the atoms in the
    dictionary. Note that here, D is a Dictionary
    object and not a list of signals. Likewise,
    Z is an ActivationMatrix object.
    """

    # Set up the axes
    fig = plt.figure(figsize=(12, 2*SDDS.K))
    grid = plt.GridSpec(1+SDDS.K, 3, hspace=0.6, wspace=0.3)
    signal = fig.add_subplot(grid[0, 1:], xticklabels=[])
    axes = []
    for n_atom in range(SDDS.K):
        row_1 = fig.add_subplot(
            grid[n_atom+1, 0], xticklabels=[])
        if n_atom == SDDS.K - 1:
            row_2 = fig.add_subplot(
                grid[n_atom+1, 1:])
        else:
            row_2 = fig.add_subplot(
                grid[n_atom+1, 1:], xticklabels=[])
        axes.append((row_1, row_2))

    # Arbitrary time vector
    t = np.arange(len(SDDS.X))

    # Plot original signal
    signal.plot(t, SDDS.X, c='royalblue')
    signal.set_ylabel('Signal')

    # Plot other rows
    colors = ['crimson', 'lime', 'darkviolet']
    max_y_atom = 0
    max_activation = 0
    for ax, atom, activation in zip(axes,
                                    SDDS.D.yieldAtoms(),
                                    SDDS.Z.yieldActivations()):

        # New time vector
        t = t[:SDDS.N-SDDS.L+1]

        # Plot atom
        ax[0].plot(atom.getSigmoid(), c=colors[atom.id % len(colors)])
        ax[0].set_ylabel(f'Atom {atom.id}')

        # Plot atom activations
        # ax[1].vlines(t, 0, activation, color='crimson')
        ax[1].scatter(t, activation, marker='o',
                      s=5, c=colors[atom.id % len(colors)])

        # For plotting purposes
        max_y = np.abs(atom.getSigmoid()).max()
        if max_y > max_y_atom:
            max_y_atom = max_y
        max_act = np.abs(activation).max()
        if max_act > max_activation:
            max_activation = max_act

    for ax in axes:
        ax[0].set_ylim((-max_y_atom, max_y_atom))
        ax[1].set_ylim((-max_activation, max_activation))

    plt.show()

    return


def reconstruct(X, D, Z):
    """
    Reconstruct the signal using the atoms in the
    dictionary. Note that here, D is a Dictionary
    object and not a list of signals. Likewise,
    Z is an ActivationMatrix object.
    """

    _, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Arbitrary time vector
    t = np.arange(len(X))

    # Plot original signal
    ax.plot(t, X, label='Signal')

    for atom, activation in zip(D.yieldAtoms(), Z.yieldActivations()):

        # Get atom content
        sigmoid = atom.getSigmoid(bypass_norm=False)
        sigmoid_length = len(sigmoid)

        # Iterate over the activation vector
        n_plot = 0
        for t_activation, val_activation in enumerate(activation):

            # If activation value is non-zero, plot the atom
            if abs(val_activation) > 0.3:
                alpha = val_activation/np.abs(activation).max()
                ax.plot(np.arange(t_activation, t_activation+sigmoid_length),
                        sigmoid*val_activation,
                        label=f'Atom {atom.id}' if n_plot == 0 else '',
                        alpha=alpha)
                n_plot += 1

    ax.legend()
    plt.show()

    return
