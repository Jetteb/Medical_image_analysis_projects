"""
Project code for image registration topics.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output



def intensity_based_registration(I, Im, adaptive_learning = False, ncc = True, rigid = False):


    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    # shearx, sheary, scalingx, scalingy, rotate, translationx, translationy
    if not rigid:
        x = np.array([0., 1., 1., 0., 0., 0., 0.])
    else:
        x = np.array([0., 0., 0.])

    
    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    
    if not rigid:
        if ncc:
            fun = lambda x: reg.affine_corr(I, Im, x, return_transform=False)
        else:
            fun = lambda x: reg.affine_mi(I, Im, x, return_transform=False)

    else:
        fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)

    learning_rate = 0.001
    decay_rate = 0.9  # Decay rate (you can adjust this)
    decay_steps = 100

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(12,7))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    if rigid and ncc:
        ax2.set_title('correlation, rigid transform')
    elif not ncc and not rigid:
        ax2.set_title('mutual information, affine transform')
    elif ncc and not rigid:
        ax2.set_title('correlation, affine transform')
    else:
        ax2.set_title('mutual information, rigid transform')

    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        if adaptive_learning:
            if k % decay_steps == 0 and k > 0:
                learning_rate = learning_rate * (decay_rate ** (k // decay_steps))
                print(f' interation {k}: Learning rate updated to {learning_rate}')
            else:
                learning_rate = learning_rate

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*learning_rate

        # for visualization of the result
        if not rigid:
            if ncc:
                S, Im_t, _ = reg.affine_corr(I, Im, x, return_transform=True)
            else:
                S, Im_t, _ = reg.affine_mi(I, Im, x, return_transform=True)
        else:
            S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True)


        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)
    
    return im2, S
