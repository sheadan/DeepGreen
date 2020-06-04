import matplotlib.pyplot as plt
import numpy as np


def plot_model_prediction(model, index, u_data, f_data, save_plot=False):
    # Compare model prediction to true value in data
    u_aec, f_aec, f_pred, u_pred = model.predict(x=[u_data, f_data])

    _, n = u_data.shape
    x_pts = np.linspace(0,2*np.pi,n)

    plt.figure() 
    plt.plot(x_pts, f_data[index,:], 'C1', linewidth=2, label="F true")
    plt.plot(x_pts, f_aec[index,:], '--r', markersize=2, label="F -> f -> F")
    plt.plot(x_pts, f_pred[index,:], 'o--k', markersize=5, label="Lv -> F")
    plt.legend(loc="upper right",fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    plt.figure() 
    plt.plot(x_pts, u_data[index,:], 'C1', linewidth=2, label="u true")
    plt.plot(x_pts, u_aec[index,:], '--r', markersize=2, label="u -> v -> u")
    plt.plot(x_pts, u_pred[index,:], 'o--k', markersize=5, label=r"$L^{-1}$f -> u")
    plt.legend(loc="upper right",fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()