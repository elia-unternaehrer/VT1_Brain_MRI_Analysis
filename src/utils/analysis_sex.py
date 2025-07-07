import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import torch
import numpy as np

sns.set_theme(style='white', font_scale=1.2)

plt.rcParams.update({
    "text.usetex":       False,
    "figure.figsize":    (5.0, 5.0),
    "figure.dpi":        100,
    "font.size":         10,
    "axes.titlesize":    15,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.autolayout": True,
})

def plot_confusion_matrix(test_result, threshold=None, save_path=None):
    sns.set_theme(style='white', font_scale=1.2)
    
    if threshold is not None:
        save_path = save_path + "confusion_matrix_optim_thres.png"
    else:
        save_path = save_path + "confusion_matrix.png"

    y_true = test_result['targets'].numpy()
    y_pred = test_result['predictions'].numpy()
    y_probs = test_result['probabilities'].numpy()

    # If a threshold is provided, apply it to the predictions
    if threshold is not None:
        y_pred = (y_probs[:, 1] >= threshold).astype(int) # female probabilities

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print the accuracy
    accuracy = np.trace(cm) / np.sum(cm)

    # normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["male", "female"])

    disp.plot(cmap=plt.cm.Blues, values_format='.2f', colorbar=False)
    plt.title("Confusion Matrix", fontsize=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Accuracy: {accuracy:.4f}")

    

def plot_roc_curve(test_result, save_path=None):
    sns.set_theme(style='white', font_scale=1.2)

    save_path = save_path + "roc_curve.png" if save_path else "roc_curve.png"

    y_true = test_result['targets'].numpy()
    y_probs = test_result['probabilities'].numpy()

    # Compute ROC
    y_prob_f = y_probs[:,1]           # female probabilities
    fpr, tpr, thr = roc_curve(y_true, y_prob_f, pos_label=1)
    auc = roc_auc_score(y_true, y_prob_f)

    # 1) Find optimal threshold (max Youden's J)
    j = tpr - fpr
    ix = np.argmax(j)
    opt_fpr, opt_tpr, opt_thr = fpr[ix], tpr[ix], thr[ix]


    sns.set_theme(style='whitegrid', font_scale=1.2)
    plt.figure()

    # ROC curve
    plt.plot(fpr, tpr, color='black',
            label=f'Female ROC')
    # Chance line
    plt.plot([0,1],[0,1], linestyle='--', color='navy', label='Random Guess')

    plt.fill_between(fpr, tpr, alpha=0.2, color='orange')

    plt.text(0.65, 0.4,               # choose coords inside the shaded region
            f"AUC = {auc:.4f}",
            fontsize=12,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            )

    # Optimum point & guides
    plt.scatter(opt_fpr, opt_tpr, color='red', s=20, zorder=10,
                label=f'Optimum Thr={opt_thr:.4f}')
    plt.axvline(opt_fpr, linestyle='--', color='red', alpha=0.7)
    plt.axhline(opt_tpr, linestyle='--', color='red', alpha=0.7)
    plt.text(opt_fpr+0.03, opt_tpr-0.1,
            f"TPR={opt_tpr:.2f}\nFPR={(opt_fpr):.2f}",
            color='red',
            fontsize=12,)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=20)
    plt.legend(loc='lower right')
    plt.axis('square')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "threshold": opt_thr,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }

def plot_probability_histogram(test_result):
    y_probs = test_result['probabilities'].numpy()
    plt.hist(y_probs[:,1], bins=50, color='orange', alpha=0.6, label="female")
    plt.hist(y_probs[:,0], bins=50, color='blue', alpha=0.6, label="male")
    plt.title("Histogram of Predicted Probabilities", fontsize=20)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.xlim(0, 1)
    plt.legend()
    plt.show()