import numpy as np
import matplotlib.pyplot as plt

def parse_logger(log_file):
    with open(log_file, 'r') as f:
        log = f.readlines()
    
    names = [x.split(':')[0].strip() for x in log[0].split(',')]
    values = np.array([[x.split(':')[1].strip() for x in line.split(',')] for line in log]).T.tolist()
    metrics = {}
    for name, value in zip(names, values):
        if all([v.replace('.','',1).isdigit() for v in value]):
            metrics[name] = list(map(float, value))
        else:
            metrics[name] = value
    # metrics = {name: list(map(float, values)) for name,values in zip(names, values)}
    
    # metrics['Epoch'] = list(map(int, metrics['Epochs']))
    # metrics['Train loss'] = list(map(float, metrics['Train loss']))
    # metrics['imgs'] = list(map(int, metrics['Images']))
    # metrics['Cells'] = list(map(int, metrics['Cells']))
    # metrics['TP'] = list(map(int, metrics['True positives']))
    # metrics['FP'] = list(map(int, metrics['False positives']))
    # metrics['FN'] = list(map(int, metrics['False negatives']))
    # metrics['Precision'] = list(map(float, metrics['Precision']))
    # metrics['Recall'] = list(map(float, metrics['Recall']))
    # metrics['aP'] = list(map(float, metrics['aP']))
    
    return metrics

def plot_metric(log_file, metric_name, save_path="/home/anton/Documents/results/figures/example.png"):
    metrics = parse_logger(log_file)

    plt.plot(metrics['Epoch'], metrics[metric_name])
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.savefig(save_path)


def plot_two_metrics(log_file, metric1_name, metric2_name, save_path="/home/anton/Documents/results/figures/example.png"):
    metrics = parse_logger(log_file)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(metric1_name, color=color)
    ax1.plot(metrics['Epoch'], metrics[metric1_name], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(metric2_name, color=color)  # we already handled the x-label with ax1
    ax2.plot(metrics['Epoch'], metrics[metric2_name], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_path)

def plot_metrics(log_file, save_path="/home/anton/Documents/results/figures/example.png"):
    metrics = parse_logger(log_file)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train_loss', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim([0, 200])
    lns = ax1.plot(metrics['epoch'], metrics['train_loss'], color='red', label='Train loss')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim([0, 1])

    ax2.set_ylabel('Metrics', color='black')  # we already handled the x-label with ax1

    lns += ax2.plot(metrics['epoch'], metrics['test_aP'], label='Test aP')
    lns += ax2.plot(metrics['epoch'], metrics['test_precision'], label='Test precision')
    lns += ax2.plot(metrics['epoch'], metrics['test_recall'], label='Test recall')
    
    if 'train_aP' in metrics.keys():
        lns +=  ax2.plot(metrics['epoch'], metrics['train_aP'], label='Train aP')

    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper center')

    # ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_path)
    plt.close()

