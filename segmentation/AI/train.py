import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

from segmentation.AI.logger import Logger
from segmentation.AI import visualize_log
from utils import data_utils

import torch
from collections import OrderedDict

from tqdm import tqdm
import gc
from transformers import MaskFormerImageProcessor

def train(model, train_loader, evaluator, num_epochs, optimizer, get_loss_func, device, log_file=None, figure_file=None, model_path=None, 
          early_stop=300, metric_for_best='aP', scheduler=None):
    """
    Trains a PyTorch model on a training dataset and evaluates on a test dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): A DataLoader object that provides the training dataset.
        test_loader (torch.utils.data.DataLoader): A DataLoader object that provides the test dataset.
        num_epochs (int): The number of epochs to train the model for.
        learning_rate (float): The learning rate to use for the optimizer.
        print_every (int): The number of steps between printouts of the training loss.

    Returns:
        None
    """

    # Initialize logger
    logger = Logger(log_file, metric=metric_for_best)

    # Loop over training data
    for epoch in range(num_epochs):

        model.train()
        epoch_training_loss = 0
        for batch in tqdm(train_loader, leave=False, desc='Train loop'):
            batch = data_utils.move_to_device(batch, device)

            optimizer.zero_grad() # zero the gradients
            loss = get_loss_func(model, batch)
            # print(torch.cuda.memory_allocated()*1e-9)

            loss.backward() # backward pass
            # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping) if gradient_clipping else None
            # print(torch.cuda.memory_allocated()*1e-9)
            # del batch
            # torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated()*1e-9)

            optimizer.step() # update the weights

            epoch_training_loss += loss.detach().item()

        scheduler.step() if scheduler else None

        # Write info to log
        epoch_dict = OrderedDict([('epoch', str(epoch + 1)), ('train_loss', round(epoch_training_loss, 3))])

        test_results = evaluator(model)
        epoch_dict.update(test_results)
        
        # Write epoch results to log
        best_epoch = logger.log(epoch_dict)

        # Save model if performance is best thus far
        if (best_epoch == epoch+1) and model_path:
            best = '[x]'
            torch.save(model.state_dict(), f=model_path)
        else:
            best = '[ ]'
        
        # Print statistics
        print('{} - epoch [{}/{}], train_loss: {}, '.format(best, epoch+1, num_epochs, round(epoch_training_loss, 3)) + 
                ', '.join([': '.join([k, str(epoch_dict[k])]) for k in [k for k in epoch_dict.keys() if not hasattr(epoch_dict[k], 'keys')]]))

        # Check for early stopping criteria
        if epoch+1 - best_epoch >= early_stop > 0:
            print('Stopping early as at epoch {} there was no {} improvement for {} epochs.'.format(epoch+1, metric_for_best, early_stop))
            return
        
        # Visualize log file
        # if figure_file:
        #     visualize_log.plot_metrics(log_file=log_file, save_path=figure_file)
