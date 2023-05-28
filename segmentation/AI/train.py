import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[0] + 'LiverStagePipeline')

from segmentation.AI.logger import Logger
from segmentation.AI import visualize_log

import torch
from collections import OrderedDict

from tqdm import tqdm

def train(model, train_loader, evaluator, num_epochs, optimizer, scheduler, print_every, device, log_file=None, figure_file=None, model_path=None, 
          early_stop=300, eval_trainloader=False, metric_for_best='aP', printed_vals=None):
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
    model.to(device)
    i = 0

    # Initialize logger
    logger = Logger(log_file, metric=metric_for_best)

    # Loop over training data
    for epoch in range(num_epochs):

        model.train()
        epoch_training_loss = 0
        for images, targets, _, _ in tqdm(train_loader, leave=False):
            # b, r = images[0][0,:,:], images[0][1,:,:]
            # print(type(b), type(r), b.dtype, r.dtype, b.shape, r.shape, b.min(), b.max(), b.median(), r.min(), r.max(), r.median())

            # Send the data to the device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(images, targets)
            loss = sum(loss for loss in output.values())
            epoch_training_loss += loss.item()

            # print([(k,round(output[k].item(), 3)) for k in output.keys()])

            # del images
            # del targets
            torch.cuda.empty_cache()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update the weights
            optimizer.step()

        scheduler.step()

        # Write info to log
        epoch_dict = OrderedDict([('epoch', str(epoch + 1)), ('train_loss', round(epoch_training_loss, 3))])

        # Compute metrics on the train and test data
        if eval_trainloader:
            x = evaluator.eval_train(model)
            epoch_dict.update(x)
        y = evaluator.eval_test(model)
        epoch_dict.update(y)
        
        # Write epoch results to log
        best_epoch = logger.log(epoch_dict)

        # Save model if performance is best thus far
        if (best_epoch == epoch+1) and model_path:
            best_string = '[x]'
            torch.save(model.state_dict(), f=model_path)
        else:
            best_string = '[ ]'
        
        # Print statistics
        if (epoch+1) % print_every == 0:
            print('{} - epoch [{}/{}], train_loss: {}, '.format(best_string, epoch+1, num_epochs, round(epoch_training_loss, 3)) + 
                  ', '.join([': '.join([k, str(epoch_dict[k])]) for k in [k for k in epoch_dict.keys() if k in printed_vals]]))

        # Check for early stopping criteria
        if epoch+1 - best_epoch >= early_stop > 0:
            print('Stopping early as at epoch {} there was no {} improvement for {} epochs.'.format(epoch+1, metric_for_best, early_stop))
            return
        
        # Visualize log file
        if figure_file:
            visualize_log.plot_metrics(log_file=log_file, save_path=figure_file)
