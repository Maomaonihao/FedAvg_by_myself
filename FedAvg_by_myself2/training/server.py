import time
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt
from training.client import ClientUpdate


def server_training(model, rounds, batch_size, lr, ds, data_dict, C, K, E, plt_title, plt_color):
    """
    Function implements the Federated Averaging Algorithm from the FedAvg paper.
    Specifically, this function is used for the server side training and weight update

    Params:
      - model:           PyTorch model to train
      - rounds:          Number of communication rounds for the client update
      - batch_size:      Batch size for client update training
      - lr:              Learning rate used for client update training
      - ds:              Dataset used for training
      - data_dict:       Type of data partition used for training (IID or non-IID)
      - C:               Fraction of clients randomly chosen to perform computation on each round
      - K:               Total number of clients
      - E:               Number of training passes each client makes over its local dataset per round
      - tb_writer_name:  Directory name to save the tensorboard logs
    Returns:
      - model:           Trained model on the server
    """

    # global model weights
    global_weights = model.state_dict()

    # training loss
    train_loss = []

    # measure time
    start = time.time()

    for curr_round in range(1, rounds + 1):
        w, local_loss = [], []

        m = max(int(C * K), 1)

        S_t = np.random.choice(range(K), m, replace=False)
        for k in S_t:
            local_update = ClientUpdate(dataset=ds, batchSize=batch_size, learning_rate=lr, epochs=E, idxs=data_dict[k])
            weights, loss = local_update.train(model=copy.deepcopy(model))

            w.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))

        # updating the global weights
        weights_avg = copy.deepcopy(w[0])
        for k in weights_avg.keys():
            for i in range(1, len(w)):
                weights_avg[k] += w[i][k]

            weights_avg[k] = torch.div(weights_avg[k], len(w))

        global_weights = weights_avg

        # move the updated weights to our model state dict
        model.load_state_dict(global_weights)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        print('Round: {}... \tAverage Loss: {}'.format(curr_round, round(loss_avg, 3)))
        train_loss.append(loss_avg)

    end = time.time()
    fig, ax = plt.subplots()
    x_axis = np.arange(1, rounds + 1)
    y_axis = np.array(train_loss)
    ax.plot(x_axis, y_axis, 'tab:' + plt_color)

    ax.set(xlabel='Number of Rounds', ylabel='Train Loss',
           title=plt_title)
    ax.grid()
    fig.savefig(plt_title + '.jpg', format='jpg')
    print("Training Done!")
    print("Total time taken to Train: {}".format(end - start))

    return model