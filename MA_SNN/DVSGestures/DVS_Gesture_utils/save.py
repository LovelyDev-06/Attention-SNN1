import pandas as pd
import os


def save_csv(config):
    if not config.epoch_list:
        config.epoch_list.append(config.best_epoch if config.best_epoch else 1)

    n = len(config.epoch_list)
    while len(config.loss_train_list) < n:
        config.loss_train_list.append(0.0)
    while len(config.loss_test_list) < n:
        config.loss_test_list.append(config.test_loss)
    while len(config.acc_train_list) < n:
        config.acc_train_list.append(0.0)
    while len(config.acc_test_list) < n:
        config.acc_test_list.append(config.best_acc)

    lists = [config.loss_train_list[:n],
             config.loss_test_list[:n],
             config.acc_train_list[:n],
             config.acc_test_list[:n]]
    csv = pd.DataFrame(
        data=lists,
        index=['Train_Loss',
               'Test_Loss',
               'Train_Accuracy',
               'Test_Accuracy'],
        columns=config.epoch_list[:n])
    csv.index.name = 'Epochs'

    if not os.path.exists(config.recordPath):
        os.makedirs(config.recordPath)
    csv.to_csv(config.recordPath + os.sep + config.recordNames)
