from matplotlib import pyplot as plt
import csv
import time

def plot_save(loss_list, acc_list, model_save_path):
    '''
    plot temporary loss of training and accuracy of test dataset after each epoch training
    Args:
        loss_list: list of loss value of each iteration
        acc_list: list of PSNR value of each epoch

    Returns: nothing

    '''
    x1 = range(len(acc_list))
    x2 = range(len(loss_list))
    y1 = acc_list
    y2 = loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Validation PSNR vs. epoches')
    plt.ylabel('Validation PSNR')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Training loss vs. iteration')
    plt.ylabel('Training loss')
    # plt.show()
    plt.savefig(model_save_path / (str(model_save_path.stem) +".jpg"))
    create_csv(model_save_path / 'acc_list.csv', acc_list)
    create_csv(model_save_path / 'loss_list.csv', loss_list)


def create_csv(path, result_list):
    '''
    save the records of training
    Args:
        path: csv
        result_list: save path of csv file

    Returns: nothing

    '''
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["origin psnr", "model psnr"])
        csv_write.writerows([i] for i in result_list)