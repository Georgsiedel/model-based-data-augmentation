import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shutil
import argparse
import ast
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Error: Boolean value expected for argument {v}.')


class str2dictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Parse the dictionary string into a dictionary object
        # if values == '':

        try:
            dictionary = ast.literal_eval(values)
            if not isinstance(dictionary, dict):
                raise ValueError("Invalid dictionary format")
        except (ValueError, SyntaxError) as e:
            raise argparse.ArgumentTypeError(f"Invalid dictionary format: {values}") from e

        setattr(namespace, self.dest, dictionary)



import matplotlib.pyplot as plt
import torch

def plot_images(number, mean, std, images, corrupted_images = None, second_corrupted_images = None):
    images = images * std + mean
    corrupted_images = corrupted_images * std + mean
    
    # Define a consistent figure size for each row of images
    row_height = 1.0  # Height per row
    col_width = 1.0   # Width per column
    columns = 1
    if corrupted_images is not None:
        columns = 2
        if second_corrupted_images is not None:
            columns = 3
    fig, axs = plt.subplots(number, columns, figsize=(2 * col_width, number * row_height), squeeze=False)
    
    images = images.cpu()
    corrupted_images = corrupted_images.cpu() if corrupted_images is not None else corrupted_images
    second_corrupted_images = second_corrupted_images.cpu() if second_corrupted_images is not None else second_corrupted_images
    
    for i in range(number):
        image = images[i]
        image = torch.squeeze(image)
        image = image.permute(1, 2, 0)
        axs[i, 0].imshow(image)
        axs[i, 0].axis('off')  # Turn off axes for cleaner visualization
        
        if corrupted_images is not None:
            corrupted_image = corrupted_images[i]
            corrupted_image = torch.squeeze(corrupted_image)
            corrupted_image = corrupted_image.permute(1, 2, 0)
            axs[i, 1].imshow(corrupted_image)
            axs[i, 1].axis('off')  # Turn off axes for cleaner visualization
        
        if second_corrupted_images is not None:
            second_corrupted_image = second_corrupted_images[i]
            second_corrupted_image = torch.squeeze(second_corrupted_image)
            second_corrupted_image = second_corrupted_image.permute(1, 2, 0)
            axs[i, 2].imshow(second_corrupted_image)
            axs[i, 2].axis('off')  # Turn off axes for cleaner visualization

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()

def calculate_steps(dataset, batchsize, epochs, start_epoch, warmupepochs, validontest, validonc, swa, swa_start_factor):
    #+0.5 is a way of rounding up to account for the last partial batch in every epoch
    if dataset == 'ImageNet':
        if validontest == True:
            trainsteps_per_epoch = round(1281167 / batchsize + 0.5)
            validsteps_per_epoch = round(50000 / batchsize + 0.5)
        else:
            trainsteps_per_epoch = round(0.8 * 1281167 / batchsize + 0.5)
            validsteps_per_epoch = round(0.2 * 1281167 / batchsize + 0.5)
    elif dataset == 'TinyImageNet':
        if validontest == True:
            trainsteps_per_epoch = round(100000 / batchsize + 0.5)
            validsteps_per_epoch = round(10000 / batchsize + 0.5)
        else:
            trainsteps_per_epoch = round(0.8 * 100000 / batchsize + 0.5)
            validsteps_per_epoch = round(0.2 * 100000 / batchsize + 0.5)
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
        if validontest == True:
            trainsteps_per_epoch = round(50000 / batchsize + 0.5)
            validsteps_per_epoch = round(10000 / batchsize + 0.5)
        else:
            trainsteps_per_epoch = round(0.8 * 50000 / batchsize + 0.5)
            validsteps_per_epoch = round(0.2 * 50000 / batchsize + 0.5)

    if validonc == True:
        validsteps_per_epoch += 1

    if swa == True:
        total_validsteps = validsteps_per_epoch * int((2-swa_start_factor) * epochs) + warmupepochs
    else:
        total_validsteps = validsteps_per_epoch * (epochs + warmupepochs)
    total_trainsteps = trainsteps_per_epoch * (epochs + warmupepochs)

    if swa == True:
        started_swa_epochs = start_epoch - warmupepochs - int(swa_start_factor * epochs) if start_epoch - warmupepochs - int(swa_start_factor * epochs) > 0 else 0
        start_validsteps = validsteps_per_epoch * (start_epoch + started_swa_epochs)
    else:
        start_validsteps = validsteps_per_epoch * (start_epoch)
    start_trainsteps = trainsteps_per_epoch * start_epoch

    total_steps = int(total_trainsteps+total_validsteps)
    start_steps = int(start_trainsteps+start_validsteps)
    return total_steps, start_steps

class Checkpoint:
    """Early stops the training if validation loss doesn't improve after a given patience.
    credit to https://github.com/Bjarten/early-stopping-pytorch/tree/master for early stopping functionality"""

    def __init__(self, combine_train_corruptions, dataset, modeltype, experiment, train_corruption, run,
                 earlystopping=False, patience=7, verbose=False, delta=0, trace_func=print,
                 checkpoint_path=f'../trained_models/checkpoint.pt',
                 ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = False
        self.val_loss_min = 1000  # placeholder initial value
        self.delta = delta
        self.trace_func = trace_func
        self.early_stopping = earlystopping
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        if combine_train_corruptions:
            self.final_model_path = os.path.abspath(f'../trained_models/{dataset}/{modeltype}/config{experiment}_run_{run}.pth')
        else:
            self.final_model_path = os.path.abspath(f'../trained_models/{dataset}/{modeltype}/config{experiment}_' \
                    f'{train_corruption["noise_type"]}_eps_{train_corruption["epsilon"]}_{train_corruption["sphere"]}_run_{run}.pth')


    def earlystopping(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.best_model = True
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.best_model = False
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and self.early_stopping == True:
                self.early_stop = True
                print("Early stopping")
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = True

    def load_model(self, model, swa_model, optimizer, scheduler, swa_scheduler, type='standard'):
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        if type == 'standard':
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            start_epoch = checkpoint['epoch'] + 1
        elif type == 'best':
            model.load_state_dict(checkpoint['best_model_state_dict'], strict=True)
            start_epoch = checkpoint['best_epoch'] + 1
        else:
            print('only best_checkpoint or checkpoint can be loaded')

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if swa_model != None:
            swa_model.load_state_dict(checkpoint['swa_model_state_dict'], strict=True)
            swa_scheduler.load_state_dict(checkpoint['swa_scheduler_state_dict'])

        return start_epoch, model, swa_model, optimizer, scheduler, swa_scheduler

    def save_checkpoint(self, model, swa_model, optimizer, scheduler, swa_scheduler, epoch):

        swa_model = None if swa_model == None else swa_model.state_dict()
        swa_scheduler = None if swa_scheduler == None else swa_scheduler.state_dict()

        if self.best_model == True:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'swa_model_state_dict': swa_model,
                'swa_scheduler_state_dict': swa_scheduler,
                'best_epoch': epoch,
                'best_model_state_dict': model.state_dict(),
            }, self.checkpoint_path)

        else:
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            checkpoint['epoch'] = epoch
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint['swa_model_state_dict'] = swa_model
            checkpoint['swa_scheduler_state_dict'] = swa_scheduler

            torch.save(checkpoint, self.checkpoint_path)

    def save_final_model(self, model, optimizer, scheduler, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, self.final_model_path)

class TrainTracking:
    def __init__(self, dataset, modeltype, lrschedule, experiment, run, validonc, validonadv, swa):
        self.dataset = dataset
        self.modeltype = modeltype
        self.lrschedule = lrschedule
        self.experiment = experiment
        self.run = run
        self.validonc = validonc
        self.validonadv = validonadv
        self.swa = swa
        self.train_accs, self.train_losses, self.valid_accs, self.valid_losses, self.valid_accs_robust = [],[],[],[],[]
        self.valid_accs_adv, self.valid_accs_swa, self.valid_accs_robust_swa, self.valid_accs_adv_swa = [],[],[],[]

    def load_learning_curves(self):

        learning_curve_frame = pd.read_csv(os.path.abspath(f'./results/{self.dataset}/{self.modeltype}/config{self.experiment}_'
                                           f'learning_curve_run_{self.run}.csv'), sep=';', decimal=',')
        train_accs = learning_curve_frame.iloc[:, 0].values.tolist()
        train_losses = learning_curve_frame.iloc[:, 1].values.tolist()
        valid_accs = learning_curve_frame.iloc[:, 2].values.tolist()
        valid_losses = learning_curve_frame.iloc[:, 3].values.tolist()
        columns=4

        valid_accs_robust, valid_accs_adv, valid_accs_swa, valid_accs_robust_swa, valid_accs_adv_swa = [],[],[],[],[]
        if self.validonc == True:
            valid_accs_robust = learning_curve_frame.iloc[:, columns].values.tolist()
            columns = columns + 1
        if self.validonadv == True:
            valid_accs_adv = learning_curve_frame.iloc[:, columns].values.tolist()
            columns = columns + 1
        if self.swa['apply'] == True:
            valid_accs_swa = learning_curve_frame.iloc[:, columns].values.tolist()
            if self.validonc == True:
                valid_accs_robust_swa = learning_curve_frame.iloc[:, columns+1].values.tolist()
                columns = columns + 1
            if self.validonadv == True:
                valid_accs_adv_swa = learning_curve_frame.iloc[:, columns+1].values.tolist()

        self.train_accs = train_accs
        self.train_losses = train_losses
        self.valid_accs = valid_accs
        self.valid_losses = valid_losses
        self.valid_accs_robust = valid_accs_robust
        self.valid_accs_adv = valid_accs_adv
        self.valid_accs_swa = valid_accs_swa
        self.valid_accs_robust_swa = valid_accs_robust_swa
        self.valid_accs_adv_swa = valid_accs_adv_swa

    def save_metrics(self, train_acc, valid_acc, valid_acc_robust, valid_acc_adv, valid_acc_swa,
                             valid_acc_robust_swa, valid_acc_adv_swa, train_loss, valid_loss):

        self.train_accs.append(train_acc)
        self.train_losses.append(train_loss)
        self.valid_accs.append(valid_acc)
        self.valid_losses.append(valid_loss)
        self.valid_accs_robust.append(valid_acc_robust)
        self.valid_accs_adv.append(valid_acc_adv)
        self.valid_accs_swa.append(valid_acc_swa)
        self.valid_accs_robust_swa.append(valid_acc_robust_swa)
        self.valid_accs_adv_swa.append(valid_acc_adv_swa)

    def save_learning_curves(self):

        learning_curve_frame = pd.DataFrame({"train_accuracy": self.train_accs, "train_loss": self.train_losses,
                                                 "valid_accuracy": self.valid_accs, "valid_loss": self.valid_losses})
        columns = 4
        if self.validonc == True:
            learning_curve_frame.insert(columns, "valid_accuracy_robust", self.valid_accs_robust)
            columns = columns + 1
        if self.validonadv == True:
            learning_curve_frame.insert(columns, "valid_accuracy_adversarial", self.valid_accs_adv)
            columns = columns + 1
        if self.swa['apply'] == True:
            learning_curve_frame.insert(columns, "valid_accuracy_swa", self.valid_accs_swa)
            if self.validonc == True:
                learning_curve_frame.insert(columns+1, "valid_accuracy_robust_swa", self.valid_accs_robust_swa)
                columns = columns + 1
            if self.validonadv == True:
                learning_curve_frame.insert(columns+1, "valid_accuracy_adversarial_swa", self.valid_accs_adv_swa)
        learning_curve_frame.to_csv(os.path.abspath(f'./results/{self.dataset}/{self.modeltype}/config{self.experiment}_'
                                    f'learning_curve_run_{self.run}.csv'),
                                    index=False, header=True, sep=';', float_format='%1.4f', decimal=',')

        x = list(range(1, len(self.train_accs) + 1))
        plt.figure()
        plt.plot(x, self.train_accs, label='Train Accuracy')
        plt.plot(x, self.valid_accs, label='Validation Accuracy')
        if self.validonc == True:
            plt.plot(x, self.valid_accs_robust, label='Robust Validation Accuracy')
        if self.validonadv == True:
            plt.plot(x, self.valid_accs_adv, label='Adversarial Validation Accuracy')
        if self.swa['apply'] == True:
            swa_diff = [self.valid_accs_swa[i] if self.valid_accs[i] != self.valid_accs_swa[i] else None for i in
                        range(len(self.valid_accs))]
            plt.plot(x, swa_diff, label='SWA Validation Accuracy')
            if self.validonc == True:
                swa_robust_diff = [self.valid_accs_robust_swa[i] if self.valid_accs_robust[i] != self.valid_accs_robust_swa[i]
                            else None for i in range(len(self.valid_accs_robust))]
                plt.plot(x, swa_robust_diff, label='SWA Robust Validation Accuracy')
            if self.validonadv == True:
                swa_adv_diff = [self.valid_accs_adv_swa[i] if self.valid_accs_adv[i] != self.valid_accs_adv_swa[i]
                            else None for i in range(len(self.valid_accs_adv))]
                plt.plot(x, swa_adv_diff, label='SWA Adversarial Validation Accuracy')
        plt.title('Learning Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.xticks(np.linspace(1, len(self.train_accs), num=10, dtype=int))
        plt.legend(loc='best')
        plt.savefig(os.path.abspath(f'results/{self.dataset}/{self.modeltype}/config{self.experiment}_learning_curve_run_{self.run}.svg'))
        plt.close()

    def save_config(self):
        shutil.copyfile(os.path.abspath(f'./experiments/configs/config{self.experiment}.py'),
                        os.path.abspath(f'./results/{self.dataset}/{self.modeltype}/config{self.experiment}.py'))

    def print_results(self):
        print("Maximum (non-SWA) validation accuracy of", max(self.valid_accs), "achieved after",
              np.argmax(self.valid_accs) + 1, "epochs; ")
        if self.validonc:
            print("Maximum (non-SWA) robust validation accuracy of", max(self.valid_accs_robust), "achieved after",
                  np.argmax(self.valid_accs_robust) + 1, "epochs; ")
        if self.validonadv:
            print("Maximum (non-SWA) adversarial validation accuracy of", max(self.valid_accs_adv), "achieved after",
                  np.argmax(self.valid_accs_adv) + 1, "epochs; ")

class TestTracking:
    def __init__(self, dataset, modeltype, experiment, runs, combine_train_corruptions, combine_test_corruptions,
                      test_on_c, calculate_adv_distance, calculate_autoattack_robustness, train_corruptions,
                 test_corruptions, adv_distance_params):
        self.dataset = dataset
        self.modeltype = modeltype
        self.experiment = experiment
        self.runs = runs
        self.combine_train_corruptions = combine_train_corruptions
        self.combine_test_corruptions = combine_test_corruptions
        self.test_on_c = test_on_c
        self.calculate_adv_distance = calculate_adv_distance
        self.calculate_autoattack_robustness = calculate_autoattack_robustness
        self.train_corruptions = train_corruptions
        self.test_corruptions = test_corruptions
        self.adv_distance_params = adv_distance_params
        self.results_folder = f'./results/{self.dataset}/{self.modeltype}/config{self.experiment}'

        if combine_train_corruptions:
            self.model_count = 1
        else:
            self.model_count = train_corruptions.shape[0]

        self.test_count = 2
        if test_on_c:
            self.test_count += 34
        if combine_test_corruptions:
            self.test_count += 1
        else:
            self.test_count += test_corruptions.shape[0]
        if calculate_adv_distance:
            self.adv_count = len(self.adv_distance_params["norm"]) * (2+len(self.adv_distance_params["clever_samples"])) + 1
            self.test_count += self.adv_count
        if calculate_autoattack_robustness:
            self.test_count += 2

        self.all_test_metrics = np.empty([self.test_count, self.model_count, self.runs])

    def create_report(self):

        self.avg_test_metrics = np.empty([self.test_count, self.model_count])
        self.std_test_metrics = np.empty([self.test_count, self.model_count])
        self.max_test_metrics = np.empty([self.test_count, self.model_count])

        for idm in range(self.model_count):
            for ide in range(self.test_count):
                self.avg_test_metrics[ide, idm] = self.all_test_metrics[ide, idm, :].mean()
                self.std_test_metrics[ide, idm] = self.all_test_metrics[ide, idm, :].std()
                self.max_test_metrics[ide, idm] = self.all_test_metrics[ide, idm, :].max()

        if self.combine_train_corruptions == True:
            train_corruptions_string = ['config_model']
        else:
            train_corruptions_string = np.array([','.join(map(str, row.values())) for row in self.train_corruptions])

        test_corruptions_string = np.array(['Standard_Acc', 'RMSCE'])
        if self.test_on_c == True:
            test_corruptions_label = np.loadtxt(os.path.abspath('../data/c-labels.txt'), dtype=list)
            if self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
                test_corruptions_bar_label = np.loadtxt(os.path.abspath('../data/c-bar-labels-cifar.txt'), dtype=list)
            elif self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
                test_corruptions_bar_label = np.loadtxt(os.path.abspath('../data/c-bar-labels-IN.txt'), dtype=list)
            test_corruptions_string = np.append(test_corruptions_string, test_corruptions_label, axis=0)
            test_corruptions_string = np.append(test_corruptions_string, test_corruptions_bar_label, axis=0)
            test_corruptions_string = np.append(test_corruptions_string,
                                                ['Acc_C-all-19', 'Acc_C-original-15', 'Acc_C-bar-10', 'Acc_all-ex-pixelwise-noise-24', 'RMSCE_C'],
                                                axis=0)

        if self.calculate_adv_distance == True:
            test_corruptions_string = np.append(test_corruptions_string, ['Acc_from_adv_dist_calculation'])
            for _, n in enumerate(self.adv_distance_params["norm"]):
                test_corruptions_string = np.append(test_corruptions_string,
                                                    [f'{n}-norm-Mean_adv_dist_with_misclassifications_0',
                                                    f'{n}-norm-Mean_adv_dist_without_misclassifications'], axis=0)
                for _, b in enumerate(self.adv_distance_params["clever_samples"]):
                    test_corruptions_string = np.append(test_corruptions_string,
                                                        [f'{n}-norm-Mean_CLEVER-{b}-samples'], axis=0)
        if self.calculate_autoattack_robustness == True:
            test_corruptions_string = np.append(test_corruptions_string,
                                                ['Adversarial_accuracy_autoattack', 'Mean_adv_distance_autoattack)'],
                                                axis=0)
        if self.combine_test_corruptions == True:
            test_corruptions_string = np.append(test_corruptions_string, ['Combined_Noise'])
        else:
            test_corruptions_labels = np.array([','.join(map(str, row.values())) for row in self.test_corruptions])
            test_corruptions_string = np.append(test_corruptions_string, test_corruptions_labels)

        avg_report_frame = pd.DataFrame(self.avg_test_metrics, index=test_corruptions_string,
                                        columns=train_corruptions_string)
        avg_report_frame.to_csv(os.path.abspath(f'{self.results_folder}_metrics_test_avg.csv'), index=True, header=True,
                                sep=';', float_format='%1.4f', decimal=',')
        if self.runs >= 2:
            max_report_frame = pd.DataFrame(self.max_test_metrics, index=test_corruptions_string,
                                            columns=train_corruptions_string)
            std_report_frame = pd.DataFrame(self.std_test_metrics, index=test_corruptions_string,
                                            columns=train_corruptions_string)
            max_report_frame.to_csv(
                os.path.abspath(f'{self.results_folder}_metrics_test_max.csv'), index=True, header=True,
                sep=';', float_format='%1.4f', decimal=',')
            std_report_frame.to_csv(
                os.path.abspath(f'{self.results_folder}_metrics_test_std.csv'), index=True, header=True,
                sep=';', float_format='%1.4f', decimal=',')

    def initialize(self, run, model):
        self.run = run
        self.model = model
        self.accs = []

        if self.combine_train_corruptions:
            print(f"Run {run}, evaluating combined model ")
            self.fileaddition = f'_'
        else:
            train_corruption = self.train_corruptions[model]
            print(f"Run {run}, evaluating model trained on noise of type:", train_corruption)
            self.fileaddition = f'_{train_corruption["noise_type"]}_eps_{train_corruption["epsilon"]}_' \
                           f'{train_corruption["sphere"]}_'
        self.filename = os.path.abspath(f'../trained_models/{self.dataset}/{self.modeltype}/config{self.experiment}' \
                   f'{self.fileaddition}run_{run}.pth')

    def track_results(self, new_results):
        for element in new_results:
            self.accs.append(element)
        self.all_test_metrics[:len(self.accs), self.model, self.run] = np.array(self.accs)

    def save_adv_distance(self, dist_sorted, adv_distance_params):

        if adv_distance_params["clever"] == False:
            adv_distance_params["clever_batches"], adv_distance_params["clever_samples"] = [0.0], [0.0]
        columns = []
        for x in adv_distance_params["norm"]:
            columns.append(f"{x}-norm-min-adv-dist")
            columns.extend([f"{x}-norm-PGD-dist", f"{x}-norm-sec-att-dist"])
            columns.extend([f"{x}-norm-Clever-{y}-samples" for y in adv_distance_params["clever_samples"]])

        adv_distance_frame = pd.DataFrame(index=range(adv_distance_params["setsize"]), columns=columns)
        col_counter = 0

        for id, n in enumerate(adv_distance_params["norm"]):
            adv_distance_frame.iloc[:, col_counter:col_counter+3] = dist_sorted[:, id*3:(id+1)*3]
            col_counter += 3

            for j, (batches, samples) in enumerate(zip(adv_distance_params["clever_batches"], adv_distance_params["clever_samples"])):

                indices1 = np.where((dist_sorted[:,id*3+1] <= dist_sorted[:,id*3+2]) & (dist_sorted[:, id*3+1] != 0))[0]
                indices2 = np.where((dist_sorted[:,id*3+2] < dist_sorted[:,id*3+1]) & (dist_sorted[:, id*3+2] != 0))[0]
                # Find indices where column id*3+1 is 0 and column id*3+2 is not 0
                indices_zero1 = np.where((dist_sorted[:,id*3+1] == 0) & (dist_sorted[:,id*3+2] != 0))[0]
                # Find indices where column id*3+2 is 0 and column id*3+1 is not 0
                indices_zero2 = np.where((dist_sorted[:,id*3+2] == 0) & (dist_sorted[:,id*3+1] != 0))[0]
                # Find indices where both are 0 and asign them to PGD attack
                indices_doublezero = np.where((dist_sorted[:, id * 3 + 2] == 0) & (dist_sorted[:, id * 3 + 1] == 0))[0]
                # Concatenate the indices with appropriate conditions
                indices1 = np.concatenate((indices1, indices_zero2, indices_doublezero))
                indices2 = np.concatenate((indices2, indices_zero1))

                adv_fig = plt.figure(figsize=(15, 5))
                plt.scatter(indices1, dist_sorted[:,id*3+1][indices1], s=5, label="PGD Adversarial Distance")
                plt.scatter(indices2, dist_sorted[:,id*3+2][indices2], s=5, label="Second Attack Adversarial Distance")
                if adv_distance_params["clever"]:
                    plt.scatter(range(len(dist_sorted[:,len(adv_distance_params["norm"]) * 3 + id *
                                                        len(adv_distance_params["clever_batches"]) + j])),
                                dist_sorted[:,len(adv_distance_params["norm"]) * 3 + id * len(adv_distance_params["clever_batches"]) + j],
                                s=5, label=f"Clever Score: {samples} samples")
                plt.title(f"{n}-norm adversarial distance vs. CLEVER score")
                plt.xlabel("Image ID sorted by adversarial distance")
                plt.ylabel("Distance")
                plt.legend()
                plt.close()

                adv_fig.savefig(os.path.abspath(f'results/{self.dataset}/{self.modeltype}/config{self.experiment}{self.fileaddition}run'
                                f'_{self.run}_adversarial_distances_{n}-norm_{samples}-CLEVER-samples.svg'))
                adv_distance_frame.iloc[:, col_counter] = dist_sorted[:,len(adv_distance_params["norm"])*3+id*
                                                                     len(adv_distance_params["clever_batches"]) + j]
                col_counter += 1

        adv_distance_frame.to_csv(os.path.abspath(f'./results/{self.dataset}/{self.modeltype}/config{self.experiment}{self.fileaddition}'
                                  f'run_{self.run}_adversarial_distances.csv'),
                                  index=False, header=True, sep=';', float_format='%1.4f', decimal=',')

