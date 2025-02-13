import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    import importlib

    for experiment in [211,213,218,219,220,221,223]:

        configname = (f'experiments.configs.config{experiment}')
        config = importlib.import_module(configname)

        print('Starting experiment #',experiment, 'on', config.dataset, 'dataset')

        runs = 1
        run_iter = [0] 

        if experiment == 194:
            runs = 5
            run_iter = [4] 

        for run in run_iter:
            if experiment in [211] and run in [0]:
                resume = True
            else:
                resume = False

            print("Training run #",run)
            cmd0 = f"python experiments/train.py --resume={resume} --run={run} --experiment={experiment} --epochs=" \
                    f"{config.epochs} --learningrate={config.learningrate} --dataset={config.dataset} --validontest=" \
                    f"{config.validontest} --lrschedule={config.lrschedule} --lrparams=\"{config.lrparams}\" " \
                    f"--earlystop={config.earlystop} --earlystopPatience={config.earlystopPatience} --optimizer=" \
                    f"{config.optimizer} --optimizerparams=\"{config.optimizerparams}\" --modeltype=" \
                    f"{config.modeltype} --modelparams=\"{config.modelparams}\" --resize={config.resize} " \
                    f"--train_aug_strat_orig={config.train_aug_strat_orig} " \
                    f"--train_aug_strat_gen={config.train_aug_strat_gen} --loss=" \
                    f"{config.loss} --lossparams=\"{config.lossparams}\" --trades_loss={config.trades_loss} " \
                    f"--trades_lossparams=\"{config.trades_lossparams}\" --robust_loss={config.robust_loss} " \
                    f"--robust_lossparams=\"{config.robust_lossparams}\" --mixup=\"{config.mixup}\" --cutmix=" \
                    f"\"{config.cutmix}\" --manifold=\"{config.manifold}\" --concurrent_combinations={config.concurrent_combinations}" \
                    f" --batchsize={config.batchsize} --number_workers={config.number_workers} " \
                    f"--RandomEraseProbability={config.RandomEraseProbability} --warmupepochs={config.warmupepochs}" \
                    f" --normalize={config.normalize} --pixel_factor={config.pixel_factor} --minibatchsize=" \
                    f"{config.minibatchsize} --validonc={config.validonc} --validonadv={config.validonadv} --swa=" \
                    f"\"{config.swa}\" --noise_sparsity={config.noise_sparsity} --noise_patch_scale=" \
                    f"\"{config.noise_patch_scale}\" --generated_ratio={config.generated_ratio} "
            os.system(cmd0)
            
        # Calculate accuracy and robust accuracy, evaluating each trained network on each corruption

        print('Beginning metric evaluation')
        cmdeval = "python experiments/eval.py --resume={} --experiment={} --runs={} --batchsize={} --dataset={} " \
                "--modeltype={} --modelparams=\"{}\" --resize={} --combine_test_corruptions={} --number_workers={} " \
                "--normalize={} --pixel_factor={} --test_on_c={} --calculate_adv_distance={} --adv_distance_params=\"{}\" " \
                "--calculate_autoattack_robustness={} --autoattack_params=\"{}\" " \
                .format(resume, experiment, runs, 1000, config.dataset, config.modeltype, config.modelparams,
                        config.resize, config.combine_test_corruptions, 0, config.normalize,
                        config.pixel_factor, config.test_on_c, config.calculate_adv_distance, config.adv_distance_params,
                        config.calculate_autoattack_robustness, config.autoattack_params)
        os.system(cmdeval)

