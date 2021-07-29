from subprocess import call

def run_model(args_dict, batch_size, lr, ratio):
    train_dir = args_dict['train_dir']
    features_dir = args_dict['features_dir']
    log_dir = args_dict['log_dir']
    save_dir = args_dict['save_dir']

    log_name = f'{batch_size} - {lr} - {ratio}'

    log_path = f'{log_dir}/{log_name}'
    save_path = f'{save_dir}/{log_name}'
    
    current_run = call(f"./mmf_run.sh {train_dir} {features_dir} {log_path} {save_path} {batch_size} {lr} {ratio}", shell=True)
    print(f'Currently Training: {log_name}')


def run_sweep(train_dir, features_dir, log_dir, save_dir):
    hparams = {
        "batch_size": [30, 40, 50],
        "learning_rate": [5e-3, 5e-5, 5e-6],
        "lr_ratio": [0.1, 0.3, 0.5]
    }

    args_dict = {
        "train_dir": train_dir,
        "features_dir": features_dir,
        "log_dir": log_dir,
        "save_dir": save_dir
    }

    for batch_size in hparams['batch_size']:
        for lr in hparams['learning_rate']:
            for ratio in hparams['lr_ratio']:
                run_model(args_dict, batch_size, lr, ratio)



