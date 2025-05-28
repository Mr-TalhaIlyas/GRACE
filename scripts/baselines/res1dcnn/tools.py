from termcolor import cprint
import torch, os
from config import config

def save_chkpt(model, optimizer, epoch=0, loss=0, acc=0, return_chkpt=False):
    cprint('-> Saving checkpoint', 'green')
    torch.save({
                'epoch': epoch,
                'loss': loss,
                'acc': acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(config["checkpoint_path"], f'{config["experiment_name"]}.pth'))#_epoch{epoch}
    cprint(os.path.join(config["checkpoint_path"], f'{config["experiment_name"]}.pth'), 'cyan')#_epoch{epoch}
    if return_chkpt:
        return os.path.join(config["checkpoint_path"], f'{config["experiment_name"]}.pth')#_epoch{epoch}
    

def values_fromreport(report):
    p = report['weighted avg']['precision']
    r = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    return p,r, f1