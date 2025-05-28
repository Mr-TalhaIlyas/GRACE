from termcolor import cprint
import torch, os
from configs.config import config

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

def load_chkpt(model, optimizer, chkpt_path):
    if os.path.isfile(chkpt_path):
        print(f'-> Loading checkpoint from {chkpt_path}')
        checkpoint = torch.load(chkpt_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        acc = checkpoint['acc']

        print(f'-> Loaded checkpoint for epoch {epoch} with loss {loss} and accuracy {acc}')
        return epoch, loss, acc
    else:
        print(f'Error: No checkpoint found at {chkpt_path}')
        return None, None, None
    
def load_pretrained_chkpt(model, pretrained_path=None):
        if pretrained_path is not None:
            chkpt = torch.load(pretrained_path,
                               map_location='cuda' if torch.cuda.is_available() else 'cpu')
            try:
                # load pretrained
                del chkpt['state_dict']['backbone.A'] # delete the saved adjacency matrix 
                pretrained_dict = chkpt['state_dict']
                # load model state dict
                state = model.state_dict()
                # loop over both dicts and make a new dict where name and the shape of new state match
                # with the pretrained state dict.
                matched, unmatched = [], []
                new_dict = {}
                for i, j in zip(pretrained_dict.items(), state.items()):
                    pk, pv = i # pretrained state dictionary
                    nk, nv = j # new state dictionary
                    # if name and weight shape are same
                    if pk.strip('backbone.') == nk: #.strip('backbone.')
                        new_dict[nk] = pv
                        matched.append(pk)
                    elif pv.shape == nv.shape:
                        new_dict[nk] = pv
                        matched.append(pk)
                    else:
                        unmatched.append(pk)

                state.update(new_dict)
                model.load_state_dict(state)
                print('Pre-trained state loaded successfully...')
                print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
                # print(unmatched)
            except:
                print(f'ERROR in pretrained_dict @ {pretrained_path}')
        else:
            print('Enter pretrained_dict path.')