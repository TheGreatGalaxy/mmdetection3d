from subprocess import check_output
import torch
import collections
import copy
import io
from mmcv.fileio import FileClient
# from mmcv.runner import load_checkpoint


def PrintSplit():
    print("\n" + "=" * 20 + "\n")


def PrintDict(dicts, input_key="", level=0):
    parent_layer_dict = dict()
    for key, value in dicts.items():
        is_next_layer = False
        if isinstance(value, dict):
            PrintDict(value, input_key + "-" + str(key), level=level + 1)
            # if key in parent_layer_dict.keys():
            #    PrintSplit()
            #    print("this key " + key + "is in dict")
            #    PrintSplit()
            #    parent_layer_dict[key] = value
            # else:
            #    parent_layer_dict[key] = value
        else:
            if isinstance(value, torch.Tensor):
                value = "This value is a tensor"
            print("level: ", level, " input key: ",
                  input_key, " this key: ", key, " \nvalues: ", value)


def ModifyLayers(dicts, contain_key, delete_key):
    '''
    Modify ordered dict will failed.
    '''
    for key, value in dicts.items():
        if isinstance(key, str) and key.find(contain_key) > -1 and key.find(delete_key) > -1:
            dicts.pop(key)
            orig_key = key
            key.replace(delete_key, "")
            print(orig_key, " change into ", key)
            dicts[key] = value
        elif isinstance(value, dict):
            ModifyLayers(value, contain_key, delete_key)
        else:
            pass
    return dicts


def ModifyLayers(dicts, contain_key, delete_key):
    '''
    Modify ordered dict.
    '''
    res = collections.OrderedDict()
    for key, value in dicts.items():
        if isinstance(key, str) and key.find(contain_key) > -1 and key.find(delete_key) > -1:
            modi_key = copy.deepcopy(key)

            modi_key = key.replace(delete_key, "")
            print(key, " change into ", modi_key)
            res[modi_key] = value
        elif isinstance(value, dict):
            res[key] = ModifyLayers(value, contain_key, delete_key)
        else:
            res[key] = value
    return res


def SaveCheckPoint(check_point, file_name, file_client_args=dict(backend='disk')):
    file_client = FileClient.infer_client(file_client_args, file_name)
    with io.BytesIO() as f:
        torch.save(check_point, f)
        file_client.put(f.getvalue(), file_name)


if __name__ == "__main__":

    check_point_file = "checkpoints/train_11_nuscenes/epoch_24.pth"
    # checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    check_point = torch.load(check_point_file)
    PrintDict(check_point, "Toplevel")
    PrintSplit()
    PrintSplit()
    modified_check_point = ModifyLayers(
        check_point, "pts_voxel_encoder", "point_pillars_core.")
    PrintSplit()
    PrintSplit()
    PrintDict(modified_check_point, "ChangedTopLevel")

    modified_file_name = "checkpoints/train_11_nuscenes/modi_epoch_24.pth"
    SaveCheckPoint(modified_check_point, modified_file_name)
