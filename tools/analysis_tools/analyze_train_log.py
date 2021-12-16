import json
import matplotlib.pyplot as plt
import pdb


def ParseTrainLog(logs: list):
    for log in logs:
        epochs = []
        iters = []
        loss = []
        loss_dirs = []
        loss_clss = []
        loss_locs = []
        max_iter = 1
        with open(log) as f:
            for line in f:
                map = json.loads(line)
                # print(map)
                if "mode" not in map.keys():
                    continue
                if map["mode"] != "train":
                    continue
                epochs.append(map["epoch"])
                if max_iter < map["iter"]:
                    max_iter = map["iter"]
                if map["epoch"] > 1:
                    iters.append((map["epoch"] - 1) * max_iter + map["iter"])
                else:
                    iters.append(map["iter"])
                loss_clss.append(map["loss_cls"])
                loss_locs.append(map["loss_bbox"])
                loss_dirs.append(map["loss_dir"])
                loss.append(map["loss"])

        plt.figure()
        plt.plot(iters, loss)
        plt.plot(iters, loss_clss)
        plt.plot(iters, loss_locs)
        plt.plot(iters, loss_dirs)
        plt.legend(["loss", "loss_cls", "loss_locs", "loss_dirs"])
        plt.grid()
        plt.title(log)
        f.close()
    plt.show()


if __name__ == "__main__":
    logs = []
    file_1 = "checkpoints/train_15_nuscenes/20220513_071526.log.json"
    logs.append(file_1)
    # file_2 = "checkpoints/train_6_nuscenes/20220118_172016.log.json"
    # logs.append(file_2)
    ParseTrainLog(logs)
