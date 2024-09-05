import copy

import numpy as np
import torch
import time
from models import CNN, ResNet18
from data_utils import partition_data_iid, partition_data_noniid,add_gaussian_noise_dataset
from torch.utils.data import DataLoader, Subset
from semantic_attack import load_dataset
import torch.multiprocessing as mp
from collections import OrderedDict
from attack_train import  attack_train, test_backdoor, test,train

def main():
    # Device configuration
    device = torch.device("cpu")
    device_train = torch.device("cuda" if torch.cuda.is_available() else "mps")
    if torch.cuda.is_available():
        mp.set_start_method('spawn')
    print("Using device:", device)
    torch.set_num_threads(1)
    num_processes =1
    train_data, test_data = load_dataset(False)
    attack_data, attack_test_data = load_dataset(True)
    # 定义 CIFAR 数据集索引

    racing_car = [2180, 2771, 3233, 4932, 6241, 6813, 6869, 9476, 11395, 11744, 14209, 14238,
    18716, 19793, 20781, 21529, 31311, 40518, 40633, 42119, 42663, 49392]
    racing_car_train=[4932, 6241, 6813, 6869, 9476, 11395, 11744, 14209, 14238,
    18716, 19793, 20781, 21529, 31311, 40518, 40633, 42119, 42663, 49392]
    racing_car_test=[2180, 2771, 3233]
    green_car=[389, 561, 874, 1605, 3378, 3678, 4528, 9744, 19165, 19500, 21422, 22984, 32941,
    34287, 34385, 36005, 37365, 37533, 38658, 38735, 39824, 40138, 41336, 41861,
    47001, 47026, 48003, 48030, 49163, 49588]
    green_car_train=[1605, 3378, 3678, 4528, 9744, 19165, 19500, 21422, 22984, 32941,
    34287, 34385, 36005, 37365, 37533, 38658, 38735, 39824, 40138, 41336, 41861,
    47001, 47026, 48003, 48030, 49163, 49588]
    green_car_test=[389, 561, 874]
    back_ground_wall=[330, 568, 3934, 12336, 30560, 30696,
    33105, 33615, 33907, 36848, 40713, 41706]
    back_ground_wall_train=[12336, 30560, 30696,
    33105, 33615, 33907, 36848, 40713, 41706]
    back_ground_wall_test=[330, 568, 3934]
    poison_train_data = Subset(attack_data, racing_car_train + green_car_train + back_ground_wall_train)
    backdoor_test={}
    backdoor_test['racing_car'] = Subset(attack_data, racing_car_test)
    backdoor_test['green_car'] = Subset(attack_data, green_car_test)
    backdoor_test['back_ground_wall'] = Subset(attack_data, back_ground_wall_test)



    attack_methods = ["Semantic-backdoors"]

    #Global and Client Model Initialization

    # Parameters for Federated Learning
    C = 0.1 # Fraction of clients
    B = 64  # Batch size
    E = 2  # Number of local epochs
    L = 0.1  # Learning rate
    ifIID = False  # If IID or non-IID
    num_rounds = 100  # Number of rounds
    for attack_method in attack_methods:
        print("Attack method:", attack_method)

        # Initialize models
        models = [CNN(num_classes=10, device=device).to(device) for _ in range(100)]
        global_model = ResNet18(num_classes=10, device=device).to(device)

        global_model, accuracy,accuracy_backdoor = FedAvg(num_rounds, C, B, E,L, ifIID, num_processes, device_train, models,
                                                                                          global_model, train_data, test_data,poison_train_data,backdoor_test,attack_method)

        # Save the results
        np.save(attack_method + 'accuracy', np.array(accuracy))
        for key in backdoor_test.keys():
            np.save(attack_method + 'accuracy_(clip=100)'+key, np.array(accuracy_backdoor[key]))
        torch.save(global_model.state_dict(), f'Rest18_(Single_attack)_{attack_method}.pth')



# Main Federated Learning Loop
def FedAvg(num_rounds, C, B, E, L, ifIID, num_processes, device_train, models,
           global_model, train_data, test_data, poison_train_data, backdoor_test, attack_method):
    accuracy = []
    accuracy_backdoor = {}
    for key in backdoor_test.keys():
        accuracy_backdoor[key] = []
    client_num = int(C * 100)
    for round in range(0, 5):
        print(f"Round {round -5}")
        # global_model_state_dict = global_model.state_dict()
        #
        # # Prepare data
        # if ifIID:
        #     client_dataset_dicts = partition_data_iid(train_data, client_num)
        # else:
        #     client_dataset_dicts = partition_data_noniid(train_data, client_num,0.9)
        #
        # print("Training clients")
        #
        # client_state_dicts = []
        # for client_id in range(client_num):
        #     client_train_dataset = DataLoader(
        #         client_dataset_dicts[client_id], batch_size=64)
        #     current_client_dict = train(copy.deepcopy(global_model_state_dict), client_train_dataset, L, epochs=E)
        #     client_state_dicts.append(current_client_dict)
        #
        # total_state_dict = copy.deepcopy(client_state_dicts[0])
        # for i in range(1, client_num):
        #     for key in total_state_dict.keys():
        #         total_state_dict[key] += client_state_dicts[i][key]
        # ave_state_dict = OrderedDict()
        # for key in total_state_dict.keys():
        #     ave_state_dict[key] = total_state_dict[key] / client_num
        #
        # global_model.load_state_dict(ave_state_dict)
        #
        # test_all(accuracy, accuracy_backdoor, global_model, test_data, backdoor_test, device_train)
        accuracy.append(0)
        accuracy_backdoor['racing_car'].append(0)
        accuracy_backdoor['green_car'].append(0)
        accuracy_backdoor['back_ground_wall'].append(0)

    global_model.load_state_dict(torch.load("pretrain.pth"))
    # Attack
    for round in range(5,6):
        print(f"Round {round -5}")
        print("attack_test")
        global_model_state_dict = global_model.state_dict()
        # Prepare data
        if ifIID:
            client_dataset_dicts = partition_data_iid(train_data, client_num)

        else:
            client_dataset_dicts = partition_data_noniid(train_data, client_num,0.9)



        print("Training backdoor clients")
        #Attack
        client_state_dicts = []
        client_train_dataset = DataLoader(
            client_dataset_dicts[0], batch_size=64)
        current_client_dict = attack_train(copy.deepcopy(global_model_state_dict), client_train_dataset,
                                           poison_train_data, attack_method)
        attack_model = ResNet18(num_classes=10, device=device_train).to(device_train)
        attack_model.load_state_dict(current_client_dict)

        client_state_dicts.append(current_client_dict)
        # test_all(accuracy, accuracy_backdoor, attack_model, test_data, backdoor_test, device_train)

        for client_id in range(1,client_num):
            client_train_dataset = DataLoader(
                client_dataset_dicts[client_id], batch_size=64)
            current_client_dict = train(copy.deepcopy(global_model_state_dict), client_train_dataset, L, epochs=E)
            client_state_dicts.append(current_client_dict)

        total_state_dict = copy.deepcopy(client_state_dicts[0])
        for i in range(1, client_num):
            for key in total_state_dict.keys():
                total_state_dict[key] += client_state_dicts[i][key]
        ave_state_dict = OrderedDict()
        for key in total_state_dict.keys():
            ave_state_dict[key] = (total_state_dict[key].double() / client_num).float()

        global_model.load_state_dict(ave_state_dict)

        test_all(accuracy, accuracy_backdoor, global_model, test_data, backdoor_test, device_train)

    for round in range(6, num_rounds):
        print(f"Round {round - 5}")
        global_model_state_dict = global_model.state_dict()

        # Prepare data
        if ifIID:
            client_dataset_dicts = partition_data_iid(train_data, client_num)
        else:
            client_dataset_dicts = partition_data_noniid(train_data, client_num,0.9)

        print("Training clients")

        client_state_dicts = []
        for client_id in range(client_num):
            client_train_dataset = DataLoader(
                client_dataset_dicts[client_id], batch_size=64)
            current_client_dict = train(copy.deepcopy(global_model_state_dict), client_train_dataset, L, epochs=E)
            client_state_dicts.append(current_client_dict)

        total_state_dict = copy.deepcopy(client_state_dicts[0])
        for i in range(1, client_num):
            for key in total_state_dict.keys():
                total_state_dict[key] += client_state_dicts[i][key]
        ave_state_dict = OrderedDict()
        for key in total_state_dict.keys():
            ave_state_dict[key] = total_state_dict[key] / client_num

        global_model.load_state_dict(ave_state_dict)

        test_all(accuracy, accuracy_backdoor, global_model, test_data, backdoor_test, device_train)



    return global_model, accuracy,accuracy_backdoor


def test_all(accuracy, accuracy_backdoor, global_model,test_data ,backdoor_test, device_train):
    print("Test the global model")
    accuracy.append(test(global_model, DataLoader(test_data, shuffle=True), device_train))

    print("Testing backdoor")
    ac = test_backdoor(global_model, backdoor_test, device_train)
    for key in backdoor_test.keys():
        accuracy_backdoor[key].append(ac[key])

if __name__ == "__main__":
    main()
