import numpy as np
import torch
import time
from models import CNN, ResNet18
from data_utils import partition_data_iid, partition_data_noniid
from attack_train import test, train_process, attack_process, test_backdoor
from torch.utils.data import DataLoader, Subset
from semantic_attack import load_dataset
import torch.multiprocessing as mp

import torchvision.datasets as datasets


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



    attack_methods = ["Pixel-backdoors","Semantic-backdoors", "Trojan-backdoors"]

    #Global and Client Model Initialization

    # Parameters for Federated Learning
    C = 0.1 # Fraction of clients
    B = 64  # Batch size
    E = 2  # Number of local epochs
    l = 0.1  # Learning rate
    ifIID = False  # If IID or non-IID
    num_rounds = 100  # Number of rounds
    for attack_method in attack_methods:
        print("Attack method:", attack_method)

        # Initialize models
        models = [ResNet18(num_classes=10, device=device).to(device) for _ in range(100)]
        global_model = ResNet18(num_classes=10, device=device).to(device)

        global_model, accuracy,accuracy_backdoor = FedAvg(num_rounds, C, B, E, l, ifIID, num_processes, device_train, models,
                                                                                          global_model, train_data, test_data,poison_train_data,backdoor_test,attack_method)

        # Save the results
        np.save(attack_method + 'accuracy', np.array(accuracy))
        for key in backdoor_test.keys():
            np.save(attack_method + 'accuracy_'+key, np.array(accuracy_backdoor[key]))
        torch.save(global_model.state_dict(), f'Rest18_(Single_attack)_{attack_method}.pth')



# Main Federated Learning Loop
def FedAvg(num_rounds, C, B, E, l, ifIID, num_processes, device_train, models,
                        global_model, train_data, test_data,poison_train_data,backdoor_test, attack_method):
    accuracy = []
    accuracy_backdoor = {}
    for key in backdoor_test.keys():
        accuracy_backdoor[key] = []


    for round in range(0, 5):
        print(f"Round {round -5}")

        # Select clients
        normal_clients = torch.randperm(len(models))[:int( C * len(models))]
        normal_clients = torch.tensor(list(normal_clients))
        normal_clients_number = len(normal_clients)
        normal_clients_process = normal_clients_number // num_processes  # number of clients per process
        # Prepare data
        if ifIID:
            data = partition_data_iid(train_data, normal_clients_number)

        else:
            data = partition_data_noniid(train_data, normal_clients_number)


        # 初始化 weight_accumulator
        weight_accumulator = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}
        print("Training normal clients")
        processes = []
        queue = mp.Queue()
        events = [mp.Event() for _ in range(num_processes)]

        for process_idx in range(num_processes):
            clients_process = normal_clients[
                              process_idx * normal_clients_process: min((process_idx + 1) * normal_clients_process,
                                                                        normal_clients_number)]
            p = mp.Process(target=train_process, args=(
                process_idx * normal_clients_process, process_idx, events[process_idx], clients_process, models, data,
                B, E, l, global_model,
                queue, device_train))
            p.start()
            processes.append(p)

        weight_accumulator = {name: torch.zeros_like(param, dtype=torch.float32) for name, param in
                              global_model.state_dict().items()}
        for _ in range(num_processes):
            trained_models = queue.get()

            for client, model in trained_models.items():
                for name, param in model.named_parameters():
                    weight_accumulator[name] += (param.data - global_model.state_dict()[name]) / normal_clients_number
            events[_].set()

        for p in processes:
            p.join(timeout=10)

        # 使用 weight_accumulator 更新 global_model
        for name, global_param in global_model.state_dict().items():
            update_per_layer = weight_accumulator[name]
            global_param.data.add_(update_per_layer.to(global_param.dtype))

        test_all(accuracy, accuracy_backdoor, global_model, test_data, backdoor_test, device_train)

    # Attack
    for round in range(5,6):
        print(f"Round {round -5}")
        print("attack_test")

        # Select clients
        backdoor_clients = torch.randperm(len(models))[:int(0.1 * C * len(models))]
        normal_clients = torch.randperm(len(models))[:int(0.9 * C * len(models))]
        normal_clients = torch.tensor(list(normal_clients))
        backdoor_clients = torch.tensor(list(backdoor_clients))
        normal_clients_number = len(normal_clients)
        backdoor_clients_number = len(backdoor_clients)
        total_clients_number = normal_clients_number + backdoor_clients_number
        normal_clients_process = normal_clients_number // num_processes  #number of clients per process
        backdoor_clients_process = backdoor_clients_number // num_processes
        # Prepare data
        if ifIID:
            data = partition_data_iid(train_data, normal_clients_number)

        else:
            data = partition_data_noniid(train_data, normal_clients_number, )

        # 初始化 weight_accumulator
        weight_accumulator = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}

        print("Training backdoor clients")
        #Attack
        queue = mp.Queue()
        events = [mp.Event() for _ in range(num_processes)]
        processes = []
        for process_idx in range(num_processes):
            clients_process = backdoor_clients[
                              process_idx * backdoor_clients_process: min((process_idx + 1) * backdoor_clients_process,
                                                                          backdoor_clients_number)]
            p = mp.Process(target=attack_process, args=(
                process_idx * backdoor_clients_process, events[process_idx], clients_process, models,
                data,poison_train_data,B,  global_model,
                queue, attack_method, device_train))
            p.start()
            processes.append(p)

        for _ in range(num_processes):
            # 从队列中获取完整的模型对象字典
            trained_models = queue.get()
            # 替换本地模型
            for client, model in trained_models.items():
                for name, param in model.named_parameters():
                    # if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                    #     continue
                    weight_accumulator[name] += (param.data - global_model.state_dict()[name]) / total_clients_number
        for event in events:
            event.set()

        # del trained_models
        for p in processes:
            # print("p", p.name)
            p.join(timeout=10)
        del trained_models
        print("Training normal clients")
        processes = []
        queue = (mp.Queue())
        events = [mp.Event() for _ in range(num_processes)]

        for process_idx in range(num_processes):
            clients_process = normal_clients[
                              process_idx * normal_clients_process: min((process_idx + 1) * normal_clients_process,
                                                                        normal_clients_number)]
            p = mp.Process(target=train_process, args=(
                process_idx * normal_clients_process, process_idx, events[process_idx], clients_process, models, data,
                B, E, l, global_model,
                queue, device_train))
            p.start()
            processes.append(p)
        for _ in range(num_processes):
            trained_models = queue.get()
            for client, model in trained_models.items():
                for name, param in model.named_parameters():
                    weight_accumulator[name] += (param.data - global_model.state_dict()[name]) / total_clients_number

        del trained_models

        for event in events:
            event.set()

        for p in processes:
            p.join(timeout=10)

        for name, param in global_model.named_parameters():
            if name in weight_accumulator:
                param.data += weight_accumulator[name]

        test_all(accuracy, accuracy_backdoor, global_model,test_data ,backdoor_test, device_train)


    for round in range(6, num_rounds):
        print(f"Round {round + 1}")

        # Select clients
        normal_clients = torch.randperm(len(models))[:int(1 * C * len(models))]
        normal_clients = torch.tensor(list(normal_clients))
        normal_clients_number = len(normal_clients)
        backdoor_clients_number = 0
        normal_clients_process = normal_clients_number // num_processes  # number of clients per process
        # Prepare data
        if ifIID:
            data = partition_data_iid(train_data, normal_clients_number)

        else:
            data = partition_data_noniid(train_data, normal_clients_number)

        # 初始化 weight_accumulator
        weight_accumulator = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}

        print("Training normal clients")
        processes = []
        queue = (mp.Queue())
        events = [mp.Event() for _ in range(num_processes)]

        for process_idx in range(num_processes):
            clients_process = normal_clients[
                              process_idx * normal_clients_process: min((process_idx + 1) * normal_clients_process,
                                                                        normal_clients_number)]
            p = mp.Process(target=train_process, args=(
                process_idx * normal_clients_process, process_idx, events[process_idx], clients_process, models, data,
                B, E, l, global_model,
                queue, device_train))
            p.start()
            processes.append(p)
        for _ in range(num_processes):
            # 从队列中获取完整的模型对象字典
            trained_models = queue.get()
            # 替换本地模型
            for client, model in trained_models.items():

                # for name, param in model.named_parameters():
                #     if 'fc' in name:
                #         print(f"Parameter name: {name}")
                #         print(param.data)  # 打印参数的具体值
                #         print("------")
                for name, param in model.named_parameters():
                    # if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                    #     continue
                    weight_accumulator[name] += (param.data - global_model.state_dict()[name]) / normal_clients_number

        del trained_models

        for event in events:
            event.set()
        # print("Processes finished")
        for p in processes:
            p.join(timeout=10)
            # if p.is_alive():
            #     print(f"Thread {p.name} did not finish in time")
            # else:
            #     print(f"Thread {p.name} finished in time")

        # 使用 weight_accumulator 更新 global_model
        for name, param in global_model.named_parameters():
            if name in weight_accumulator:
                param.data += weight_accumulator[name]

        test_all(accuracy, accuracy_backdoor, global_model,test_data ,backdoor_test, device_train)



    return global_model, accuracy,accuracy_backdoor


def test_all(accuracy, accuracy_backdoor, global_model,test_data ,backdoor_test, device_train):
    print("Test the global model")
    accuracy.append(test(global_model, DataLoader(test_data, shuffle=True), device_train))

    print("Testing backdoor")
    ac = test_backdoor(global_model, backdoor_test, device_train)
    for key in backdoor_test.keys():
        accuracy_backdoor[key].append(ac[key])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
