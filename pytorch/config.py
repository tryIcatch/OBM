import torch


class Config:

    DATA_ROOT = './data'
    SAMPLE_SIZE = 10000
    TEST_SIZE = 2000

    RANDOM_SEED = 42

    INPUT_DIM = {
        'mnist': 784,
        'fashion': 784,
    }


    MAX_ITER = 1000
    LEARNING_RATE = 0.01
    TOLERANCE = 1e-4
    BATCH_SIZE = 1024

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    AGGREGATION_DEVICE = torch.device("cpu")

    @staticmethod
    def tensor_to_numpy(tensor):

        return tensor.detach().cpu().numpy()

    @staticmethod
    def numpy_to_tensor(array):

        return torch.from_numpy(array).to(Config.DEVICE)