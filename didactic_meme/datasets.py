import hashlib
import sklearn
import sklearn.model_selection


def get_file_hash(file_path, BLOCKSIZE=65536):
    sha = hashlib.sha256()
    with open(file_path, 'rb') as kali_file:
        file_buffer = kali_file.read(BLOCKSIZE)
        while len(file_buffer) > 0:
            sha.update(file_buffer)
            file_buffer = kali_file.read(BLOCKSIZE)

    return sha.hexdigest()


def get_dataframe_hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def train_test_split(*data, test_size=0.25, random_state=None):
    n_inputs = len(data)
    split = sklearn.model_selection.train_test_split(*data, test_size=test_size, random_state=random_state)
    return [
        [split[split_index + 2*i] for i in range(n_inputs)]
        for split_index in [0, 1]
    ]

def train_test_split(*data, test_size=0.25, random_state=None):
    n_inputs = len(data)
    split = sklearn.model_selection.train_test_split(*data, test_size=test_size, random_state=random_state)
    return [
        [split[split_index + 2*i] for i in range(n_inputs)]
        for split_index in [0, 1]
    ]

def train_validate_test_split(*data, validate_size=0.25, test_size=0.25, random_state=None):
    test_split = train_test_split(*data, test_size=test_size, random_state=random_state)
    validate_split = train_test_split(*test_split[0], test_size=validate_size/(1 - test_size), random_state=random_state)
    return validate_split[0], validate_split[1], test_split[1]


class FeatureLabelDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(list(self.features.values())[0])

    def _get_item(self, d, index):
        return {
            key: value[index]
            for key, value in
            d.items()
        }

    def __getitem__(self, index):
        return (
            self._get_item(self.features, index),
            self._get_item(self.labels, index),
        )

    # Hack to allow ignite.metrics.Loss to get the shape
    class DictWithShape(dict):
        @property
        def shape(self):
            return list(self.values())[0].shape

    @staticmethod
    def prepare_batch(batch, device, non_blocking):
        return [
            FeatureLabelDataset.DictWithShape(**{
                key: ignite.engine.convert_tensor(value, device=device, non_blocking=non_blocking)
                for key, value in b.items()
            })
            for b in batch
        ]

    @staticmethod
    def batch_size(labels):
        return list(labels.values())[0].shape[0]

    @staticmethod
    def get_batch_size_fn(sample_weight=None):
        if sample_weight is None:
            return FeatureLabelDataset.batch_size
        else:
            return lambda labels: labels[sample_weight].sum().item()
