import hashlib


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
