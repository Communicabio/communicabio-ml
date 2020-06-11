import io
import torch
import os
import json
import pickle

def iterable_to_stream(iterable, buffer_size=io.DEFAULT_BUFFER_SIZE):
    """
    Lets you use an iterable (e.g. a generator) that yields bytestrings as a read-only
    input stream.
    The stream implements Python 3's newer I/O API (available in Python 2's io module).
    For efficiency, the stream is buffered.
    """
    class IterStream(io.RawIOBase):
        def __init__(self):
            self.leftover = None
        def readable(self):
            return True
        def readinto(self, b):
            try:
                l = len(b)  # We're supposed to return at most this much
                chunk = self.leftover or next(iterable)
                output, self.leftover = chunk[:l], chunk[l:]
                b[:len(output)] = output
                return len(output)
            except StopIteration:
                return 0    # indicate EOF
    return io.BufferedReader(IterStream(), buffer_size=buffer_size)

def iterate_chunks(path):
    filenames = os.listdir(path)
    # config = json.load(open(os.path.join(path, 'chunk_config.json')))
    filenames = sorted(filenames, key=lambda x: int(x))
    for filename in filenames:
        filename = os.path.join(path, filename)
        with open(filename, 'rb') as file:
            yield file.read()
        os.remove(filename)

def load_model(path):
    return pickle.load(iterable_to_stream(iterate_chunks(path)))

def save_model(model, path, N=60):
    try:
        os.mkdir(path)
    except FileExistsError as exp:
        pass
    '''pickle.dump(model, os.path.join(path, 'tmp'))
    stream = open(os.path.join(path, 'tmp'), 'rb').read()
    os.remove(os.path.join(path, 'tmp'))'''
    stream = pickle.dumps(model)

    print('Stream len', len(stream))
    chunk_size = len(stream) // N + 1
    prev = 0
    for i in range(N):
        chunk = stream[prev:prev + chunk_size]
        prev += len(chunk)
        with open(os.path.join(path, str(i)), 'wb') as file:
            file.write(chunk)

if __name__ == '__main__':
    import transformers
    model = transformers.BertModel.from_pretrained('rubert-base-uncased')
    model = model.cpu()
    save_model(model, 'splitted')
    #model1 = load_model('splitted')
    #print(model1)
