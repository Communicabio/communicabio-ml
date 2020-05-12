import pickle
import os

def load_model(path):
    stream = None
    for filename in sorted(os.listdir(path)):
        with open(os.path.join(path, filename), 'rb') as file:
            new_stream = file.read()
        if stream is None:
            stream = new_stream
        else:
            stream += new_stream
        os.remove(os.path.join(path, filename))
    print('Stream len', len(stream))
    model = pickle.loads(stream)
    return model

def save_model(model, path, N=9, prefix='chunk_'):
    assert(N < 10)
    try:
        os.mkdir(path)
    except FileExistsError as exp:
        pass
    stream = pickle.dumps(model)
    print('Stream len', len(stream))
    chunk_size = len(stream) // N + 1
    prev = 0
    for i in range(N):
        chunk = stream[prev:prev + chunk_size]
        prev += len(chunk)
        with open(os.path.join(path, prefix + str(i)), 'wb') as file:
            file.write(chunk)

if __name__ == '__main__':
    import transformers
    '''import torch
    model = torch.rand((int(10**6), ))
    torch.save(model, 'model')
    model = model.half()
    torch.save(model, open('half.torch', 'wb'))
    pickle.dump(model, open('half.pckl', 'wb'))'''
    model = transformers.GPT2LMHeadModel.from_pretrained('../ru-GPT2Like')
    model = model.half()
    save_model(model, 'splited')
    # load_model('splited')
