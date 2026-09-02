import numpy as np
words = np.load('outputs/alignment/ep1_words.npy', allow_pickle=True).tolist()
print('Words:', len(words))
for w in words[:5]:
    print(' ', w['word'], w['start_ms'], 'ms to', w['end_ms'], 'ms chunk=', w['chunk_index'])
print('...')
for w in words[-3:]:
    print(' ', w['word'], w['start_ms'], 'ms to', w['end_ms'], 'ms chunk=', w['chunk_index'])