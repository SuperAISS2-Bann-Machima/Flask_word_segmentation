from flask import Flask
from flask_restful import Api, Resource,reqparse
from flask_cors import CORS

import torch as T
import pandas as pd
import pythainlp
import torch.nn as N
import torch.optim as O

char2idx = {None: 0, ' ': 1, '!': 2, '"': 3, '#': 4, '$': 5, '%': 6, '&': 7, "'": 8, '(': 9, ')': 10, '*': 11, '+': 12, ',': 13, '-': 14, '.': 15, '/': 16, '0': 17, '1': 18, '2': 19, '3': 20, '4': 21, '5': 22, '6': 23, '7': 24, '8': 25, '9': 26, ':': 27, ';': 28, '=': 29, '?': 30, '@': 31, 'A': 32, 'B': 33, 'C': 34, 'D': 35, 'E': 36, 'F': 37, 'G': 38, 'H': 39, 'I': 40, 'J': 41, 'K': 42, 'L': 43, 'M': 44, 'N': 45, 'O': 46, 'P': 47, 'Q': 48, 'R': 49, 'S': 50, 'T': 51, 'U': 52, 'V': 53, 'W': 54, 'X': 55, 'Y': 56, 'Z': 57, '[': 58, ']': 59, 'a': 60, 'b': 61, 'c': 62, 'd': 63, 'e': 64, 'f': 65, 'g': 66, 'h': 67, 'i': 68, 'j': 69, 'k': 70, 'l': 71, 'm': 72, 'n': 73, 'o': 74, 'p': 75, 'q': 76, 'r': 77, 's': 78, 't': 79, 'u': 80, 'v': 81, 'w': 82, 'x': 83, 'y': 84, 'z': 85, '\xa0': 86, '®': 87, 'é': 88, 'ü': 89, 'ก': 90, 'ข': 91, 'ฃ': 92, 'ค': 93, 'ฅ': 94, 'ฆ': 95, 'ง': 96, 'จ': 97, 'ฉ': 98, 'ช': 99, 'ซ': 100, 'ฌ': 101, 'ญ': 102, 'ฎ': 103, 'ฏ': 104, 'ฐ': 105, 'ฑ': 106, 'ฒ': 107, 'ณ': 108, 'ด': 109, 'ต': 110, 'ถ': 111, 'ท': 112, 'ธ': 113, 'น': 114, 'บ': 115, 'ป': 116, 'ผ': 117, 'ฝ': 118, 'พ': 119, 'ฟ': 120, 'ภ': 121, 'ม': 122, 'ย': 123, 'ร': 124, 'ฤ': 125, 'ล': 126, 'ฦ': 127, 'ว': 128, 'ศ': 129, 'ษ': 130, 'ส': 131, 'ห': 132, 'ฬ': 133, 'อ': 134, 'ฮ': 135, 'ฯ': 136, 'ะ': 137, 'ั': 138, 'า': 139, 'ำ': 140, 'ิ': 141, 'ี': 142, 'ึ': 143, 'ื': 144, 'ุ': 145, 'ู': 146, 'ฺ': 147, '฿': 148, 'เ': 149, 'แ': 150, 'โ': 151, 'ใ': 152, 'ไ': 153, 'ๅ': 154, 'ๆ': 155, '็': 156, '่': 157, '้': 158, '๊': 159, '๋': 160, '์': 161, 'ํ': 162, '๎': 163, '๏': 164, '๐': 165, '๑': 166, '๒': 167, '๓': 168, '๔': 169, '๕': 170, '๖': 171, '๗': 172, '๘': 173, '๙': 174, '๚': 175, '๛': 176, '\u200e': 177, '–': 178, '—': 179, '‘': 180, '’': 181, '…': 182, '™': 183}
idx2char = [None, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\xa0', '®', 'é', 'ü', 'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ', 'ล', 'ฦ', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ฯ', 'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'ฺ', '฿', 'เ', 'แ', 'โ', 'ใ', 'ไ', 'ๅ', 'ๆ', '็', '่', '้', '๊', '๋', '์', 'ํ', '๎', '๏', '๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙', '๚', '๛', '\u200e', '–', '—', '‘', '’', '…', '™']

def str2idxseq(charseq):
    idxseq = []
    for char in charseq:
        char = char.lower()
        if char in char2idx:
            idxseq.append(char2idx[char])
        else:
            idxseq.append(char2idx[None])
    return idxseq

def idxseq2str(idxseq):
    charseq = []
    for idx in idxseq:
        if idx < len(idx2char):
            charseq.append(idx2char[idx])
        else:
            charseq.append(' ')
    return charseq

def sent2data(sent):
    charidxs = []
    wordbrks = []
    for charseq in sent:
        idxs = str2idxseq(charseq)
        charidxs.extend(idxs)
        wordbrks.extend((len(idxs) - 1) * [False] + [True])
    return (charidxs, wordbrks)

def corpus2dataset(corpus):
    dataset = []
    for sent in corpus:
        charidxs, wordbrks = sent2data(sent)
        dataset.append((charidxs, wordbrks))
    return dataset

def wordbrks2brkvec(wordbrks):
    brkvec = T.LongTensor(len(wordbrks))
    for i in range(len(wordbrks)):
        if wordbrks[i]: brkvec[i] = 0
        else: brkvec[i] = 1
    return brkvec

class WordsegModel(N.Module):
    def __init__(self, dim_charvec, dim_trans, no_layers):
        super(WordsegModel, self).__init__()
        self._dim_charvec = dim_charvec
        self._dim_trans = dim_trans
        self._no_layers = no_layers
        
        self._charemb = N.Embedding(183, self._dim_charvec)
        
        self._rnn = N.GRU(
            self._dim_charvec, self._dim_trans, self._no_layers,
            batch_first=True, bidirectional=True ,dropout=0.2
        )
        self._tanh = N.Tanh()
        self._hidden = N.Linear(2 * self._dim_trans, 2)    # Predicting two classes: break / no break
        self._log_softmax = N.LogSoftmax(dim=1)
        
    def forward(self, charidxs):
        try:
            charvecs = self._charemb(T.LongTensor(charidxs))
            # print('charvecs =\n{}'.format(charvecs))
            ctxvecs, lasthids = self._rnn(charvecs.unsqueeze(0))
            ctxvecs, lasthids = ctxvecs.squeeze(0), lasthids.squeeze(1)
            # print('ctxvecs =\n{}'.format(ctxvecs))
            statevecs = self._hidden(self._tanh(ctxvecs))
            # print('statevecs =\n{}'.format(statevecs))
            brkvecs = self._log_softmax(statevecs)
            # print('brkvecs =\n{}'.format(brkvecs))
            return brkvecs
        except RuntimeError:
            raise RuntimeError(statevecs)

wordseg_model = WordsegModel(dim_charvec=64, dim_trans=32, no_layers=3)
#ใส่ path model
wordseg_model.load_state_dict(T.load('30000DataVersion.pt'))
wordseg_model.eval()

def tokenize(wordseg_model, charseq):
    charidxs = str2idxseq(charseq)
    pred_brkvecs = wordseg_model(charidxs)
    # return pred_brkvecs
    pred_wordbrks = []
    for i in range(len(charidxs)):
        pred_wordbrk = (pred_brkvecs[i][0] > pred_brkvecs[i][1])
        # print(pred_wordbrk)
        pred_wordbrks.append(pred_wordbrk)
    
    sent = []
    word = []
    begpos = 0
    for i in range(len(pred_wordbrks)):
        if pred_wordbrks[i]:
            word.append(charseq[i])
            sent.append(word)
            word = []
            begpos = i
        else:
            word.append(charseq[i])
    if len(word) > 0: sent.append(word)
        
    return sent


app = Flask(__name__)
api = Api(app)
CORS(app)

transection_checked = reqparse.RequestParser()
transection_checked.add_argument("txt",required=True, type=str, help="Required str txt")

class wordseg(Resource):
    def get(self):
        #print("Call! wordseg")
        return 'word segment'

    def post(self):
        args = transection_checked.parse_args()
        print(args['txt'])
        words = tokenize(wordseg_model, args['txt'])
        words = list(map(lambda x:''.join(x),words))
        return words

api.add_resource(wordseg, "/")

if __name__ == '__main__':
    app.run(debug=True)
