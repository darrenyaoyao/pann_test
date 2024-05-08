import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import struct

f = open("golden.txt", "r")
goldendata = f.read().split()
gdata = []
for i in range(int(len(goldendata)/2)):
    gdata.append(int(goldendata[2*i+1]+goldendata[2*i], 16))

fix_length = int(8192/2)
nfft = 1024
step = int(nfft/2)

def float_to_hex(f):
    return ("%f" % struct.unpack('<f', struct.pack('<f', f)))

def mystft(indata, n_fft, hop_length = step):
    fft_window = scipy.signal.windows.hann(n_fft, sym=False)
    for j in range(n_fft):
        fft_window[j] = float(float_to_hex(fft_window[j]))
    outdata = []
    for i in range(int(fix_length/hop_length)-1):
        data = indata[i*hop_length:i*hop_length+n_fft]
        data = data.reshape(n_fft)
        out = np.fft.fft(data*fft_window)
        outdata.append(out)

    return np.array(outdata)

x = np.array(gdata)
X = np.abs(mystft(x, n_fft=nfft))
X = X[:,0:512]
print("Golden Data")
for j in range(512):
    print(X[0, j])

f = open("test.txt", "r")
testdata = f.read().split()
tdata = []
for i in range(int(len(testdata)/2)):
    tdata.append(int(testdata[2*i+1]+testdata[2*i], 16))

y = np.array(tdata)
Y = np.abs(mystft(y, n_fft=nfft))
Y = Y[:,0:512]
print("Test Data")
for j in range(512):
    print(Y[0, j])

# import matplotlib.pyplot as plt

indexs = range(4096)
freqindexs = range(512)
# plt.plot(indexs, gdata, 'r')
# plt.plot(indexs, tdata, 'b')
# plt.plot(freqindexs, X[0, 0:512], 'r')
# plt.plot(freqindexs, Y[0, 0:512], 'b')
# plt.show()


cosine_similarity_result = cosine_similarity(X[0, 1:512].reshape(1, -1), Y[0, 1:512].reshape(1, -1))
print(f"Cosine Distance using scikit-learn: {1 - cosine_similarity_result[0][0]}")

