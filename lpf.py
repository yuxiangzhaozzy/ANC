# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import wave
import time
from pylab import *
from lms import *
#import scipy as sc
#from scipy.signal import savgol_filter
# import seaborn as sns
# from pandas import Series, DataFrame

##############
## Resample ##
##############
def Resample(input_signal,src_fs,tar_fs):
    '''

    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    '''

    dtype = input_signal.dtype
    audio_len = len(input_signal)
    audio_time_max = 1.0*(audio_len-1) / src_fs
    src_time = 1.0 * np.linspace(0,audio_len,audio_len) / src_fs
    tar_time = 1.0 * np.linspace(0,np.int(audio_time_max*tar_fs),np.int(audio_time_max*tar_fs)) / tar_fs
    output_signal = np.interp(tar_time,src_time,input_signal).astype(dtype)

    return output_signal

time_start = time.time()
# CHUNK = 1024
wf = wave.open('/home/raspi/Desktop/ANC/test.wav', 'rb')
params = wf.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
# print(nchannels)
# print(sampwidth)
# print(framerate)

# read data
str_data = wf.readframes(nframes)
wf.close()
# generate wave_data
wave_data = np.fromstring(str_data,dtype=np.int16)
#wave_data = smooth(wave_data,29)
# lowess(wave_data,nframes)
# wave_data = lowess(wave_data,nframes)
# wave_data=savgol_filter(wave_data, 101, 3)
# time_plot(wave_data,framerate,nframes,'wave_data')

# upsampling
upsample_rate = np.int(96000)
wave_data_upsample = Resample(wave_data,framerate,upsample_rate)
wave_data_upsample = wave_data_upsample*1.0/(max(abs(wave_data_upsample)))
nframes_upsample = int((nframes-1)*(upsample_rate/framerate))
# 可能是噪声太多,上采之后做FFT变换频谱图出不来(-inf)
# freq(wave_data_upsample,upsample_rate,nframes_upsample,'wave_data_upsample')
# time_plot(wave_data_upsample,upsample_rate,nframes_upsample,'wave_data_upsample')
# lowpass filter: 进入麦克风之后信号的截止
b, a = signal.butter(30, 0.47, 'low')
before_anc_sig = signal.filtfilt(b, a, wave_data_upsample)
# freq(before_anc_sig,upsample_rate,nframes_upsample,'before_anc_sig')
# time_plot(before_anc_sig,upsample_rate,nframes_upsample,'before_anc_sig')

# 提取基带信号+攻击信号的混合信号
b, a = signal.butter(15, 0.06, 'low')
mix_base_sig = signal.filtfilt(b, a, wave_data_upsample)
# freq(mix_base_sig,upsample_rate,nframes_upsample,'mix_base_sig')
# time_plot(mix_base_sig,upsample_rate,nframes_upsample,'mix_base_sig')

mix_base_sig = np.dot(mix_base_sig,1000)

time_end = time.time()

#
# write mix_base_sig.wav(amplified by 10000)
out = wave.open('/home/raspi/Desktop/ANC/mix_base_sig.wav', 'wb')
out.setnchannels(nchannels)
out.setsampwidth(sampwidth)
out.setframerate(upsample_rate)
mix_base_sig = np.dot(mix_base_sig,30)
mix_base_sig = mix_base_sig.astype(np.int16)
out.writeframes(mix_base_sig.tostring())
out.close()


print('time:',time_end - time_start)

