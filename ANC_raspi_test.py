# -*- coding: utf-8 -*-

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

##############
##   freq   ##
##############
def freq(wave_data,framerate,nframes):
    # 通过取样点数和取样频率计算出每个取样的时间。
    time = np.arange(0, nframes) / framerate
    # 从波形数据中取样nframes个点进行运算
    xs = wave_data[:nframes]
    xf = np.fft.rfft(xs)
    # 于是可以通过下面的np.linspace计算出返回值中每个下标对应的真正的频率：
    freqs = np.linspace(0, framerate / 2, nframes / 2 + 1)

    plt.subplot(211)
    plt.plot(time[:nframes], xs)
    plt.xlabel("time(s)")
    plt.title('Original wave')
    plt.subplot(212)
    plt.plot(freqs, np.abs(xf), 'r')  # 显示原始信号的FFT模值
    plt.title('FFT of Mixed wave(two sides frequency range)')

    plt.savefig("freq.jpg")
    # plt.show()

##############
##   time   ##
##############
def time_plot(wave_data,framerate,nframes,name):
    # 通过取样点数和取样频率计算出每个取样的时间。
    time = np.arange(0, nframes) / framerate
    # 从波形数据中取样nframes个点进行运算
    xs = wave_data[:nframes]
    plt.plot(time[:nframes], xs)
    plt.xlabel("time(s)")
    plt.title(name)

    # plt.savefig("freq.jpg")
    plt.show()

##############
##  lowess  ##
##############
def lowess(wave_data,nframes):
    # x = np.linspace(1, nframes, nframes)
    # # print(x)
    # # result = lowess(wave_data, x, frac=0.2, it=3, delta=0.0)
    # import statsmodels.api as sm
    # lowess = sm.nonparametric.lowess
    # print(nframes)
    # rate = 30/nframes
    # print(rate)
    # result = lowess(wave_data, x, frac=rate,delta=1)
    # print(1)

    x = np.linspace(1, nframes, nframes)
    y = wave_data
    d = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    df = DataFrame(d, columns=['xdata', 'ydata'])
    sns.lmplot(x='xdata', y='ydata', data=df, lowess=True)
    result = 0

    return result

# 自定义函数 np.convolve()
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

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

# 提取高频一次攻击信号
b, a = signal.butter(20, 0.27, 'high')
attack_base_sig = signal.filtfilt(b, a, wave_data_upsample)
# attack_base_sig = attack_base_sig*1.0/(max(abs(attack_base_sig)))
b, a = signal.butter(30, 0.38, 'low')
attack_base_sig = signal.filtfilt(b, a, attack_base_sig)
attack_base_sig = attack_base_sig*1.0/(max(abs(attack_base_sig)))
# freq(attack_base_sig,upsample_rate,nframes_upsample,'attack_base_sig')
# time_plot(attack_base_sig,upsample_rate,nframes_upsample,'attack_base_sig')

# 提取高频二次攻击信号
attack_sec_sig = attack_base_sig*attack_base_sig
b, a = signal.butter(30, 0.20, 'low')
attack_sec_sig = signal.filtfilt(b, a, attack_sec_sig)
attack_sec_sig = attack_sec_sig*1.0/(max(abs(attack_sec_sig)))
# freq(attack_sec_sig,upsample_rate,nframes_upsample,'attack_sec_sig')
# time_plot(attack_sec_sig,upsample_rate,nframes_upsample,'attack_sec_sig')

# single time slot ANC
# LMS filter parameters

mix_base_sig = np.dot(mix_base_sig,1000)
attack_sec_sig = np.dot(attack_sec_sig,1000)

matrix_tmp = np.array([mix_base_sig,attack_sec_sig])
coef = np.corrcoef(matrix_tmp)
coef = coef[0,1]
print("coef:",coef)

framesize = 30  # 采样的点数
M = 30  # 滤波器的阶数
mu = 3.0342e-005  # 步长因子

Niter = int(nframes_upsample/framesize)
error_anc = np.zeros(nframes_upsample)
for k in range(Niter):
    xn = attack_sec_sig[k*framesize:(k+1)*framesize]
    dn = mix_base_sig[k*framesize:(k+1)*framesize]
    # 调用LMS算法
    (yn, en) = LMS(xn, dn, M, mu, framesize)
    error_anc[k*framesize:(k+1)*framesize] = en

# lpf
b, a = signal.butter(15, 0.06, 'low')
error_anc = signal.filtfilt(b, a, error_anc)

time_end = time.time()
# freq(error_anc,upsample_rate,nframes_upsample,'error_anc')

# time_plot(mix_base_sig,upsample_rate,nframes_upsample,'mix_base_sig')
# time_plot(attack_sec_sig,upsample_rate,nframes_upsample,'attack_sec_sig')
# time_plot(error_anc,upsample_rate,nframes_upsample,'error_anc')
#
# write mix_base_sig.wav(amplified by 10000)
#out = wave.open('/home/raspi/Desktop/ANC/mix_base_sig.wav', 'wb')
#out.setnchannels(nchannels)
#out.setsampwidth(sampwidth)
#out.setframerate(upsample_rate)
# time_plot(mix_base_sig,upsample_rate,nframes_upsample,'mix_base_sig')
#mix_base_sig = np.dot(mix_base_sig,30)
#mix_base_sig = mix_base_sig.astype(np.int16)
# time_plot(mix_base_sig,upsample_rate,nframes_upsample,'mix_base_sig')
#out.writeframes(mix_base_sig.tostring())
#out.close()

# write attack_sec_sig.wav(amplified by 30)
out = wave.open('/home/raspi/Desktop/ANC/attack_sec_sig.wav', 'wb')
out.setnchannels(nchannels)
out.setsampwidth(sampwidth)
out.setframerate(upsample_rate)
attack_sec_sig = np.dot(attack_sec_sig,30)
attack_sec_sig = attack_sec_sig.astype(np.int16)
out.writeframes(attack_sec_sig.tostring())
out.close()

# write error_out.wav
out = wave.open('/home/raspi/Desktop/ANC/error_out.wav', 'wb')
out.setnchannels(nchannels)
out.setsampwidth(sampwidth)
out.setframerate(upsample_rate)
# time_plot(mix_base_sig,upsample_rate,nframes_upsample,'error_anc')
# error_anc = error_anc*1.0/(max(abs(error_anc)))
error_anc = np.dot(error_anc,100)
absmean = np.mean(abs(error_anc))
print(absmean)
#if absmean < 50:
	#error_anc = np.dot(error_anc,0)
	#error_anc = zeros(nframes_upsample)
# time_plot(error_anc,upsample_rate,nframes_upsample,'error_anc')
error_anc = error_anc.astype(np.int16)
out.writeframes(error_anc.tostring())
out.close()

print('time:',time_end - time_start)

