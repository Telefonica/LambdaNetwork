import os
import torch
import torchaudio
import librosa
import random
import math
import numpy as np
import torchvision.transforms as transforms
from torchaudio import load
import glob

"""Quick Guide for all transforms:

class PadTrim(object):
def __init__(self, max_len, fill_value=0, channels_first=True):

class VolTransform(object):
def __init__(self, min = -5, max = 5, gain_type = 'db', p=0.5):

class VadTransform(object):
def __init__(self, sample_rate = 16000, p=0.5):

class FadeTransform(object):
def __init__(self, min = 0, max = 0.5, fade_shape = 'linear', p=0.5):
    # fade_shape: “quarter_sine”, “half_sine”, “linear”, “logarithmic”, “exponential”

class CropTransform(object):
def __init__(self, sample_rate = 16000, miliseconds = 100, p=0.5):

class RIRTransform(object):
def __init__(self, fake_rirs_path = '/disks/md1-8T/users/b.dbs/databases/fake_room_impulse_reponse/', p=0.5):

class GaussianSNRTransform(object):
def __init__(self, min_SNR=0.001, max_SNR=1.0, sample_rate=16000, p=0.5):

class TimeStretchTransform(object):
def __init__(self, min_rate=0.75, max_rate=1.25, p=0.5):

class PitchShiftTransform(object):
def __init__(self, min_semitones=-4, max_semitones=4, sample_rate=16000, p=0.5):

class ShiftTransform(object):
def __init__(self, min_fraction=-0.2, max_fraction=0.2, rollover=False, p=0.5):

class ClippingDistortionTransform(object):
def __init__(self, min_percentile_threshold=20, max_percentile_threshold=40, p=0.5):

"""

class PadTrim(object):
    def __init__(self, max_len, fill_value=0, channels_first=True):
        super(PadTrim, self).__init__()
        self.max_len = max_len
        self.fill_value = fill_value
        self.len_dim, self.ch_dim = int(channels_first), int(not channels_first)

    def __call__(self, tensor):
        if self.max_len > tensor.size(self.len_dim):
            padding = [self.max_len - tensor.size(self.len_dim)
                       if (i % 2 == 1) and (i // 2 != self.len_dim)
                       else 0
                       for i in range(4)]
            with torch.no_grad():
                tensor = torch.nn.functional.pad(tensor, padding, "constant", self.fill_value)
        elif self.max_len <= tensor.size(self.len_dim):
            tensor = tensor.narrow(self.len_dim, 0, self.max_len)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(max_len={0})'.format(self.max_len)

'''
Increases or decreases the volume of the tensor (Amplitude)
'''
class VolTransform(torch.nn.Module):
    def __init__(self, min=-5, max=5, gain_type='db', p=0.5):
        super(VolTransform, self).__init__()
        # gain_type: 'amplitude', 'power', 'db'
        self.min = min
        self.max = max
        self.gain_type = gain_type
        self.p = p
        
    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        
        gain = random.uniform(self.min, self.max)
        self.transform = torchaudio.transforms.Vol(gain=gain, gain_type=self.gain_type)
        tensor = self.transform(tensor)
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(gain_type={2}, min={0}, max={1}, p={3})'.format(self.min, self.max, self.gain_type, self.p)
    
'''
Attempts to trim silence and quiet background sounds from the ends of recordings of speech
'''
class VadTransform(torch.nn.Module):
    def __init__(self, sample_rate=16000, p=0.5):
        super(VadTransform, self).__init__()
        self.sample_rate = sample_rate
        self.transform = torchaudio.transforms.Vad(self.sample_rate)
        self.p = p
        
    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        tensor = self.transform(tensor)
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)
    
    
'''
Add a fade in and/or fade out to an waveform.
'''
class FadeTransform(torch.nn.Module):
    def __init__(self, min=0, max=0.5, fade_shape='linear', p=0.5):
        super(FadeTransform, self).__init__()
        # fade_shape: “quarter_sine”, “half_sine”, “linear”, “logarithmic”, “exponential”
        self.min = min
        self.max = max
        self.fade_shape = fade_shape
        self.p = p
    
    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        
        length = tensor.shape[1]
        fade_in = int(random.uniform(self.min, self.max)) * length - 1
        fade_out = int(random.uniform(self.min, self.max)) * length - 1
        self.transform = torchaudio.transforms.Fade(fade_in_len=fade_in, fade_out_len=fade_out, fade_shape=self.fade_shape)
        tensor = self.transform(tensor)
        
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(fade_shape={0}, min={1}, max={2}, p={3})'.format(self.fade_shape, self.min, self.max, self.p)

    
'''
Crops (turns 0) a random length of miliseconds
'''
class CropTransform(torch.nn.Module):
    def __init__(self, sample_rate=16000, miliseconds=100, p=0.5):
        super(CropTransform, self).__init__()
        self.sample_rate = sample_rate
        self.miliseconds = miliseconds
        self.crop_length = int(self.miliseconds * 1e-3 * self.sample_rate)
        self.p = p
    
    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        
        max_start = tensor.shape[1] - self.crop_length
        start = random.randint(0, max_start)
        tensor[:,start:start+self.crop_length] = torch.zeros(self.crop_length)

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(miliseconds={0}, p={1})'.format(self.miliseconds,
                                                                           self.p)

'''
Add a Room impulse response to the waveform
'''
class RIRTransform(torch.nn.Module):
    def __init__(self, fake_rirs_path='/disks/md1-8T/users/b.dbs/databases/fake_room_impulse_reponse/', p=0.5):
        super(RIRTransform, self).__init__()
        self.fake_rirs_path = fake_rirs_path
        self.fake_rirs = next(os.walk(self.fake_rirs_path))[2]
        self.p = p

    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        
        fake_rir = np.loadtxt(self.fake_rirs_path + 'rir_' + str(np.random.randint(0, len(self.fake_rirs))) + '.txt')
        audio = tensor.numpy()
        audio = audio.flatten()
        audio = np.convolve(audio, fake_rir)
        audio -= audio.mean()
        audio = np.reshape(audio,[1, audio.size])
        tensor = torch.tensor(audio)
        tensor = tensor.float()
        
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(path={0}, p={1})'.format(self.fake_rirs_path, self.p)

'''
Add Gaussian Signal to Noise Ratio to the waveform
'''
class GaussianSNRTransform(torch.nn.Module):
    def __init__(self, min_SNR=0.001, max_SNR=1.0, sample_rate=16000, p=0.5):
        super(GaussianSNRTransform, self).__init__()
        self.min_SNR = min_SNR
        self.max_SNR = max_SNR
        self.p = p
    
    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        
        audio = tensor.numpy()
        audio = audio.flatten()
        std = np.std(audio)
        noise_std =  random.uniform(self.min_SNR * std, self.max_SNR * std)
        noise = np.random.normal(0.0, noise_std, len(audio)).astype(np.float32)
        audio += noise
        audio = np.reshape(audio,[1, audio.size])
        tensor = torch.tensor(audio)
        tensor = tensor.float()
        
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(min_SNR={0}, max_SNR={1}, p={2})'.format(self.min_SNR, self.max_SNR, self.p)

'''
Stretch the time of the waveform without affecting its Pitch
'''
class TimeStretchTransform(torch.nn.Module):
    def __init__(self, min_rate=0.75, max_rate=1.25, p=0.5):
        super(TimeStretchTransform, self).__init__()
        assert min_rate > 0.1
        assert max_rate < 5
        assert min_rate <= max_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p
    
    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        
        rate = random.uniform(self.min_rate, self.max_rate)
        audio = tensor.numpy()
        audio = audio.flatten()
        time_stretched = librosa.effects.time_stretch(audio, rate)
        audio = np.reshape(time_stretched,[1, time_stretched.size])
        tensor = torch.tensor(audio)
        tensor = tensor.float()
        
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(min_rate={0}, max_rate={1}, p={2})'.format(self.min_rate, self.max_rate, self.p)

    
'''
Change the pitch of the waveform
'''
class PitchShiftTransform(torch.nn.Module):
    def __init__(self, min_semitones=-4, max_semitones=4, sample_rate=16000, p=0.5):
        super(PitchShiftTransform, self).__init__()
        assert min_semitones >= -12
        assert max_semitones <= 12
        assert min_semitones <= max_semitones
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.sample_rate = sample_rate
        self.p = p
    
    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        
        num_semitones = random.uniform(self.min_semitones, self.max_semitones)

        audio = tensor.numpy()
        audio = audio.flatten()
        pitch_shifted = librosa.effects.pitch_shift(audio, self.sample_rate, n_steps=num_semitones)
        audio = np.reshape(pitch_shifted,[1, pitch_shifted.size])
        tensor = torch.tensor(audio)
        tensor = tensor.float()
        
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(min_semit={0}, max_semit={1}, p={2})'.format(self.min_semitones, self.max_semitones, self.p)


'''
Shift the audio signal some a fraction places (min/max)
'''
class ShiftTransform(torch.nn.Module):
    def __init__(self, min_fraction=-0.2, max_fraction=0.2, rollover=False, p=0.5):
        super(ShiftTransform, self).__init__()
        assert min_fraction >= -1
        assert max_fraction <= 1
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.rollover = rollover
        self.p = p
    
    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        
        audio = tensor.numpy()
        audio = audio.flatten()
        
        num_places_to_shift = int(round(random.uniform(self.min_fraction, self.max_fraction) * len(audio)))
        shifted_samples = np.roll(audio, num_places_to_shift)
        if not self.rollover:
            if num_places_to_shift > 0:
                shifted_samples[:num_places_to_shift] = 0.0
            elif num_places_to_shift < 0:
                shifted_samples[num_places_to_shift:] = 0.0
                
        audio = np.reshape(shifted_samples,[1, shifted_samples.size])
        tensor = torch.tensor(audio)
        tensor = tensor.float()
        
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(min_fract={0}, max_fract={1}, rollover={3}, p={2})'.format(self.min_fraction, self.max_fraction, self.p, self.rollover)


'''
Generate clipping (staturation) distortion to the audio signal
'''
class ClippingDistortionTransform(torch.nn.Module):
    def __init__(self, min_percentile_threshold=20, max_percentile_threshold=40, p=0.5):
        super(ClippingDistortionTransform, self).__init__()
        assert min_percentile_threshold <= max_percentile_threshold
        assert 0 <= min_percentile_threshold <= 100
        assert 0 <= max_percentile_threshold <= 100
        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold
        self.p = p
    
    def __call__(self,tensor):
        if self.p < random.random():
            return tensor
        
        percentile_threshold = random.randint(
                self.min_percentile_threshold, self.max_percentile_threshold
        )
        lower_percentile_threshold = int(percentile_threshold/2)
        lower_threshold, upper_threshold = np.percentile(
            tensor, [lower_percentile_threshold, 100 - lower_percentile_threshold]
        )
        tensor = torch.clamp(tensor, lower_threshold, upper_threshold)
        
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(min_th={0}, max_th={1}, p={2})'.format(self.min_percentile_threshold, self.max_percentile_threshold, self.p)



# PadTrimSeq
class PadTrimSeq(torch.nn.Module):
    """Pad/Trim a 1d-Tensor (Signal or Labels)
    Args:
        tensor (Tensor): Tensor of audio of size (slices x n x c) or (slices x c x n)
        max_len (int): Length to which the tensor will be padded
        channels_first (bool): Pad for channels first tensors.  Default: `True`
    """

    def __init__(self, max_len, fill_value=0, channels_first=True):
        self.max_len = max_len
        self.fill_value = fill_value
        self.len_dim, self.ch_dim = int(channels_first), int(not channels_first)

    def __call__(self, tensor):
        """
        Returns:
            Tensor: (c x n) or (n x c)
        """
        assert tensor.size(self.ch_dim) < 128, \
            "Too many channels ({}) detected, see channels_first param.".format(tensor.size(self.ch_dim))
        
        slices = math.ceil(tensor.size(self.len_dim) / float(self.max_len))
        output = torch.zeros(slices, 1, self.max_len)
        for s in range(slices):
            if self.max_len > tensor.size(self.len_dim):
                padding = [self.max_len - tensor.size(self.len_dim)
                        if (i % 2 == 1) and (i // 2 != self.len_dim)
                        else 0
                        for i in range(4)]
                with torch.no_grad():
                    output[s,:,:] = torch.nn.functional.pad(tensor, padding, "constant", self.fill_value)
            elif self.max_len <= tensor.size(self.len_dim):
                output[s,:,:] = tensor.narrow(self.len_dim, 0, self.max_len)
                tensor = tensor.narrow(self.len_dim, self.max_len, tensor.size(self.len_dim) - self.max_len)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(max_len={0})'.format(self.max_len)


# PadTrimSeq
class PadTrimSeqOverlapping(torch.nn.Module):
    """Pad/Trim a 1d-Tensor (Signal or Labels)
    Args:
        tensor (Tensor): Tensor of audio of size (slices x n x c) or (slices x c x n)
        max_len (int): Length to which the tensor will be padded
        channels_first (bool): Pad for channels first tensors.  Default: `True`
    """

    def __init__(self, max_len, fill_value=0, overlapping=0.5, channels_first=True):
        self.max_len = max_len
        self.fill_value = fill_value
        self.len_dim, self.ch_dim = int(channels_first), int(not channels_first)
        self.overlapping = overlapping

    def __call__(self, tensor):
        """
        Returns:
            Tensor: (c x n) or (n x c)
        """
        assert tensor.size(self.ch_dim) < 128, \
            "Too many channels ({}) detected, see channels_first param.".format(tensor.size(self.ch_dim))
        
        slices = math.ceil(tensor.size(self.len_dim) / float(self.max_len - self.max_len*self.overlapping)-1)
        output = torch.zeros(slices, 1, self.max_len)
        for s in range(slices):
            if self.max_len > tensor.size(self.len_dim):
                padding = [self.max_len - tensor.size(self.len_dim)
                        if (i % 2 == 1) and (i // 2 != self.len_dim)
                        else 0
                        for i in range(4)]
                with torch.no_grad():
                    output[s,:,:] = torch.nn.functional.pad(tensor, padding, "constant", self.fill_value)
            elif self.max_len <= tensor.size(self.len_dim):
                output[s,:,:] = tensor.narrow(self.len_dim, 0, self.max_len)
                tensor = tensor.narrow(self.len_dim, int(self.max_len*self.overlapping), tensor.size(self.len_dim) - int(self.max_len*self.overlapping))

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(max_len={0})'.format(self.max_len)



'''
Transforms the input audio into a fixed length
'''
class LengthTransform(torch.nn.Module):
    def __init__(self, length=16000):
        super(LengthTransform, self).__init__()
        self.length = length
    
    def __call__(self, tensor):

        audio_length = tensor.shape[1]
        if audio_length > self.length:
            start = random.randint(0, audio_length - self.length)
            audio = tensor[:, start:start+self.length]
        elif audio_length < self.length:
            start = random.randint(0, self.length - audio_length)
            audio = torch.zeros(1, self.length)
            audio[:, start:start+audio_length] = tensor
        else:
            audio = tensor

        return audio
        
    def __repr__(self):
        return self.__class__.__name__ + '(length={0})'.format(self.length)


class AddBackgroundNoise(torch.nn.Module):
    def __init__(self, p=0.7, vol_range=(0.1, 0.25), background_path='datasets/SpeechCommands/speech_commands_v0.02/_background_noise_/'):
        super(AddBackgroundNoise, self).__init__()
        self.p = p
        self.vol_range = vol_range
        self.backgrounds = glob.glob(background_path + '*.wav')
    
    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        
        background_noise_path = self.backgrounds[random.randint(0, len(self.backgrounds)-1)]
        background_noise = load(background_noise_path)[0]
        
        audio_length = tensor.shape[1]
        background_length = background_noise.shape[1]
        
        start_sample = random.randint(0, background_length-audio_length)
        background_noise = background_noise[:,start_sample:start_sample+audio_length]

        return tensor + random.uniform(*self.vol_range)*background_noise

    def __repr__(self):
        return self.__class__.__name__ + '(vol_range={0}'.format(self.vol_range)