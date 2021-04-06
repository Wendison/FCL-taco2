# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:13:27 2020

@author: Disong Wang

data processing:
(1) use textgrid obtained from MFA to make phn-seq and extract phn-duration
(2) extract & normalize mel-spectrograms/F0/energy
(3) make json files for train/val/test
"""

import configargparse
import sys

from espnet.transform.spectrogram import logmelspectrogram
from espnet.transform.spectrogram import stft
import numpy as np
import resampy
import soundfile as sf
from joblib import Parallel, delayed
from glob import glob
import os
import random
from tqdm import tqdm
import json
import tgt
import pyworld as pw

def acoustic_features_process_one_utterance(wav_path, args, utt2dur_phn):
    uttid = os.path.basename(wav_path).split('.')[0]
    # extract mel-spectrogram (log)
    wav, fs = sf.read(wav_path)
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    if fs != args.set_fs:
        wav = resampy.resample(wav, fs, args.set_fs, axis=0)
        fs = args.set_fs
    mel = logmelspectrogram(
                x=wav,
                fs=fs,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                n_shift=args.n_shifts,
                win_length=args.win_length,
                window=args.windows,
                fmin=args.fmin,
                fmax=args.fmax,
            )
    
    # make sum(dur) = mel length & save durations
    durations = utt2dur_phn[uttid][0]
    durations[-1] += mel.shape[0] - sum(durations)
    durations = np.array(durations,dtype=float).reshape(-1,1)
    dur_save_root = f'{args.feature_root}/durations_MFA'
    os.makedirs(dur_save_root, exist_ok=True)
    dur_save_path = f'{dur_save_root}/{uttid}.npy'
    np.save(dur_save_path, durations)

    # extract phn-level F0 & energy
    tlen = mel.shape[0]
    frame_period = args.n_shifts/fs*1000
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    
    x_mag = np.abs(stft(wav, args.n_fft, args.n_shifts, win_length=args.win_length, window=args.windows)) # T x F
    energy = np.linalg.norm(x_mag, axis=1).reshape(-1)
    assert len(energy) == tlen
    
    durs = durations.reshape(-1)
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))
    pitch_phn = np.zeros((durs.shape[0],), dtype=np.float)
    energy_phn = np.zeros((durs.shape[0],), dtype=np.float)
    for idx, a, b in zip(range(durs.shape[0]), durs_cum[:-1], durs_cum[1:]):
        a = int(a)
        b = int(b)
        values = lf0[a:b][np.where(f0[a:b] != 0.0)[0]] # use avg-lf0 instead of avg-f0
        pitch_phn[idx] = np.mean(values) if len(values) > 0 else 0.0
        values = energy[a:b]
        energy_phn[idx] = np.mean(values) if len(values) > 0 else 0.0
    
    f0 = pitch_phn
    energy = energy_phn
    
    mel_save_path = f'{args.feature_root}/mels-ori/{uttid}.npy'
    f0_save_path = f'{args.feature_root}/f0-ori/{uttid}.npy'
    en_save_path = f'{args.feature_root}/en-ori/{uttid}.npy'
    os.makedirs(os.path.dirname(mel_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(f0_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(en_save_path), exist_ok=True)
    np.save(mel_save_path, mel)
    np.save(f0_save_path, f0)
    np.save(en_save_path, energy)
    
    return uttid, mel, f0, energy


def acoustic_features_save_one_utterance(uttid, utt2mel_f0_en, args):
    mel, f0, en = utt2mel_f0_en[uttid]
    mel_save_path = f'{args.feature_root}/mels/{uttid}.npy'
    f0_save_path = f'{args.feature_root}/f0/{uttid}.npy'
    en_save_path = f'{args.feature_root}/en/{uttid}.npy'
    os.makedirs(os.path.dirname(mel_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(f0_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(en_save_path), exist_ok=True)
    np.save(mel_save_path, mel)
    np.save(f0_save_path, f0)
    np.save(en_save_path, en)
    return uttid, mel_save_path, f0_save_path, en_save_path


def normalize_save_acoustic_features(utt2mel_f0_en, train_uttid, args):
    all_mels = []
    all_f0 = []
    all_en = []
    for uttid in train_uttid:
        mel, f0, en = utt2mel_f0_en[uttid]
        all_mels.append(mel)
        all_f0.append(f0[np.where(f0!=0.0)[0]]) # only use non-zero lf0 for mean/std calculation
        all_en.append(en)
    
    all_mels = np.concatenate(all_mels, 0)
    all_f0 = np.concatenate(all_f0, 0)
    all_en = np.concatenate(all_en, 0)
    mel_mean, mel_std = np.mean(all_mels,0), np.std(all_mels,0)
    f0_mean, f0_std = np.mean(all_f0,0), np.std(all_f0,0)
    en_mean, en_std = np.mean(all_en,0), np.std(all_en,0)
    
    mel_stats = np.concatenate([mel_mean.reshape(1,-1), mel_std.reshape(1,-1)],0)
    f0_en_stats = np.array([f0_mean, f0_std, en_mean, en_std])
    np.save(f'{args.feature_root}/mel_stats.npy', mel_stats)
    np.save(f'{args.feature_root}/f0_en_stats.npy', f0_en_stats)
    
    for uttid in utt2mel_f0_en.keys():
        mel, f0, en = utt2mel_f0_en[uttid]
        mel = (mel - mel_mean) / (mel_std + 1e-8)
        zero_idxs = np.where(f0 == 0.0)[0]
        f0 = (f0 - f0_mean) / (f0_std + 1e-8)
        f0[zero_idxs] = 0.0
        en = (en - en_mean) / (en_std + 1e-8)
        utt2mel_f0_en[uttid] = [mel, f0.reshape(-1,1), en.reshape(-1,1)]
    
    # TODO: also here parallel goes wrong...
    # results = Parallel(n_jobs=-1)(delayed(acoustic_features_save_one_utterance)(uttid, utt2mel_f0_en, args) for uttid in tqdm(utt2mel_f0_en.keys()))
    results = []
    for uttid in tqdm(utt2mel_f0_en.keys()):
        tmp = acoustic_features_save_one_utterance(uttid, utt2mel_f0_en, args)
        results.append(tmp)
    utt2ac_path = {re[0]: [re[1], re[2], re[3]] for re in results}    
    return utt2ac_path
    

def get_phones(tier):
    phones = []
    for t in tier._objects:
        phones.append(t.text)
    return phones


def get_alignment(tgt_path, phn2idx, feature_root, sampling_rate, hop_length, args):
    textgrid = tgt.io.read_textgrid(tgt_path)
    tier = textgrid.get_tier_by_name('phones')
    uttid = tgt_path.split('/')[-1].split('.')[0]
    sil_phones = ['sil', 'sp', 'spn']
    phones = []
    durations = []
    parts = []
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text
        parts.append([int(s*sampling_rate), int(e*sampling_rate), p])
    
    if parts[-1][-1] in ['', 'sp', 'spn']: # set last empty to sil
        parts[-1][-1] = 'sil'
    
    if parts[-2][-1] in sil_phones and parts[-1][-1] == 'sil': 
        parts[-2][-1] = 'sil'
        parts[-2][1] = parts[-1][1]
        parts = parts[:-1]
    
    for ix, item in enumerate(parts):
        phones.append(item[-1])
        durations.append(int(item[1]/hop_length) - int(item[0]/hop_length))
    
    idx = [phn2idx[p] for p in phones]
    
    dur_save_root = f'{args.feature_root}/durations_MFA-ori'
    os.makedirs(dur_save_root, exist_ok=True)
    dur_save_path = f'{dur_save_root}/{uttid}.npy'
    np.save(dur_save_path, np.array(durations).reshape(-1,1))
    
    return uttid, durations, phones, idx


def make_json(utt2dur_phn, utt2mel_f0_en, utt2ac_path, uttidx, mode, num_phns, args):
    js = {}
    for uttid in uttidx:
        durations, phones, idx = utt2dur_phn[uttid]
        if max(durations) <= 50: # only select utterances with max-phn-dur <= 50
            mel, f0, en = utt2mel_f0_en[uttid]
            mel_path, f0_path, en_path = utt2ac_path[uttid]
            mel_info = {'feat': mel_path, 
                        'filetype': 'npy', 
                        'name': 'input1', 
                        'shape': mel.shape}
            dur_info = {'feat': f'{args.feature_root}/durations_MFA/{uttid}.npy', 
                        'filetype': 'npy', 
                        'name': 'input2', 
                        'shape': [len(durations),1]}
            f0_info = {'feat': f0_path, 
                        'filetype': 'npy', 
                        'name': 'input3', 
                        'shape': [len(f0),1]}
            en_info = {'feat': en_path, 
                        'filetype': 'npy', 
                        'name': 'input4', 
                        'shape': [len(en),1]}
            txt_info = {'name': 'target1',
                        'shape': [len(phones),num_phns],
                        'text': ' '.join(phones),
                        'token': ' '.join(phones),
                        'tokenid': ' '.join(idx)}
            js[uttid] = {}
            js[uttid]['input'] = [mel_info, dur_info, f0_info, en_info]
            js[uttid]['output'] = [txt_info]
            js[uttid]['utt2spk'] = 'LJ'
        
    js = {'utts': js}
    fp = open(f'{args.feature_root}/{mode}_data.json', 'w')
    json.dump(js, 
              fp, 
              indent=4,
              ensure_ascii=False,
              sort_keys=True,
              separators=(",", ": "),
             )
    fp.close()
        

def get_parser():
    """Get parser of data processing arguments."""
    parser = configargparse.ArgumentParser()
    # path
    parser.add("--data-root", type=str, default='/mnt/data02/d00225230/wangdisong/Dataset/LJSpeech-1.1', help="path of original wavs")
    parser.add("--feature-root", type=str, default='data', help="path for saving acoustic features")
    parser.add("--textgrid-root", type=str, default='TextGrid', help="path for textgrid files obtained from MFA")
    # feature settings
    parser.add("--set-fs", type=int, default=22050, help="default sampling frequency")
    parser.add("--fmax", type=int, default=7600, help="maximum frequency")
    parser.add("--fmin", type=int, default=80, help="minimum frequency")
    parser.add("--n-mels", type=int, default=80, help="number of mel basis")
    parser.add("--n-fft", type=int, default=1024, help="fft size")
    parser.add("--n-shifts", type=int, default=256, help="hop size")
    parser.add("--win-length", default=None, help="window length, if it's None, it's set to be n_fft")
    parser.add("--windows", type=str, default='hann', help="window type")
    return parser


def main(cmd_args):
    """Run training."""
    #print('cmd_args:', cmd_args)
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    os.makedirs(args.feature_root, exist_ok=True)
    #================== extract phn-durations & phn-seq for each sentence ==================#
    print('extract phn-durations & phn-seq for each sentence...')
    all_phones = []
    for tgt_path in tqdm(glob(f'{args.textgrid_root}/*.TextGrid')):
        textgrid = tgt.io.read_textgrid(tgt_path)
        phones = get_phones(textgrid.get_tier_by_name('phones'))
        all_phones += phones
    
    all_phones = list(set(all_phones))
    print('all_phones:', all_phones, 'len:', len(all_phones))
    all_phones = sorted(all_phones)
    phn2idx = {p:str(i) for i,p in enumerate(all_phones,1)}
    phn2idx["PAD"] = 0
    fr = open(f'{args.feature_root}/phn2idx.json','w')
    json.dump(
            phn2idx,
            fr,
            indent=4,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ": "),
            )
    fr.close()
    tg_paths = glob(f'{args.textgrid_root}/*.TextGrid')
    results = Parallel(n_jobs=-1)(delayed(get_alignment)(tgt_path, phn2idx, args.feature_root, args.set_fs, args.n_shifts, args) for tgt_path in tqdm(tg_paths))
    utt2dur_phn = {re[0]: [re[1], re[2], re[3]] for re in results}
    
    #================== extract & normalize mel/F0/energy each sentence & split dataset into train/val/test ==================#
    print('extract & normalize mel/F0/energy for each sentence...')
    wav_paths = glob(f'{args.data_root}/wavs/*.wav')
    # TODO: Something goes wrong when using parallel...
    # results = Parallel(n_jobs=-1)(delayed(acoustic_features_process_one_utterance)(wav_path, args, utt2dur_phn) for wav_path in tqdm(wav_paths))
    results = []
    for wav_path in tqdm(wav_paths):
        re = acoustic_features_process_one_utterance(wav_path, args, utt2dur_phn)
        results.append(re)
    utt2mel_f0_en = {re[0]: [re[1], re[2], re[3]] for re in results}
    all_uttidx = [key for key in utt2mel_f0_en.keys()]
    val_test_uttid = random.sample(all_uttidx, 1000)
    val_uttid = random.sample(val_test_uttid, 500)
    test_uttid = [fi for fi in val_test_uttid if fi not in val_uttid]
    train_uttid = [fi for fi in all_uttidx if fi not in val_test_uttid]
    utt2ac_path = normalize_save_acoustic_features(utt2mel_f0_en, train_uttid, args)
    print('acoustic features processing done!')
    
    #================= make train/val/test json files ====================#
    print('make train/val/test json files...')
    num_phns = len(phn2idx.keys())
    make_json(utt2dur_phn, utt2mel_f0_en, utt2ac_path, train_uttid, 'train', num_phns, args)
    make_json(utt2dur_phn, utt2mel_f0_en, utt2ac_path, val_uttid, 'val', num_phns, args)
    make_json(utt2dur_phn, utt2mel_f0_en, utt2ac_path, test_uttid, 'test', num_phns, args)
    print('make json files done!')
    
        
if __name__ == "__main__":
    main(sys.argv[1:])
    
    
    


