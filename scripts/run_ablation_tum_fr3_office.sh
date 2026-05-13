#!/bin/sh

# TUM fr3_office — 渐进消融实验 (GPU 1)
#   A: All OFF → B: +VO → C: +RSKM → D: +FFT+Freq → E: +Error+RGB → F: +SA depth → G: +SA dist

export CUDA_VISIBLE_DEVICES=1

python slam.py --config configs/rgbd/tum/ablation_fr3/A_all_off.yaml --eval 2>&1 | tee outputs/aba_fr3_office_A_all_off.log
python slam.py --config configs/rgbd/tum/ablation_fr3/B_vo.yaml --eval 2>&1 | tee outputs/aba_fr3_office_B_vo.log
python slam.py --config configs/rgbd/tum/ablation_fr3/C_vo_rskm.yaml --eval 2>&1 | tee outputs/aba_fr3_office_C_vo_rskm.log
python slam.py --config configs/rgbd/tum/ablation_fr3/D_vo_rskm_fft.yaml --eval 2>&1 | tee outputs/aba_fr3_office_D_vo_rskm_fft.log
python slam.py --config configs/rgbd/tum/ablation_fr3/E_vo_rskm_fft_err.yaml --eval 2>&1 | tee outputs/aba_fr3_office_E_vo_rskm_fft_err.log
python slam.py --config configs/rgbd/tum/ablation_fr3/F_vo_rskm_fft_err_sad.yaml --eval 2>&1 | tee outputs/aba_fr3_office_F_vo_rskm_fft_err_sad.log
python slam.py --config configs/rgbd/tum/ablation_fr3/G_full.yaml --eval 2>&1 | tee outputs/aba_fr3_office_G_full.log
