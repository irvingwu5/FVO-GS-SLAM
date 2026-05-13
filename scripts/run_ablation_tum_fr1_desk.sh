#!/bin/sh

# TUM fr1_desk — 渐进消融实验 (GPU 1)
# 每轮在前一轮基础上新增一个模块组
#
#   A: All OFF          (baseline)
#   B: + VOPrior        (tracking refine 60 iters)
#   C: + RSKM
#   D: + FFT mask + Freq sampling
#   E: + Error mask + RGB error
#   F: + SA depth
#   G: + SA dist        (FULL)

export CUDA_VISIBLE_DEVICES=1

python slam.py --config configs/rgbd/tum/ablation_fr1/A_all_off.yaml --eval 2>&1 | tee outputs/aba_fr1_desk_A_all_off.log
python slam.py --config configs/rgbd/tum/ablation_fr1/B_vo.yaml --eval 2>&1 | tee outputs/aba_fr1_desk_B_vo.log
python slam.py --config configs/rgbd/tum/ablation_fr1/C_vo_rskm.yaml --eval 2>&1 | tee outputs/aba_fr1_desk_C_vo_rskm.log
python slam.py --config configs/rgbd/tum/ablation_fr1/D_vo_rskm_fft.yaml --eval 2>&1 | tee outputs/aba_fr1_desk_D_vo_rskm_fft.log
python slam.py --config configs/rgbd/tum/ablation_fr1/E_vo_rskm_fft_err.yaml --eval 2>&1 | tee outputs/aba_fr1_desk_E_vo_rskm_fft_err.log
python slam.py --config configs/rgbd/tum/ablation_fr1/F_vo_rskm_fft_err_sad.yaml --eval 2>&1 | tee outputs/aba_fr1_desk_F_vo_rskm_fft_err_sad.log
python slam.py --config configs/rgbd/tum/ablation_fr1/G_full.yaml --eval 2>&1 | tee outputs/aba_fr1_desk_G_full.log
