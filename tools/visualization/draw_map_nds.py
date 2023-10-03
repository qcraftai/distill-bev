import matplotlib.pyplot as plt

# large alpha and context background weight
# epoch = [5, 10, 15, 20]
#
# context_c1_5_map = [26.38, 29.32, 31.56, 31.29]
# context_c1_5_nds = [32.47, 36.26, 40.35, 40.57]
#
# context_c2_5_map = [26.02, 29.45, 31.33, 31.54]
# context_c2_5_nds = [33.16, 37.08, 40.33, 41.03]
#
# large_alpha_8x_map = [26.15, 29.93, 31.91, 32.18] # 4.8e-2
# large_alpha_8x_nds = [33.29, 37.88, 39.98, 41.24] # 4.8e-2
#
# large_alpha_32x_map = [23.47, 28.29, 31.78, 32.68] # 4.8e-2
# large_alpha_32x_nds = [31.21, 36.00, 40.01, 41.04] # 4.8e-2
#
# plt.figure()
# plt.plot(epoch, context_c1_5_map, 'bo-', label='context_c1')
# plt.plot(epoch, context_c2_5_map, 'rv-', label='context_c2')
# plt.plot(epoch, large_alpha_8x_map, 'g^-', label='alpha8x(1.2e-2)')
# plt.plot(epoch, large_alpha_32x_map, 'cs-', label='alpha32x(4.8e-2)')
# plt.xlabel('epoch')
# plt.ylabel('mAP')
# plt.legend()
# plt.title('mAP')
# plt.savefig('mAP.jpg')
#
# plt.figure()
# plt.plot(epoch, context_c1_5_nds, 'bo-', label='context_c1')
# plt.plot(epoch, context_c2_5_nds, 'rv-', label='context_c2')
# plt.plot(epoch, large_alpha_8x_nds, 'g^-', label='alpha8x(1.2e-2)')
# plt.plot(epoch, large_alpha_32x_nds, 'cs-', label='alpha32x(4.8e-2)')
# plt.xlabel('epoch')
# plt.ylabel('NDS')
# plt.legend()
# plt.title('NDS')
# plt.savefig('NDS.jpg')


# epoch = [5, 10, 15, 20]
#
# large_alpha_32x_map = [23.47, 28.29, 31.78, 32.68] # 6e-2
# large_alpha_32x_nds = [31.21, 36.00, 40.01, 41.04] # 6e-2
#
# large_alpha_32x_twostage5_map = [26.90, 28.86, 31.45, 32.06] # 4.8e-2
# large_alpha_32x_twostage5_nds = [35.38, 36.89, 40.30, 40.76] # 4.8e-2
#
# plt.figure()
# plt.plot(epoch, large_alpha_32x_map, 'cs-', label='alpha32x(4.8e-2)')
# plt.plot(epoch, large_alpha_32x_twostage5_map, 'g^-', label='alpha32x(4.8e-2) two_stage=5+20')
# plt.xlabel('epoch')
# plt.ylabel('mAP')
# plt.legend()
# plt.title('mAP')
# plt.savefig('mAP.jpg')
#
# plt.figure()
# plt.plot(epoch, large_alpha_32x_nds, 'cs-', label='alpha32x(4.8e-2)')
# plt.plot(epoch, large_alpha_32x_twostage5_nds, 'g^-', label='alpha32x(4.8e-2) two_stage=5+20')
# plt.xlabel('epoch')
# plt.ylabel('NDS')
# plt.legend()
# plt.title('NDS')
# plt.savefig('NDS.jpg')


# epoch = [5, 10, 15, 20]
#
# large_alpha_4x_map = [26.26, 29.50, 31.70, 32.08] # 4.8e-2
# large_alpha_4x_nds = [33.86, 38.07, 40.02, 41.16] # 4.8e-2
#
# context_len1_weight1_5_map = [25.54, 30.01, 31.32, 31.64]
# context_len1_weight1_5_nds = [32.32, 36.93, 39.38, 40.02]
#
# context_len1_weight05_5_map = [25.70, 29.45, 31.38, 31.57]
# context_len1_weight05_5_nds = [32.91, 37.51, 40.33, 41.19]
#
# context_len2_weight05_5_map = [26.41, 28.86, 31.08, 31.66]
# context_len2_weight05_5_nds = [33.03, 36.41, 39.38, 40.20]
#
# plt.figure()
# plt.plot(epoch, large_alpha_4x_map, 'cs-', label='alpha4x(6e-3)')
# plt.plot(epoch, context_len1_weight1_5_map, 'g^-', label='context_len1_weight1')
# plt.plot(epoch, context_len1_weight05_5_map, 'bo-', label='context_len1_weight0.5')
# plt.plot(epoch, context_len2_weight05_5_map, 'rv-', label='context_len2_weight0.5')
# plt.xlabel('epoch')
# plt.ylabel('mAP')
# plt.legend()
# plt.title('mAP')
# plt.savefig('mAP.jpg')
#
# plt.figure()
# plt.plot(epoch, large_alpha_4x_nds, 'cs-', label='alpha4x(6e-3)')
# plt.plot(epoch, context_len1_weight1_5_nds, 'g^-', label='context_len1_weight1')
# plt.plot(epoch, context_len1_weight05_5_nds, 'bo-', label='context_len1_weight0.5')
# plt.plot(epoch, context_len2_weight05_5_nds, 'rv-', label='context_len2_weight0.5')
# plt.xlabel('epoch')
# plt.ylabel('NDS')
# plt.legend()
# plt.title('NDS')
# plt.savefig('NDS.jpg')


# epoch = [5, 10, 15, 20]
#
# large_alpha_4x_map = [26.26, 29.50, 31.70, 32.08] # 6e-3
# large_alpha_4x_nds = [33.86, 38.07, 40.02, 41.16] # 6e-3
#
# large_alpha_4x_threestage_map = [27.38, 28.54, 31.07, 31.67]
# large_alpha_4x_threestage_nds = [35.11, 38.03, 40.83, 41.44]
#
# plt.figure()
# plt.plot(epoch, large_alpha_4x_map, 'cs-', label='alpha4x(6e-3)')
# plt.plot(epoch, large_alpha_4x_threestage_map, 'g^-', label='alpha4x(6e-3)_threestage')
# plt.xlabel('epoch')
# plt.ylabel('mAP')
# plt.legend()
# plt.title('mAP')
# plt.savefig('mAP.jpg')
#
# plt.figure()
# plt.plot(epoch, large_alpha_4x_nds, 'cs-', label='alpha4x(6e-3)')
# plt.plot(epoch, large_alpha_4x_threestage_nds, 'g^-', label='alpha4x(6e-3)_threestage')
# plt.xlabel('epoch')
# plt.ylabel('NDS')
# plt.legend()
# plt.title('NDS')
# plt.savefig('NDS.jpg')


# epoch = [4, 8, 12, 16, 20, 24]
#
# bevdepth_map = [18.06, 26.38, 29.50, 31.90, 33.43, 33.35]
# bevdepth_nds = [22.32, 31.91, 35.62, 38.19, 40.01, 40.04]
#
# bevdepth_h_h_1x1conv_map = [19.43, 27.42, 31.25, 32.80, 34.17, 34.38]
# bevdepth_h_h_1x1conv_nds = [24.00, 33.83, 39.08, 40.24, 41.84, 42.00]
#
# bevdepth_b1_b1_upsample4_2layer_map = [28.88, 31.24, 32.81, 33.04, 34.89, 34.83]
# bevdepth_b1_b1_upsample4_2layer_nds = [34.22, 36.50, 39.16, 39.05, 41.30, 41.23]
#
# bevdepth_b2b2_upsample4_2layer_map = [26.62, 30.16, 31.42, 31.85, 33.65, 33.65]
# bevdepth_b2b2_upsample4_2layer_nds = [32.97, 36.35, 37.43, 38.48, 40.88, 40.83]
#
# # bevdepth_b1_b1_downsample4_2layer_map = [27.44, 30.20, 30.95, 31.92, 33.54, 33.66]
# # bevdepth_b1_b1_downsample4_2layer_nds = [32.21, 35.75, 36.48, 37.89, 39.48, 39.55]
#
# bevdepth_hh_1x1conv_b1b1_up4_2layer_map = [27.84, 31.51, 32.91, 32.94, 34.87, 34.95]
# bevdepth_hh_1x1conv_b1b1_up4_2layer_nds = [33.52, 38.36, 39.44, 40.14, 42.24, 42.30]
#
# bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_map = [27.61, 31.46, 32.40, 33.46, 35.21, 35.20]
# bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_nds = [34.11, 38.27, 39.16, 39.57, 41.66, 41.92]
#
#
# plt.figure()
# plt.plot(epoch, bevdepth_map, label='bevdepth')
# plt.plot(epoch, bevdepth_h_h_1x1conv_map, label='head')
# plt.plot(epoch, bevdepth_b1_b1_upsample4_2layer_map, label='b1 upsample4')
# # plt.plot(epoch, bevdepth_b1_b1_downsample4_2layer_map, label='b1 downsample4')
# plt.plot(epoch, bevdepth_b2b2_upsample4_2layer_map, label='b2 upsample4')
# plt.plot(epoch, bevdepth_hh_1x1conv_b1b1_up4_2layer_map, label='h b1')
# plt.plot(epoch, bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_map, label='h b1 b2')
# plt.xlabel('epoch')
# plt.ylabel('mAP')
# plt.legend()
# plt.title('mAP')
# plt.savefig('mAP.jpg')
#
# plt.figure()
# plt.plot(epoch, bevdepth_nds, label='bevdepth')
# plt.plot(epoch, bevdepth_h_h_1x1conv_nds, label='head upsample4')
# plt.plot(epoch, bevdepth_b1_b1_upsample4_2layer_nds, label='b1 upsample4')
# # plt.plot(epoch, bevdepth_b1_b1_downsample4_2layer_nds, label='b1 downsample4')
# plt.plot(epoch, bevdepth_b2b2_upsample4_2layer_nds, label='b2 upsample4')
# plt.plot(epoch, bevdepth_hh_1x1conv_b1b1_up4_2layer_nds, label='h b1')
# plt.plot(epoch, bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_nds, label='h b1 b2')
# plt.xlabel('epoch')
# plt.ylabel('NDS')
# plt.legend()
# plt.title('NDS')
# plt.savefig('NDS.jpg')


epoch = [4, 8, 12, 16, 20, 24]

cpdcn_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_map = [27.75, 31.89, 32.52, 33.54, 35.30, 35.43]
cpdcn_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_nds = [34.31, 39.51, 40.17, 41.02, 42.91, 43.20]
cpdcn_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_aoe = [70.61, 61.43, 56.25, 56.25, 53.66, 52.07]

mvp_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_map = [29.05, 31.88, 33.11, 34.46, 36.08, 36.15]
mvp_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_nds = [34.35, 38.19, 39.60, 40.23, 42.26, 42.71]
mvp_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_aoe = [75.34, 65.00, 63.65, 62.49, 60.74, 58.55]

# baseline_bevdepth_aoe= [1.1037, 0.7255, 0.6827, 0.6255, 0.6111, 0.6105]
baseline_extrana_aoe = [103.91, 73.23, 60.73, 59.36, 55.03, 54.84,]

plt.figure()
plt.plot(epoch, cpdcn_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_map, label='centerpoint-multi_scale-fp')
plt.plot(epoch, mvp_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_map, label='mvp-multi_scale-fp')
plt.xlabel('epoch')
plt.ylabel('mAP')
plt.legend()
plt.title('mAP')
plt.savefig('mAP.jpg')

plt.figure()
plt.plot(epoch, cpdcn_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_nds, label='centerpoint-multi_scale-fp')
plt.plot(epoch, mvp_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_nds, label='mvp-multi_scale-fp')
plt.xlabel('epoch')
plt.ylabel('NDS')
plt.legend()
plt.title('NDS')
plt.savefig('NDS.jpg')

plt.figure()
plt.plot(epoch, cpdcn_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_aoe, label='centerpoint-multi_scale-fp')
plt.plot(epoch, mvp_bevdepth_hh_1x1conv_b1b1_b2b2_up4_2layer_fph_aoe, label='mvp-multi_scale-fp')
plt.plot(epoch, baseline_extrana_aoe, label='baseline-extrana')
plt.xlabel('epoch')
plt.ylabel('mAOE')
plt.legend()
plt.title('mAOE')
plt.savefig('mAOE.jpg')

# epoch = [4, 8, 12, 16, 20, 24]
#
# cpdist_bevdet4d_hh_1x1conv_b1b1_b2b2_up4_2layer_map = [12.34, 23.83, 27.65, 29.17, 33.89, 34.14]
# cpdist_bevdet4d_hh_1x1conv_b1b1_b2b2_up4_2layer_nds = [20.04, 36.10, 39.72, 42.14, 46.17, 46.62]
#
# cpdcn_maxnorm5_bevdet4d_hh_1x1conv_b1b1_b2b2_up4_2layer_map = [27.22, 31.66, 32.25, 33.23, 35.20, 35.14]
# cpdcn_maxnorm5_bevdet4d_hh_1x1conv_b1b1_b2b2_up4_2layer_nds = [39.30, 43.85, 44.03, 45.36, 47.53, 47.67]
#
#
# plt.figure()
# plt.plot(epoch, cpdist_bevdet4d_hh_1x1conv_b1b1_b2b2_up4_2layer_map, label='bevdet4d-multi_scale')
# plt.plot(epoch, cpdcn_maxnorm5_bevdet4d_hh_1x1conv_b1b1_b2b2_up4_2layer_map, label='bevdet4d-multi_scale-max_norm5')
# plt.xlabel('epoch')
# plt.ylabel('mAP')
# plt.legend()
# plt.title('mAP')
# plt.savefig('mAP.jpg')
#
# plt.figure()
# plt.plot(epoch, cpdist_bevdet4d_hh_1x1conv_b1b1_b2b2_up4_2layer_nds, label='bevdet4d-multi_scale')
# plt.plot(epoch, cpdcn_maxnorm5_bevdet4d_hh_1x1conv_b1b1_b2b2_up4_2layer_nds, label='bevdet4d-multi_scale-max_norm5')
# plt.xlabel('epoch')
# plt.ylabel('NDS')
# plt.legend()
# plt.title('NDS')
# plt.savefig('NDS.jpg')