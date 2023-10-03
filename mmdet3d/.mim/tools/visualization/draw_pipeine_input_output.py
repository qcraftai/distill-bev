from nuscenes import NuScenes


nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/vepfs/ML/Users/zeyu/Code/BEVDet/data/nuscenes/', verbose=True)
my_scene = nusc.scene[0]
print(f'my scene: \n{my_scene}')


first_sample_token = my_scene['first_sample_token']
# The rendering command below is commented out because it tends to crash in notebooks
# nusc.render_sample(first_sample_token, out_path='./first_sample.pdf')
my_sample = nusc.get('sample', first_sample_token)
print(f'my_sample: \n {my_sample}')


nusc.list_sample(my_sample['token'])
my_sample_data = my_sample['data']
print(f'my_sample_data: {my_sample_data}')



sensor = 'CAM_FRONT'
cam_front_data = nusc.get('sample_data', my_sample_data[sensor])
print(f'cam_front_data: {cam_front_data}')
nusc.render_sample_data(cam_front_data['token'], out_path='./first_sample_data_cam_front.pdf')



sensor = 'LIDAR_TOP'
lidar_top_data = nusc.get('sample_data', my_sample_data[sensor])
print(f'cam_front_data: {lidar_top_data}')
nusc.render_sample_data(lidar_top_data['token'], with_anns=False, nsweeps=10, out_path='./first_sample_data_lidar_top_10sweeps_nobox.pdf')
nusc.render_sample_data(lidar_top_data['token'], with_anns=True, nsweeps=10, out_path='./first_sample_data_lidar_top_10sweeps_withbox.pdf')