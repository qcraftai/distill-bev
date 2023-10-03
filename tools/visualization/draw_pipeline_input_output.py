from nuscenes import NuScenes


nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/vepfs/ML/Users/zeyu/Code/BEVDet/data/nuscenes/', verbose=True)

# print(f'my scene: \n{my_scene}')
#
#
# first_sample_token = my_scene['first_sample_token']
# # The rendering command below is commented out because it tends to crash in notebooks
# # nusc.render_sample(first_sample_token, out_path='./first_sample.pdf')
# my_sample = nusc.get('sample', first_sample_token)
# print(f'my_sample: \n {my_sample}')
# tokens = ['56bce8e241bd457dac12a934134b477b', '40599f85823f42b7b2af58331bc466f3', 'e2b98ffff9d249508e1986ea6191bcaa',
#            '7169eced1ce64fa7bd22e4a6fa193b8d', '7d2e7f5c87fa49ec80b5e393c6ef7230', '2bb61f7343cf45ec857b602f72f2d319',
#            'c64c6c970ae94cf8bdf7e98f804eb889', '3082b36a962f4174aff23dd948f5ff85']
tokens = ['7169eced1ce64fa7bd22e4a6fa193b8d',]

# my_scene = nusc.scene[20]
# tokens = list(my_scene['first_sample_token'])
sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP']
for index, token in enumerate(tokens):
    my_sample = nusc.get('sample', token)

    nusc.list_sample(my_sample['token'])
    my_sample_data = my_sample['data']
    # print(f'my_sample_data: {my_sample_data}')

    for sensor in sensors:
        sensor_data = nusc.get('sample_data', my_sample_data[sensor])
        # print(f'{sensor.lower()}_data: {sensor_data}')
        if sensor == 'LIDAR_TOP':
            nusc.render_sample_data(sensor_data['token'], with_anns=False, nsweeps=2, underlay_map=False,
                                    out_path=f'./input_output/{index}th_{sensor.lower()}_data_2sweeps_nobox.jpg')
            nusc.render_sample_data(sensor_data['token'], with_anns=True, nsweeps=2, underlay_map=False,
                                    out_path=f'./input_output/{index}th_{sensor.lower()}_data_2sweeps_withbox.jpg')
        else:
            nusc.render_sample_data(sensor_data['token'], with_anns=False,
                                    out_path=f'./input_output/{index}th_{sensor.lower()}_data_nobox.jpg')
            nusc.render_sample_data(sensor_data['token'], with_anns=True,
                                    out_path=f'./input_output/{index}th_{sensor.lower()}_data_withbox.jpg')