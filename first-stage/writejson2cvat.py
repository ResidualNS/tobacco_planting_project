# -*- coding:utf-8 -*-
import json
def output_cvat_json(img_path,json_path_save, height, width,xiangyuan, contours,PP,Longju):
    # 输入参数
    name = img_path.split('/')[-1]
    #anno = {}

    # # 创建version
    # anno['version'] = '1.1'
    #
    # # 创建meta
    # meta = {}
    # task = {'id': None, 'name': None, 'size': None, 'mode': 'annotation',
    #         'overlap': '0', 'bugtracker': None, 'created': None, 'updated': None,
    #         'start_frame': '0', 'stop_frame': '0', 'frame_filter': '1'}
    # tmp_label = []
    # # tmp_label.append({'name': 'region'})
    # tmp_label.append({'name': 'long_line'})
    # # tmp_label = [{'name': 'region'}, {'name': 'long_line'}]
    # task['labels'] = {'label': tmp_label}
    # task['segments'] = {'segment': {'id': None, 'start': None, 'stop': None, 'url': None}}
    # task['owner'] = {'username': 'admin', 'email': ''}
    #
    # meta['task'] = task
    # meta['dumped'] = None
    # anno['meta'] = meta

    # 创建image
    image = {}
    image['image_name'] = name
    image['image_height'] = height
    image['image_width'] = width
    image['image_xiangyuan']=xiangyuan

#轮廓
    if type(contours) == dict:
        polygon_tmp= contours
    else:
        polygon_tmp = {}
        id=0
        for i in contours:
            i=i.reshape(-1,2)
            i=i.tolist()
            polygon_tmp.setdefault('outline_'+str(id),i)
            id += 1
#陇距
    point_tmp={}
    id=0
    for pp,longju in zip(PP,Longju):
        outline_2 =[]
        if len(pp) == 0:
            id += 1
            continue
        else:
            for m in range(len(pp)):
                outpoint_m=[pp[m][0][0],pp[m][0][1],pp[m][1][0],pp[m][1][1],longju[m]]
                outline_2.append(outpoint_m)
        point_tmp.setdefault('outline_' + str(id)+'_'+str(id+1),outline_2)
        id += 1

    image['contours'] = polygon_tmp
    image['point_pair'] = point_tmp
    #anno['image'] = image

    # 创建最后的annotations
    #myinfo = {}
    #myinfo['annotations'] = anno

    # print(myinfo)
    write_json(image, json_path_save)
    print('-----------识别结果保存为： ', json_path_save)

def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)