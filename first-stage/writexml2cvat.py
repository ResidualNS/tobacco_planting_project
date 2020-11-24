def write_xml_cvat(xml_path, img_name, height, width, N, E):
    from lxml import etree

    # 1级目录
    annotation = etree.Element("annotation")

    # 2级目录
    etree.SubElement(annotation, "version").text = "1.1"
    meta = etree.SubElement(annotation, "meta")
    # image_name = img_path.split('\\')[-1]
    image = etree.Element("image", {"id": '0', "name": img_name, "width": str(width), "height": str(height)})
    annotation.append(image)

    # 3级目录
    task = etree.SubElement(meta, "task")
    etree.SubElement(meta, "dumped").text = "2020-08-26 08:22:06.258023+00:00"

    # 4级目录
    etree.SubElement(task, "id").text = "130"
    etree.SubElement(task, "name").text = "task_name"
    etree.SubElement(task, "size").text = "1"
    etree.SubElement(task, "mode").text = "annotation"
    etree.SubElement(task, "overlap").text = "0"
    etree.SubElement(task, "bugtracker")
    etree.SubElement(task, "created").text = "2020-05-20 01:38:39.362995+00:00"
    etree.SubElement(task, "updated").text = "2020-05-20 08:00:34.872446+00:00"
    etree.SubElement(task, "start_frame").text = "0"
    etree.SubElement(task, "stop_frame").text = "0"
    etree.SubElement(task, "frame_filter").text = ' '
    etree.SubElement(task, "z_order").text = "False"

    labels = etree.SubElement(task, "labels")
    # ---包含5,6级目录---
    label = etree.SubElement(labels, "label")
    etree.SubElement(label, "name").text = "dimo"

    segments = etree.SubElement(task, "segments")
    # ---包含5,6级目录---
    segment = etree.SubElement(segments, "segment")
    etree.SubElement(segment, "id").text = "112"
    etree.SubElement(segment, "start").text = "0"
    etree.SubElement(segment, "stop").text = "0"
    etree.SubElement(segment, "url").text = "http://10.10.0.120:8080/?id=112"

    owner = etree.SubElement(task, "owner")
    # ---包含5,6级目录---
    etree.SubElement(owner, "username").text = "kefgeo"
    etree.SubElement(owner, "email").text = "kefgeo@kefgeo.com"

    assignee = etree.SubElement(task, "assignee")
    # ---包含5,6级目录---
    etree.SubElement(assignee, "username").text = "kefgeo"
    etree.SubElement(assignee, "email").text = "kefgeo@kefgeo.com"

    # meta结束，开始image
    # 保存box
    long_num = 0
    for nn,ee in zip(N,E):
        long_num += 1
        #long=''.join([str(long[i])+',' for i in range(len(long))])[:-1]
        for n in range(0,len(nn)):
            #n = ''.join([str(n[i][0]) + ',' + str(n[i][1]) + ';' for i in range(len(n))])[:-1]
            np=str(nn[n][0][0])+','+str(nn[n][0][1]) +';'+str(nn[n][1][0])+','+str(nn[n][1][1])
            polyline= etree.Element("polyline", {"points": np , "object_id": str(long_num),
                                           "label": 'normal', "occluded": '0'})
            image.append(polyline)
        # for e in range(0,len(ee),2):
        #     #e = ''.join([str(e[i][0]) + ',' + str(e[i][1]) + ';' for i in range(len(e))])[:-1]
        #     ep=str(ee[e][0])+','+str(ee[e][1]) +';'+str(ee[e+1][0])+','+str(ee[e+1][1])
        #     polyline = etree.Element("polyline", {"points": ep, "object_id": str(long_num),
        #                                           "label": 'error', "occluded": '0'})
        #     image.append(polyline)
    #切图合并：
    # long_num = 0
    # for nn,ee in zip(N,E):
    #     long_num += 1
    #     #long=''.join([str(long[i])+',' for i in range(len(long))])[:-1]
    #     for n in range(0,len(nn)):
    #         #n = ''.join([str(n[i][0]) + ',' + str(n[i][1]) + ';' for i in range(len(n))])[:-1]
    #         np=str(nn[n][0][0])+','+str(nn[n][0][1]) +';'+str(nn[n][1][0])+','+str(nn[n][1][1])
    #         polyline= etree.Element("polyline", {"points": np , "object_id": str(long_num),
    #                                        "label": 'normal', "occluded": '0'})
    #         image.append(polyline)
    #     for e in range(0,len(ee)):
    #         #e = ''.join([str(e[i][0]) + ',' + str(e[i][1]) + ';' for i in range(len(e))])[:-1]
    #         ep=str(ee[e][0][0])+','+str(ee[e][0][1]) +';'+str(ee[e][1][0])+','+str(ee[e][1][1])
    #         polyline = etree.Element("polyline", {"points": ep, "object_id": str(long_num),
    #                                               "label": 'error', "occluded": '0'})
    #         image.append(polyline)

    # 2种记录点株方式
    # dianzhu_num = etree.Element("image", {"dianzhu_num": str(dianzhu_num)})
    # image.append(dianzhu_num)
    etree.SubElement(image, "long_num").text = str(long_num)
    print('long_num：', long_num)

    # print(etree.tostring(annotation, pretty_print=True, xml_declaration=True, encoding='UTF-8'))

    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    xml_string = ET.tostring(annotation)
    dom = minidom.parseString(xml_string)

    with open(xml_path, 'w', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')

    # print(etree.tostring(annotation, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
    # etree.ElementTree(anno_tree).write(save_path, encoding='UTF-8', pretty_print=True)

    print('xml_path:', xml_path)
    print('---------------finish save_xml---------------')