import json
import copy


if __name__ == "__main__":

    cat_json = json.load(open('lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json'))
    cat_json_stff = copy.deepcopy(cat_json)

    cat_idx_mapping = {}
    for idx, k in enumerate(cat_json):
        cat_idx_mapping[k['id']] = idx

    for k in range(20):
        cat_json_stff[k]['id'] = k

    json.dump(cat_json_stff, open('data/coco/annotations/panoptic_scannet_categories_stff.json', 'w'))

    for s in ['train', 'val']:

        #pano_json = json.load(open('data/coco/annotations/panoptic_{}2025.json'.format(s)))
        pano_json = json.load(open('data/coco/annotations/panoptic_{}2021.json'.format(s)))

        pano_json_stff = copy.deepcopy(pano_json)

        pano_json_stff['categories'] = cat_json_stff

        for anno in pano_json_stff['annotations']:
            for segments_info in anno['segments_info']:
                segments_info['category_id'] = cat_idx_mapping[segments_info['category_id']]


        for img in pano_json_stff['images']:
            # string.replace(oldvalue, newvalue, count)
            img['file_name'] = img['file_name'].replace('png', 'jpg')
        if s == 'val':
            pano_json_stff['images'] = sorted(pano_json_stff['images'], key=lambda x: x['id'])
        
        #json.dump(pano_json_stff, open('data/coco/annotations/panoptic_{}2025_stff.json'.format(s), 'w'))
        json.dump(pano_json_stff, open('data/coco/annotations/panoptic_{}2021_stff.json'.format(s), 'w'))


    for s in ['train', 'val']:
        pano_json = json.load(open('data/coco/annotations/panoptic_{}2021.json'.format(s)))
        pano_json_2 = copy.deepcopy(pano_json)
        for img in pano_json_2['images']:
            img['file_name'] = img['file_name'].replace('png', 'jpg')
        json.dump(pano_json_2, open('data/coco/annotations/panoptic_{}2021_jpg.json'.format(s), 'w'))




