import json
import copy


if __name__ == "__main__":

    cat_json = json.load(open('lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json'))
    cat_json_stff = copy.deepcopy(cat_json)
    json.dump(cat_json_stff, open('data/coco/annotations/panoptic_scannet_categories_stff.json', 'w'))

    for s in ['train', 'val']:

        pano_json = json.load(open('data/coco/annotations/panoptic_{}2021.json'.format(s)))

        pano_json_stff = copy.deepcopy(pano_json)

        pano_json_stff['categories'] = cat_json_stff

        #for img in pano_json_stff['images']:
        #     img['file_name'] = img['file_name'].replace('jpg', 'png')
        if s == 'val':
            pano_json_stff['images'] = sorted(pano_json_stff['images'], key=lambda x: x['id'])
        
        json.dump(pano_json_stff, open('data/coco/annotations/panoptic_{}2021_stff.json'.format(s), 'w'))


