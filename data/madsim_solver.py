import os
import json


class VisASolver(object):
    CLSNAMES = [
        # 'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        # 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        # 'pcb4', 'pipe_fryum',
#         01Gorilla/  03Mallard/  05Whale/  07Owl/         09Swan/   11Pig/     13Pheonix/   15Parrot/  17Scorpion/   19Bear/
# 02Unicorn/  04Turtle/   06Bird/   08Sabertooth/  10Sheep/  12Zalika/  14Elephant/  16Cat/     18Obesobeso/  20Puppy/
        '01Gorilla', '03Mallard', '05Whale', '07Owl', '09Swan', '11Pig', '13Pheonix', '15Parrot', '17Scorpion', '19Bear',
        '02Unicorn', '04Turtle', '06Bird', '08Sabertooth', '10Sheep', '12Zalika', '14Elephant', '16Cat', '18Obesobeso', '20Puppy',
    ]

    def __init__(self, root='./madsim'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

if __name__ == '__main__':
    runner = VisASolver(root='./data/madsim')
    runner.run()