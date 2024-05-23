import fiftyone

rename = {
    'Dining table': ['Kitchen & dining room table'],
    'Cell phone': ['Mobile phone'],
    'Cup': ['Coffee cup'],
    'Dish': ['Plate'],
    'Electric_fan': ['Mechanical fan'],
    'Faucet': ['Tap'],
    'Frisbee': ['Flying disc'],
    'Melon': ['Watermelon'],
    'Microwave': ['Microwave oven'],
    'Skis': ['Ski'],
    'Sports ball': ['Cricket ball', 'Tennis ball', 'Volleyball (Ball)'],
    'Screen/monitor': ['Computer monitor'],
    'Child': ['Boy', 'Girl'],
    'Rat': 'Mouse'
          }

need = [i.capitalize() for i in ['airplane', 'antelope', 'apple', 'baby', 'baby_seat', 'baby_walker', 'backpack', 'banana', 'baseball bat', 'bear', 'beetroot', 'bench', 'bicycle', 'bird', 'boat', 'bottle', 'bread', 'bus', 'bus', 'truck', 'cake', 'camel', 'camera', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'chicken', 'child', 'coconut', 'couch', 'cow', 'crab', 'crocodile', 'cup', 'dining table', 'dish', 'dog', 'duck', 'electric_fan', 'elephant', 'faucet', 'fish', 'frisbee', 'fruits', 'guava', 'guitar', 'hamster', 'rat', 'handbag', 'horse', 'kangaroo', 'laptop', 'lemon', 'leopard', 'lettuce', 'lion', 'melon', 'microwave', 'motorcycle', 'orange', 'oven', 'panda', 'penguin', 'person', 'piano', 'pig', 'pumpkin', 'quadbike', 'rabbit', 'rambutan', 'refrigerator', 'scooter', 'screen/monitor', 'sheep', 'sink', 'skateboard', 'skis', 'snake', 'snowboard', 'sports ball', 'squirrel', 'stingray', 'stool', 'stop sign', 'suitcase', 'surfboard', 'tennis racket', 'tiger', 'toilet', 'toy', 'tractor', 'traffic light', 'train', 'truck', 'turtle', 'van', 'vegetables']]
need_f = [rename[i] if i in rename else i for i in need]
new_list = []

for item in need_f:
    if isinstance(item, list):
        new_list.extend(item)
    else:
        new_list.append(item)


rename = {
    'Bear': ['Bear', 'Brown bear'],
    'Domestic_cat': ['Cat'],
    'Frisbee': ['Flying disc'],
    'Giant_panda': ['Panda'],
    'Red_panda': ['Red panda'],
    'Sofa': ['Couch'],
    'Turtle': ['Turtle', 'Sea turtle'],
    }

need = [i.capitalize() for i in ['airplane', 'antelope', 'ball', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'frisbee', 'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'person', 'rabbit', 'red_panda', 'sheep', 'skateboard', 'snake', 'sofa', 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra']]
need_f = [rename[i][0] if i in rename else i for i in need]
dataset = fiftyone.zoo.load_zoo_dataset("open-images-v6", split="train", label_types=["detections"], classes = need_f)