import os, random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import VidVRD
import cv2, shutil
from collections import defaultdict

dataset = VidVRD("~/vidvrd-dataset/", "~/vidvrd-dataset/videos", ["train", "test"])
index = dataset.get_index("train")

alt_name = {"adult": "person", "aircraft": "airplane"}

avail = set(os.listdir("~/dataset/vidvrd_queries/"))
need = set()

for i in tqdm(range(len(index))):
    ind = index[i]

    gt = dataset.get_object_insts(ind)

    for object_num in range(len(gt)):
        object_name = gt[object_num]["category"]

        if object_name in alt_name:
            object_name = alt_name[object_name].strip()

        object_name = object_name.capitalize()
        need.add(object_name)

query_dict = {
    "Frisbee": [
        "0a04224ac43b6290.jpg",
        "0de7c2138bd40d56.jpg",
        "0ec6cb327f1840c3.jpg",
        "0f437e8d34cd041d.jpg",
        "1e2f8995f1c9eac4.jpg",
        "3f099dd87da9a436.jpg",
        "4a509a792da2df5d.jpg",
        "4dfb589c0cb53cda.jpg",
        "05fc158105531b11.jpg",
        "9b4b1ab97d5b763a.jpg",
    ],
    "Tiger": [
        "0a5c4d09b15551a1.jpg",
        "0a9142a15c14e2d0.jpg",
        "0acc21bf636785e8.jpg",
        "0adc2b976e2e8fee.jpg",
        "0b0dec096146999f.jpg",
        "0b88a2d2fc990251.jpg",
        "0be8f77f1614556e.jpg",
        "0bef987277f0189b.jpg",
        "0c9c492e0a9c88c6.jpg",
        "0c458b86aee92fb7.jpg",
    ],
    "Skateboard": [
        "0a0038cd91556265.jpg",
        "0a71b1924e128077.jpg",
        "0aeee72c1ca662cd.jpg",
        "0b4c4f08117fabf7.jpg",
        "0b9d266ea9e32fcc.jpg",
        "0d1b990b9be8b023.jpg",
        "0dd41b298cb0e17d.jpg",
        "0f5f8c406b5c553a.jpg",
        "1a57f73efa8f6933.jpg",
        "1e4d37e9ed50c8db.jpg",
    ],
    "Hamster": [
        "0a1f5557a993fbad.jpg",
        "0c1b4aa584f6fa21.jpg",
        "00e712724ce9239f.jpg",
        "0ec3abd036c764fd.jpg",
        "0ee8c3691a386e81.jpg",
        "0ffb89ead893e264.jpg",
        "1c141a09855fc91b.jpg",
        "001e1cf3abbee3cd.jpg",
        "2a32162c146c6dc1.jpg",
        "2abb8abaa100b05a.jpg",
    ],
    "Turtle": [
        "0d69b443f346c5e3.jpg",
        "0e9b1d776d03bd5f.jpg",
        "01db270c6d7b7717.jpg",
        "2c99d34afbe645ba.jpg",
        "3b83ff95d20ba828.jpg",
        "05d46833bfb6e36d.jpg",
        "6a21c2cc85c9c4c9.jpg",
        "7fa078ca59497f76.jpg",
        "8fe116d6e752526e.jpg",
        "10bc2f4354cad463.jpg",
    ],
}

unseen_classes = set(["Frisbee", "Hamster", "Skateboard", "Tiger", "Turtle"])

gtbboxid = 0
track = []

unseen_vids = set()

for i in tqdm(range(len(index))):
    ind = index[i]

    gt = dataset.get_object_insts(ind)

    for object_num in range(len(gt)):
        object_name = gt[object_num]["category"]

        if object_name in alt_name:
            object_name = alt_name[object_name].strip()

        object_name = object_name.capitalize()
        if object_name in unseen_classes:
            unseen_vids.add(ind)

videos_path = "~/all_frames_vidvrd"
queries_path = "~/vidvrd_queries/"

src_target_dir = "./vidvrd_unseen/target"
src_query_dir = "./vidvrd_unseen/query"

Path(src_target_dir).mkdir(parents=True, exist_ok=True)

for vid in unseen_vids:
    for frame in os.listdir(os.path.join(videos_path, vid)):
        f_path = os.path.join(videos_path, vid, frame)
        shutil.copy(f_path, os.path.join(src_target_dir, str(vid) + "_" + frame))

Path(src_query_dir).mkdir(parents=True, exist_ok=True)

for class_name, examples in query_dict.items():
    for example in examples:
        dest_path = os.path.join(src_query_dir, example)
        src_path = os.path.join(queries_path, class_name, example)
        shutil.copy(src_path, dest_path)

gtbboxid = 0
track = []

root_target_dir = "~/dataset/all_frames_vidvrd/"
root_query_dir = "~/dataset/vidvrd_queries"

dest_target_dir = "~/dataset/vid_processed/target/"
dest_query_dir = "~/dataset/vid_processed/queries/images"

from pathlib import Path

Path(dest_target_dir).mkdir(parents=True, exist_ok=True)
Path(dest_query_dir).mkdir(parents=True, exist_ok=True)

seen_classes = set(os.listdir("~/dataset/vidvrd_queries"))

for i in tqdm(range(len(index))):
    val_flag = False
    ind = index[i]

    if ind in unseen_vids:
        val_flag = True

    gt = dataset.get_object_insts(ind)
    vid_path = dataset.get_video_path(ind)
    vid = vid_path.split("/")[-1][:-4]
    frame = cv2.imread(os.path.join(root_target_dir, ind, "00001.jpg"))
    h, w, _ = frame.shape

    for object_num in range(len(gt)):
        object_name = gt[object_num]["category"]

        if object_name in alt_name:
            object_name = alt_name[object_name].strip()

        object_name = object_name.capitalize()

        if object_name not in seen_classes:
            continue

        if not val_flag:
            #         if object_name in seen_classes and not val_flag:

            mode = "train"

            query_path = os.path.join(root_query_dir, object_name)
            skipper = 0

            for frame_num in gt[object_num]["trajectory"]:
                frame_num = int(frame_num)
                target_name = os.path.join(ind, str(frame_num + 1).zfill(5) + ".jpg")

                if skipper % 50 == 0 and os.path.exists(
                    os.path.join(root_target_dir, target_name)
                ):
                    shutil.copy(
                        os.path.join(root_target_dir, target_name),
                        os.path.join(
                            dest_target_dir, ind + "_" + str(frame_num) + ".jpg"
                        ),
                    )
                    lx, ty, rx, by = gt[object_num]["trajectory"][str(frame_num)]

                    query_name = random.sample(os.listdir(query_path), 1)[0]

                    row = ",".join(
                        [
                            str(gtbboxid),
                            query_name[:-4],
                            ind + "_" + str(frame_num),
                            str(lx / w),
                            str(ty / h),
                            str(rx / w),
                            str(by / h),
                            "0",
                            mode,
                        ]
                    )
                    track.append(row.split(","))
                    gtbboxid += 1

                skipper += 1
        else:
            #             if object_name not in unseen_classes:
            #                 continue

            mode = "val-new-cl"
            query_path = os.path.join(root_query_dir, object_name)
            skipper = 0

            for frame_num in gt[object_num]["trajectory"]:
                frame_num = int(frame_num)

                target_name = os.path.join(ind, str(frame_num + 1).zfill(5) + ".jpg")
                if skipper % 50 == 0 and os.path.exists(
                    os.path.join(root_target_dir, target_name)
                ):
                    shutil.copy(
                        os.path.join(root_target_dir, target_name),
                        os.path.join(
                            dest_target_dir, ind + "_" + str(frame_num) + ".jpg"
                        ),
                    )
                    lx, ty, rx, by = gt[object_num]["trajectory"][str(frame_num)]

                    query_name = random.sample(os.listdir(query_path), 1)[0]

                    row = ",".join(
                        [
                            str(gtbboxid),
                            query_name[:-4],
                            ind + "_" + str(frame_num),
                            str(lx / w),
                            str(ty / h),
                            str(rx / w),
                            str(by / h),
                            "0",
                            mode,
                        ]
                    )
                    track.append(row.split(","))
                    gtbboxid += 1

                skipper += 1

df = pd.DataFrame(track)
df.to_csv(
    f"~/dataset/vid_processed/classes/unseen.csv",
    index=False,
    header=[
        "primary_key",
        "query",
        "target",
        "lx",
        "ty",
        "rx",
        "by",
        "difficult",
        "split",
    ],
)
