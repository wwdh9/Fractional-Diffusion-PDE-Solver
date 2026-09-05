import os
import shutil


ROOT = os.path.dirname(os.path.abspath(__file__))


folders = {
    "example1": "ex1",
    "example2": "ex2",
    "example3": "ex3",
}


for folder_key, prefix in folders.items():

    folder_path = None

    for name in os.listdir(ROOT):
        if name.endswith(folder_key):
            folder_path = os.path.join(ROOT, name)
            break


    if folder_path is None:
        print(f"❌ 找不到 {folder_key}")
        continue


    print("\n处理目录:")
    print(folder_path)


    for filename in os.listdir(folder_path):

        # 只处理图片
        if not filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue


        src = os.path.join(
            folder_path,
            filename
        )


        new_name = (
            prefix
            + "_"
            + filename
        )


        dst = os.path.join(
            ROOT,
            new_name
        )


        shutil.copy2(
            src,
            dst
        )


        print(
            filename,
            " --> ",
            new_name
        )


print("\n✅ 所有图片处理完成")