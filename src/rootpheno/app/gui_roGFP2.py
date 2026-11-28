import re
import time

import numpy as np
import pandas as pd
import tkinter as tk
import multiprocessing as mp

from PIL import Image
from pathlib import Path
from datetime import datetime
from tkinter import filedialog

from rootpheno.imread.CziReader import CziReader
from rootpheno.app.utils import resource_path
from rootpheno.biosensor.utils import gray_to_lut, gray_to_rgb
from rootpheno.biosensor.cyto_roGFP2_GRX import cyto_roGFP2_GRX
from rootpheno.stitcher.stitch_pair import stitch_pair


use_cores = max(1, mp.cpu_count() - 1)
date_time = datetime.now().strftime("%Y%m%d-%H%M%S")


def get_folder_path() -> None:
    input_folder_dir.set("")
    input_message.set("")
    done_message.set("")
    timer_message.set("")
    output_folder_message.set("")
    _input_folder_path = filedialog.askdirectory()
    _input_folder_path = Path(_input_folder_path).expanduser().resolve()
    input_folder_dir.set(_input_folder_path.as_posix())
    _czi_path_list = [i for i in _input_folder_path.iterdir() if i.suffix.lower() == ".czi"]
    n_img.set(len(_czi_path_list))
    metadata_dir.set((_input_folder_path / "metadata.csv").as_posix())
    if not Path(metadata_dir.get()).exists():
        m = f'Please check the <grouping> and <ordering> columns in "metadata.csv"'
        input_message.set(m)
        generate_grouping_csv(_czi_path_list)
    else:
        csv = pd.read_csv(metadata_dir.get())
        if csv["grouping"].isna().any():
            input_message.set("grouping error: <grouping> column in 'metadata.csv'")
        else:
            input_message.set(f"CZI images found: {len(_czi_path_list)}")
        if (not "ordering" in csv.columns) or csv["ordering"].isna().any():
            csv["img_path"] = csv.dirname + "/" + csv.basename
            csv = csv.drop(columns='ordering')
            ordering = guess_ordering(csv)
            csv = csv.merge(ordering, on="img_path", how='left')
            csv = csv.drop(columns="img_path")
        csv.to_csv(f"{input_folder_dir.get()}/metadata.csv", index=False)


def generate_grouping_csv(_czi_path_list: list[Path]) -> None:
    dirname = [i.parent.as_posix() for i in _czi_path_list]
    basename = [i.name for i in _czi_path_list]
    filename = [i.stem for i in _czi_path_list]
    grouping = guess_grouping(filename)
    csv = pd.DataFrame({
        "dirname": dirname,
        "basename": basename,
        "grouping": grouping
    })
    csv["img_path"] = csv.dirname + "/" + csv.basename
    ordering = guess_ordering(csv)
    csv = csv.merge(ordering, on="img_path", how='left')
    csv = csv.drop(columns="img_path")
    csv.to_csv(metadata_dir.get(), index=False)


def guess_grouping(filename: list[str]) -> list[str | None]:
    if not isinstance(filename, list):
        filename = list(filename)
    sfx = ["tif", "tiff", "jpg", "jpeg", "png", "bmp", "czi"]
    pattern = rf"(?:\.(?:{'|'.join(sfx)}))$"
    regex = re.compile(pattern, flags=re.IGNORECASE)
    filename = [regex.sub("", f) for f in filename]
    grouping = []
    for f in filename:
        parts = f.replace("_", "-").split("-")
        if len(parts) >= 2:
            grouping.append("_".join(parts[:-1]))
        else:
            grouping.append("_".join(parts))
    label_to_id: dict = {}
    group_ids: list[int] = []
    for label in grouping:
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id) + 1
        group_ids.append(label_to_id[label])
    return group_ids


def guess_ordering(csv: pd.DataFrame) -> pd.DataFrame:
    df_all = csv.sort_values("grouping")
    df_all = df_all[df_all["grouping"].notna()].copy()
    ordering = {"img_path": [], "ordering": []}
    for _, group_df in df_all.groupby("grouping", sort=False):
        group_paths: list[str] = []
        positions: list[float] = []
        for row in group_df.itertuples(index=False):
            full_path = row.dirname + "/" + row.basename
            czi = CziReader(full_path)
            stage_position = czi.get_stage_position() or {}
            x, y, z = stage_position.values()
            pos = -999 if x is None or y is None else -x - y
            positions.append(pos)
            group_paths.append(full_path)
        sorted_pos = sorted(positions)
        pos_order = [sorted_pos.index(val) + 1 for val in positions]
        ordering["img_path"].extend(group_paths)
        ordering["ordering"].extend(pos_order)
    return pd.DataFrame(ordering)


def try_extract_pi(arr: np.ndarray):
    arr = arr.squeeze()
    if arr.ndim == 2:
        pi = arr
    elif arr.ndim == 3:
        pi = np.take(arr, 2, 2)
    else:
        pi = np.zeros(arr.shape)
    pi = np.moveaxis(pi, 0, 1)
    return pi


def compute_local():
    start_time = time.perf_counter()
    done_message.set("")
    timer_message.set("")
    output_folder_message.set("")
    input_folder = Path(input_folder_dir.get())
    output_folder = input_folder.parent / ("OUT_" + input_folder.stem)
    output_folder_dir.set(output_folder.as_posix())
    output_folder = output_folder / "pre_mosaic"
    subfolders = []
    for s in ['405 nm', '488 nm', 'PI', 'RO', 'RO_LUT']:
        sub = output_folder / s
        sub.mkdir(parents=True, exist_ok=True)
        subfolders.append(sub.as_posix())
    csv = pd.read_csv(metadata_dir.get())
    img_list = (csv.dirname + "/" + csv.basename).tolist()
    for img_dir in img_list:
        czi = CziReader(img_dir)
        rgbtiff = czi.tiff_info
        graytiff = rgbtiff
        graytiff[262] = 1
        tiff_name = czi.filename + ".tif"
        arr = czi.arr
        e405 = np.take(arr, 0, 2)  # 1st layer from C <- (X, Y, C, Z, T, R, S, I, B, M, H)
        e488 = np.take(arr, 1, 2)  # 2nd layer from C <- (X, Y, C, Z, T, R, S, I, B, M, H)
        pi = np.take(arr, 2, 2)  # 3rd layer from C <- (X, Y, C, Z, T, R, S, I, B, M, H)
        e405 = np.moveaxis(e405.squeeze(), 0, 1)  # (X, Y) -> (Y, X)
        e488 = np.moveaxis(e488.squeeze(), 0, 1)  # (X, Y) -> (Y, X)
        pi = np.moveaxis(pi.squeeze(), 0, 1)  # (X, Y) -> (Y, X)
        ro, lut = cyto_roGFP2_GRX(e405, e488)

        e405_output = subfolders[0] + f"/{tiff_name}"
        e405_img = Image.fromarray(e405)
        e405_img.save(e405_output, format="tiff", tiffinfo = graytiff)

        e488_output = subfolders[1] + f"/{tiff_name}"
        e488_img = Image.fromarray(e488)
        e488_img.save(e488_output, format="tiff", tiffinfo = graytiff)

        pi_output = subfolders[2] + f"/{tiff_name}"
        pi_img = Image.fromarray(gray_to_rgb(pi))
        pi_img.save(pi_output, format="tiff", tiffinfo = rgbtiff)
        
        ro_output = subfolders[3] + f"/{tiff_name}"
        ro_img = Image.fromarray(ro)
        ro_img.save(ro_output, format="tiff", tiffinfo = graytiff)

        lut_output = subfolders[4] + f"/{tiff_name}"
        lut_img = Image.fromarray(lut)
        lut_img.save(lut_output, format="tiff", tiffinfo = rgbtiff)

    # Stitching pre-mosaic images
    output_folder = output_folder.with_name("mosaic")
    subfolders = []
    for s in ['405 nm', '488 nm', 'PI', 'RO', 'RO_LUT']:
        sub = output_folder / s
        sub.mkdir(parents=True, exist_ok=True)
        subfolders.append(sub.as_posix())

    for g in np.unique(csv.grouping):
        df = csv[csv["grouping"] == g]
        df = df.sort_values("ordering")
        czi_list = (df.dirname + "/" + df.basename).tolist()
        if czi_list is None or len(czi_list) < 1:
            continue
        czi = CziReader(czi_list[0])
        arr = czi.arr
        e405 = np.moveaxis(arr.take(0, 2).squeeze(), 0, 1)
        e488 = np.moveaxis(arr.take(1, 2).squeeze(), 0, 1)
        pi = np.moveaxis(arr.take(2, 2).squeeze(), 0, 1)
        ro, lut = cyto_roGFP2_GRX(e405, e488)
        stitch_success = len(czi_list)
        for i in range(1, len(czi_list)):
            czi = CziReader(czi_list[i])
            tiff_name = czi.filename + ".tif"
            arr = czi.arr
            try:
                _e405 = np.moveaxis(arr.take(0, 2).squeeze(), 0, 1)
                _e488 = np.moveaxis(arr.take(1, 2).squeeze(), 0, 1)
                _ro, _lut = cyto_roGFP2_GRX(_e405, _e488)
                _pi = np.moveaxis(arr.take(2, 2).squeeze(), 0, 1)
                pi, transform = stitch_pair(pi, _pi)
                e405, _ = stitch_pair(e405, _e405, transform)
                e488, _ = stitch_pair(e488, _e488, transform)
                ro, _ = stitch_pair(ro, _ro, transform)
            except:
                stitch_success -= 1
                continue


        if stitch_success > 1:
            lut = gray_to_lut(ro)
            rgbtiff = czi.tiff_info
            height, width = pi.shape
            rgbtiff[256] = width
            rgbtiff[257] = height
            graytiff = rgbtiff
            graytiff[262] = 1

            e405_output = subfolders[0] + f"/{tiff_name}"
            e405_img = Image.fromarray(e405)
            e405_img.save(e405_output, format="tiff", tiffinfo = graytiff)

            e488_output = subfolders[1] + f"/{tiff_name}"
            e488_img = Image.fromarray(e488)
            e488_img.save(e488_output, format="tiff", tiffinfo = graytiff)

            pi_output = subfolders[2] + f"/{tiff_name}"
            pi_img = Image.fromarray(gray_to_rgb(pi))
            pi_img.save(pi_output, format="tiff", tiffinfo = rgbtiff)
            
            ro_output = subfolders[3] + f"/{tiff_name}"
            ro_img = Image.fromarray(ro)
            ro_img.save(ro_output, format="tiff", tiffinfo = graytiff)

            lut_output = subfolders[4] + f"/{tiff_name}"
            lut_img = Image.fromarray(lut)
            lut_img.save(lut_output, format="tiff", tiffinfo = rgbtiff)

    end_time = time.perf_counter()
    ct = round(end_time - start_time, 2)  # consumed time
    ips = round(n_img.get() / ct, 2)  # img per sec

    timer_message.set(f"---------- {ct} secs [{ips} img/sec] ----------")
    done_message.set("Local RO-index completed !!!")
    output_folder_message.set(f"Images save to:\n{output_folder_dir.get()}")


def compute_global():
    start_time = time.perf_counter()
    done_message.set("")
    timer_message.set("")
    output_folder_message.set("")
    input_folder = Path(input_folder_dir.get())
    output_folder = input_folder.parent / ("OUT_" + input_folder.stem)
    output_folder_dir.set(output_folder.as_posix())
    output_folder = output_folder / "mosaic"
    subfolders = []
    for s in ['405 nm', '488 nm', 'PI', 'RO', 'RO_LUT']:
        sub = output_folder / s
        sub.mkdir(parents=True, exist_ok=True)
        subfolders.append(sub.as_posix())
    
    csv = pd.read_csv(metadata_dir.get())
    for g in np.unique(csv.grouping):
        df = csv[csv["grouping"] == g]
        df = df.sort_values("ordering")
        czi_list = (df.dirname + "/" + df.basename).tolist()
        czi = CziReader(czi_list[0])
        arr = czi.arr
        e405 = np.moveaxis(arr.take(0, 2).squeeze(), 0, 1)
        e488 = np.moveaxis(arr.take(1, 2).squeeze(), 0, 1)
        pi = np.moveaxis(arr.take(2, 2).squeeze(), 0, 1)
        stitch_success = len(czi_list)
        for i in range(1, len(czi_list)):
            czi = CziReader(czi_list[i])
            tiff_name = czi.filename + ".tif"
            arr = czi.arr
            try:
                _e405 = np.moveaxis(arr.take(0, 2).squeeze(), 0, 1)
                _e488 = np.moveaxis(arr.take(1, 2).squeeze(), 0, 1)
                _pi = np.moveaxis(arr.take(2, 2).squeeze(), 0, 1)
                pi, transform = stitch_pair(pi, _pi)
                e405, _ = stitch_pair(e405, _e405, transform)
                e488, _ = stitch_pair(e488, _e488, transform)
            except:
                stitch_success -= 1
                continue

        ro, lut = cyto_roGFP2_GRX(e405, e488)

        if stitch_success > 1:
            rgbtiff = czi.tiff_info
            height, width = pi.shape
            rgbtiff[256] = width
            rgbtiff[257] = height
            graytiff = rgbtiff
            graytiff[262] = 1

            e405_output = subfolders[0] + f"/{tiff_name}"
            e405_img = Image.fromarray(e405)
            e405_img.save(e405_output, format="tiff", tiffinfo = graytiff)

            e488_output = subfolders[1] + f"/{tiff_name}"
            e488_img = Image.fromarray(e488)
            e488_img.save(e488_output, format="tiff", tiffinfo = graytiff)

            pi_output = subfolders[2] + f"/{tiff_name}"
            pi_img = Image.fromarray(gray_to_rgb(pi))
            pi_img.save(pi_output, format="tiff", tiffinfo = rgbtiff)
            
            ro_output = subfolders[3] + f"/{tiff_name}"
            ro_img = Image.fromarray(ro)
            ro_img.save(ro_output, format="tiff", tiffinfo = graytiff)

            lut_output = subfolders[4] + f"/{tiff_name}"
            lut_img = Image.fromarray(lut)
            lut_img.save(lut_output, format="tiff", tiffinfo = rgbtiff)

    end_time = time.perf_counter()
    ct = round(end_time - start_time, 2)  # consumed time
    ips = round(n_img.get() / ct, 2)  # img per sec

    timer_message.set(f"---------- {ct} secs [{ips} img/sec] ----------")
    done_message.set("Global RO-index completed !!!")
    output_folder_message.set(f"Images save to:\n{output_folder_dir.get()}")


def compute_pi():
    start_time = time.perf_counter()
    done_message.set("")
    timer_message.set("")
    output_folder_message.set("")
    input_folder = Path(input_folder_dir.get())
    output_folder = input_folder.parent / ("OUT_" + input_folder.stem)
    output_folder_dir.set(output_folder.as_posix())
    output_folder = output_folder / "pre_mosaic"
    output_folder.mkdir(parents=True, exist_ok=True)
    csv = pd.read_csv(metadata_dir.get())
    img_list = (csv.dirname + "/" + csv.basename).tolist()
    for img_dir in img_list:
        czi = CziReader(img_dir)
        tiff_name = czi.filename + ".tif"
        pi = try_extract_pi(czi.arr)
        pi_output = output_folder.as_posix() + f"/{tiff_name}"
        pi_img = Image.fromarray(gray_to_rgb(pi))
        pi_img.save(pi_output, format="tiff", tiffinfo = czi.tiff_info)
        
    # Stitching pre-mosaic images
    output_folder = output_folder.with_name("mosaic")
    output_folder.mkdir(parents=True, exist_ok=True)
    for g in np.unique(csv.grouping):
        df = csv[csv["grouping"] == g]
        df = df.sort_values("ordering")
        czi_list = (df.dirname + "/" + df.basename).tolist()
        if czi_list is None or len(czi_list) < 1:
            continue
        czi = CziReader(czi_list[0])
        pi = try_extract_pi(czi.arr)

        stitch_success = len(czi_list)
        for i in range(1, len(czi_list)):
            czi = CziReader(czi_list[i])
            tiff_name = czi.filename + ".tif"
            try:
                _pi = try_extract_pi(czi.arr)
                pi, transform = stitch_pair(pi, _pi)
            except:
                stitch_success -= 1
                continue

        if stitch_success > 1:
            rgbtiff = czi.tiff_info
            height, width = pi.shape
            rgbtiff[256] = width
            rgbtiff[257] = height

            pi_output = output_folder.as_posix() + f"/{tiff_name}"
            pi_img = Image.fromarray(gray_to_rgb(pi))
            pi_img.save(pi_output, format="tiff", tiffinfo = rgbtiff)

    end_time = time.perf_counter()
    ct = round(end_time - start_time, 2)  # consumed time
    ips = round(n_img.get() / ct, 2)      # img per sec

    timer_message.set(f"---------- {ct} secs [{ips} img/sec] ----------")
    done_message.set("PI mosaic completed !!!")
    output_folder_message.set(f"Images save to:\n{output_folder_dir.get()}")
 

if __name__ == "__main__":
    mp.freeze_support()
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Window
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    app = tk.Tk()
    app.iconbitmap(resource_path("roGFP2_icon.ico"))
    app.title("roGFP2")
    app.geometry("600x500")

    title_label = tk.Label(
        master=app, 
        text="roGFP2", 
        font="Calibri 21 bold"
    ).pack(pady=5)

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Folder path and Metadata
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    input_folder_dir = tk.StringVar()
    output_folder_dir = tk.StringVar()
    grouping_csv = tk.StringVar()
    input_message = tk.StringVar()
    metadata_dir = tk.StringVar()
    n_img = tk.IntVar()

    choose_folder_frame = tk.Frame(master=app)

    tk.Button(
        master=choose_folder_frame,
        text="Select folder",
        font="Calibri 16",
        relief="ridge",
        command=get_folder_path
    ).pack(padx=20, pady=20, fill='x')

    tk.Label(
        master=choose_folder_frame,
        text="Input directory",
        textvariable=input_folder_dir,
        font="Calibri 13"
    ).pack(padx=20, pady=5, fill='x')

    tk.Label(
        master=choose_folder_frame,
        text="CZI file",
        textvariable=input_message,
        font="Calibri 13"
    ).pack(padx=20, pady=5, fill='x')

    choose_folder_frame.pack(padx=20, fill='x')

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Compute
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    compute_frame = tk.Frame(master=app)

    compute_frame.columnconfigure(0, weight=1)
    compute_frame.columnconfigure(1, weight=1)

    # ----- PI stain -----
    tk.Button(
        master=compute_frame,
        text="PI stain mosaic",
        font="Calibri 16",
        relief="ridge",
        command=compute_pi
    ).grid(row=0, column=0, columnspan=2, sticky='nsew', padx=20, pady=10)

    # ----- Local -----
    tk.Button(
        master=compute_frame,
        text="Run local",
        font="Calibri 16",
        relief="ridge",
        command=compute_local
    ).grid(row=1, column=0, sticky='nsew', padx=20, pady=5)

    # ----- Global -----
    tk.Button(
        master=compute_frame,
        text="Run global",
        font="Calibri 16",
        relief="ridge",
        command=compute_global
    ).grid(row=1, column=1, sticky='nsew', padx=20, pady=5)


    compute_frame.pack(padx=20, pady=5, fill='x')
    
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Done Message
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    done_message = tk.StringVar()
    timer_message = tk.StringVar()
    output_folder_message = tk.StringVar()

    done_message_frame = tk.Frame(master=app)

    tk.Label(
        master=done_message_frame,
        text="Timer message",
        textvariable=timer_message,
        font="Calibri 13"
    ).pack(padx=20, pady=5, fill='x')

    tk.Label(
        master=done_message_frame,
        text="Done message",
        textvariable=done_message,
        font="Calibri 15 bold"
    ).pack(padx=20, pady=5, fill='x')

    tk.Label(
        master=done_message_frame,
        text="Output folder",
        textvariable=output_folder_message,
        font="Calibri 13"
    ).pack(padx=20, pady=5, fill='x')
    
    done_message_frame.pack(padx=20, pady=5, fill='x')

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Execute
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    app.mainloop()



