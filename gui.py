import tkinter as tk
from tkinter import filedialog, font, ttk
from PIL import Image, ImageTk

import os
import cv2
from script.predict_for_gui import sahi_prediction, yolo_prediction
import numpy as np
import shutil
from script.utils import draw_gt
import numpy as np


cap = None
cap_p = None
img_originale = None
cache_gt = None
img_for_pred = None
is_video = False
is_paused = False
path = ""
video_loop_id = None
is_not_stopped = True
is_not_cancel = True
text_for_pred= ""
gt = None
update_gt = False
GT_FOLDER = "../test/labels"

PATH_TO_YOLO12S_FT = "script/gui_utils/weights/yolo12s.pt"
def ridimensiona_immagine(event=None):
    global img_originale, img_for_pred, text_for_pred, update_gt, cache_gt, gt
    if img_originale:
        altezza = frame_origin.winfo_height()-30 
        larghezza_originale, altezza_originale = img_originale.size        
        scala = altezza / altezza_originale    
        larghezza = int(larghezza_originale * scala)    
        label_size.config(text=f"Dimensione {larghezza_originale}x{altezza_originale}")

        img = img_originale.copy()
        
        if gt is None :
            cache_gt = img_gt = img_originale.copy()
        img.thumbnail((larghezza, altezza), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        
        label_o.config(text="Originale")
        label_gt.config(text="Ground Truth non presente")

        if gt is not None:
            label_gt.config(text="Ground Truth")

        if  gt is not None and update_gt:
            update_gt = False
            img_gt_a = draw_gt(img_originale.copy(), gt)
            cache_gt = Image.fromarray(img_gt_a)
            
                  
        cache_to_draw = cache_gt.copy()
        cache_to_draw.thumbnail((larghezza, altezza), Image.Resampling.LANCZOS)
        img_gt_tk = ImageTk.PhotoImage(cache_to_draw)
        img_width, img_height = img.size
        canvas_width = canvas_original.winfo_width()
        canvas_height = canvas_original.winfo_height()
        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2

        canvas_original.delete("all")
        canvas_gt.delete("all")
        canvas_predict.delete("all")

        canvas_original.create_image(x, y, anchor="nw", image=img_tk)
        canvas_gt.create_image(x,y, anchor="nw", image=img_gt_tk)   
        
        canvas_original.image = img_tk
        canvas_gt.image = img_gt_tk


        if img_for_pred is not None:  
                     
            img_pred = img_for_pred.copy()
            img_pred.thumbnail((larghezza, altezza), Image.Resampling.LANCZOS)
            img_pred_c = ImageTk.PhotoImage(img_pred)
            canvas_width = canvas_predict.winfo_width()
            canvas_height = canvas_predict.winfo_height()
            

            x = (canvas_width - img_width) // 2
            y = (canvas_height - img_height) // 2
            canvas_predict.create_image(x, y, anchor="nw", image=img_pred_c)
            canvas_predict.image = img_pred_c
            label_p.config(text=text_for_pred)

def carica_immagine():
    global cap, img_originale, is_video, pat, gt, update_gt
    filetypes=[
        ("Immagini e video", "*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv"),
        ("Tutti i file", "*.*")
    ]
    path = filedialog.askopenfilename(filetypes=filetypes)
    if path:
        estensione = os.path.splitext(path)[1].lower()
        immagini_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        video_ext = {".mp4", ".avi", ".mov", ".mkv"}
        stop_video_loop()
        update_gt = True
        gt = search_gt(os.path.splitext(os.path.basename(path))[0]+".txt")
        
        if estensione in immagini_ext:
            is_video = False
            if cap is not None:
                cap.release()
                cap = None
            img_originale = Image.open(path)
            
            play.pack_forget()
            mostra_immagine()


        elif estensione in video_ext:
            is_video = True
            play.pack(side="left", padx=10, pady=5)
            play_on_load()
            start_video(path)
        else:
            messagebox.showerror("Errore", "Formato file non supportato.")
            
def start_video(path):
    global cap
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(path)
    play_frame()

def mostra_immagine():
    
    ridimensiona_immagine()
    predict.config(state="active")
    predict.config(bg="green", fg="black")
        
def get_Prediction():
    model_type = opzione_selezionata.get()
    global cap, cap_p,img_originale, is_video, is_not_stopped, is_not_cancel, img_for_pred, text_for_pred
    is_not_stopped = True
    is_not_cancel = True
    upload.pack_forget()
    play.pack_forget()
    predict.pack_forget()
    repeat.pack_forget()
    save_b.pack_forget()
    restart_b.pack_forget()
    text_for_pred=""
    

    if (model_type == "Yolo12n"):
        model_path = "path/to/12n"
    else:
        model_path = PATH_TO_YOLO12S_FT
    
    cache = "script/gui_utils/cache/"
    os.makedirs(cache, exist_ok = True)      
    raw_size = opzione_selezionata_sahi.get()   
    size_w, size_h = map(int, raw_size.split("x")) 
    text_for_pred = f"Predizione con {model_type}"
    if isSahi.get():
        text_for_pred+=f" con Sahi sliced window {raw_size}"
    if is_video:
        progress.pack(pady=20)
        cancel.pack(side = "left", padx=10, pady=5)
        stop.pack(side = "right", padx=10, pady=5)

        cache_video = os.path.join(cache,"cache.mp4")

        

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        altezza, larghezza = frame.shape[:2]
        video_writer = cv2.VideoWriter(cache_video, cv2.VideoWriter_fourcc(*"mp4v"), 24, (larghezza, altezza))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0        
        play_pause_video()
        while ret and is_not_stopped and is_not_cancel:
            cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_frame = None

            if isSahi.get():        
                res_frame = sahi_prediction(model_path = model_path, frame = cvt, slice_height = size_h, slice_width = size_w )  
            else:
                res_frame = yolo_prediction(model_path = model_path,  frame = cvt)                        
           
            img_originale = Image.fromarray(cvt)
            img_for_pred = res_frame
            video_writer.write(cv2.cvtColor(np.array(res_frame),cv2.COLOR_RGB2BGR))
            mostra_immagine()             


            progress_percent = (current_frame / total_frames) * 100
            progress["value"] = progress_percent
            progress.update()

            ret, frame = cap.read()
            current_frame +=1
        video_writer.release()    
        
        if is_not_cancel:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap_p = cv2.VideoCapture(cache_video)
        play_pause_video()
        progress.pack_forget()
        progress["value"] = 0
        
    else:
        cache_img = os.path.join(cache,"cache.jpg")
        
       
        if isSahi.get():                
            res_frame = sahi_prediction(model_path = model_path, frame = np.array(img_originale), slice_height = size_h, slice_width = size_w) 

        else:
            res_frame = yolo_prediction(model_path = model_path, frame = np.array(img_originale))
        img_for_pred = res_frame
        
        img_for_pred.save(cache_img)
        mostra_immagine()
    if is_not_cancel:
        cancel.pack_forget()
        stop.pack_forget()
        save_b.pack(side="left", padx=10, pady=5)
        restart_b.pack(side="right", padx=10, pady=5)
        repeat.pack(side="right", padx=10, pady=5)
        if(is_video):
            play.pack(side="left", padx=10, pady=5)
    return

def play_frame():
    global is_paused, cap, video_loop_id, img_originale, img_for_pred
    if not is_video or cap is None:
        return
    if not is_paused:
        ret, frame = cap.read()
        if cap_p is not None:
            ret_p, frame_p = cap_p.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if cap_p is not None:
                cap_p.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_loop_id = finestra.after(24, play_frame)
            return
        if cap_p is not None:
            if not ret_p:
                cap_p.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_loop_id = finestra.after(24, play_frame)
                return
        cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        img = Image.fromarray(cvt)
        img_originale = img
        if cap_p is not None:
            cvt_p = cv2.cvtColor(frame_p, cv2.COLOR_BGR2RGB)        
            img_p = Image.fromarray(cvt_p)
            img_for_pred = img_p
        mostra_immagine()
    video_loop_id = finestra.after(24, play_frame)

def play_pause_video():
    global is_paused
    if(is_paused):
        is_paused = False
        play.config(text="Pause")
    else:
        is_paused = True
        play.config(text="Play")

def save_pred():
    filetypes = []
    defaultextension= ""
    file_to_save = ""
    if is_video:
        filetypes = [("File video", "*.mp4"), ("All files", "*.*")]
        defaultextension = ".mp4"
        file_to_save = "cache/cache.mp4"
    else :
        filetypes = [("File img", "*.jpg"), ("All files", "*.*")]
        defaultextension = ".jpg"
        file_to_save = "cache/cache.jpg"

    file_path = filedialog.asksaveasfilename(
        defaultextension=defaultextension,
        filetypes=filetypes,
        title="Salva il file come..."
    )
    shutil.copy2(file_to_save, file_path)
    
def nuova_pred():
    global is_paused, img_for_pred
    carica_immagine()
    is_paused = False

    save_b.pack_forget()
    restart_b.pack_forget()
    play.pack_forget()
    repeat.pack_forget()
    img_for_pred = None
    canvas_predict.delete(all)
    label_p.config(text="")
    upload.pack(side="left", padx=10, pady=5)    
    predict.pack(side="right", padx=10, pady=5)
    
    
def stop_video_loop():
    global video_loop_id
    if video_loop_id is not None:
        finestra.after_cancel(video_loop_id)
        video_loop_id = None

def cancel():
    global is_not_cancel, cap_p, img_for_pred
    is_not_cancel = False
    cancel.pack_forget()
    stop.pack_forget()
    upload.pack(side="left", padx=10, pady=5)
    predict.pack(side="right", padx=10, pady=5)
    play.pack(side="left", padx=10, pady=5)
    canvas_predict.delete(all)
    img_for_pred = None
    label_p.config(text = "")

def stop_prediction():
    global is_not_stopped
    is_not_stopped = False
    stop.pack_forget()

def play_on_load():
    global is_paused
    if is_paused:
        is_paused = False
        play.config(text = "Pause")

def search_gt(file_to_search):
    files = []
    if os.path.isdir(GT_FOLDER):
        files = os.listdir(GT_FOLDER)
    content = None
    if file_to_search in files:
        
        full_path = os.path.join(GT_FOLDER, file_to_search)
        with open(full_path, 'r') as f:
            content = f.readlines()

    else:
        print("File non trovato.")
    return content

'''Finestra principale '''
finestra = tk.Tk()
finestra.title("Predizione di immagini")
finestra.bind("<Configure>", ridimensiona_immagine)

finestra.geometry("1920x1080")

''' Frame per il design '''
frame_bottom = tk.Frame(finestra)
frame_bottom.pack(side="bottom", fill="x")
frame_top = tk.Frame(finestra)
frame_top.pack(side="top", fill="x")
frame_undertop = tk.Frame(finestra)
frame_undertop.pack(side="top", fill="x")

contenitore_canvas = tk.Frame(finestra)
contenitore_canvas.pack(fill="both", expand=True)
contenitore_sup = tk.Frame(contenitore_canvas)
contenitore_sup.pack(side = "top", fill="both", expand=True)
frame_origin = tk.Frame(contenitore_sup)
frame_gt = tk.Frame(contenitore_sup)
frame_predit = tk.Frame(contenitore_canvas)
frame_origin.pack(side="left", fill="both", expand=True)
frame_gt.pack(side="left", fill="both", expand=True)
frame_predit.pack(side="bottom", fill="both", expand=True)


''' Menù a tendina per selezionare i pesi con cui fare la prediction '''
opzioni = ["Yolo12n", "Yolo12s"]
opzione_selezionata = tk.StringVar(finestra)
opzione_selezionata.set(opzioni[1])
combo = ttk.Combobox(frame_top, textvariable=opzione_selezionata, values=opzioni, font=("Helvetica", 16), width=20, state="readonly")
combo.current(1)
combo.pack(side="left", pady=20)
''' Menù a tendina per selezionare la dimensione della finestra se si usa sahi nella prediction '''
opzioni_sahi = ["320x320","512x512", "640x640", "800x800", "896x896","1024x1024","1280x1280","1536x1535","2048x2048"]
opzione_selezionata_sahi = tk.StringVar(finestra)
opzione_selezionata_sahi.set(opzioni_sahi[2])
sahi_dim_menu = ttk.Combobox(frame_top, textvariable=opzione_selezionata_sahi, values=opzioni_sahi, font=("Helvetica", 16), width=20, state="readonly")
sahi_dim_menu.current(2)
label_sahi = tk.Label(frame_top, text="Dim fin sahi:", font=("Helvetica", 16))
sahi_dim_menu.pack(side="right", pady=20)
label_sahi.pack(side="right", padx=(20, 10)) 



''' Bottone per l'upload delle immagini '''
upload = tk.Button(frame_bottom, text="Carica Immagine/Video", command=carica_immagine, width=40, height=3, font=font.Font(size=15))
upload.pack(side="left", padx=10, pady=5)

''' Bottone per fare il predict '''
predict = tk.Button(frame_bottom, text="Predici", command=get_Prediction, width=40, height=3, font=font.Font(size=15))
predict.config(state= "disabled")
predict.pack(side="right", padx=10, pady=5)

''' Bottone per il play del video '''
play = tk.Button(frame_bottom, text="Pause", command=play_pause_video, width=40, height=3, font=font.Font(size=15))


''' Bottone per salvare il video o immagine '''
save_b = tk.Button(frame_bottom, text="Salva", command=save_pred, width=40, height=3, font=font.Font(size=15))


''' Bottone per riniziare '''
restart_b = tk.Button(frame_bottom, text="Nuova predizione", command=nuova_pred, width=40, height=3, font=font.Font(size=15))

''' Bottone per cancellare '''
cancel = tk.Button(frame_bottom, text="Annulla", command=cancel, width=40, height=3, font=font.Font(size=15))
''' Bottone per fermarsi '''
stop = tk.Button(frame_bottom, text="Ferma", command=stop_prediction, width=40, height=3, font=font.Font(size=15))

''' Bottone per fare la predizione sempre dalla stessa fonte '''
repeat = tk.Button(frame_bottom, text="Predici di nuovo", command=get_Prediction, width=40, height=3, font=font.Font(size=15))

''' Checkbox per usare Sahi per l'inferenza '''
isSahi = tk.IntVar()  
sahi = tk.Checkbutton(frame_top, text="Inferenza con Sahi", variable=isSahi, font=font.Font(size=15))
sahi.pack(side = "left", pady=30)

''' Canvas per mostrare l'immagine/video '''
label_o = tk.Label(frame_origin, text="", font=("Helvetica", 16))
label_gt = tk.Label(frame_gt, text="", font=("Helvetica", 16))
label_p = tk.Label(frame_predit, text="", font=("Helvetica", 16))
label_o.pack(side="top", padx = 20, pady=10)
label_gt.pack(side="top", padx = 20, pady=10)
label_p.pack(side="top", padx = 20, pady=10)
canvas_original = tk.Canvas(frame_origin)
canvas_original.pack(side="left", fill="both", expand=True)
canvas_gt = tk.Canvas(frame_gt)
canvas_gt.pack(side="left", fill="both", expand=True)
canvas_predict = tk.Canvas(frame_predit)
canvas_predict.pack(fill="both", expand=True)

''' Progression Bar '''
progress = ttk.Progressbar(finestra, orient="horizontal", length=400, mode="determinate")

'''label per la dimensione dell'immagine'''
label_size = tk.Label(frame_undertop, text="", font=("Helvetica", 16))
label_size.pack()


finestra.mainloop()
