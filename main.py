import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms  # pastikan paket scikit-image sudah terinstal
from skimage.metrics import structural_similarity as ssim

import numpy as np
import cv2
import matplotlib.pyplot as plt
import asyncio
import time
from fastapi import status
from fastapi.responses import RedirectResponse
from typing import List
import sys
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import util

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

# Global variables untuk menyimpan state webcam
camera = None
is_capturing = False

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    file_path = save_image(img, "uploaded")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": file_path,
        "modified_image_path": file_path
    })

@app.post("/operation/", response_class=HTMLResponse)
async def perform_operation(
    request: Request,
    file: UploadFile = File(...),
    operation: str = Form(...),
    value: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")

    if operation == "add":
        result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "subtract":
        result_img = cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "max":
        result_img = np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "min":
        result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "inverse":
        result_img = cv2.bitwise_not(img)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/logic_operation/", response_class=HTMLResponse)
async def perform_logic_operation(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    operation: str = Form(...)
):
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)

    original_path = save_image(img1, "original")

    if operation == "not":
        result_img = cv2.bitwise_not(img1)
    else:
        if file2 is None:
            return HTMLResponse("Operasi AND dan XOR memerlukan dua gambar.", status_code=400)
        image_data2 = await file2.read()
        np_array2 = np.frombuffer(image_data2, np.uint8)
        img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)

        if operation == "and":
            result_img = cv2.bitwise_and(img1, img2)
        elif operation == "xor":
            result_img = cv2.bitwise_xor(img1, img2)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/grayscale/", response_class=HTMLResponse)
async def grayscale_form(request: Request):
    # Menampilkan form untuk upload gambar ke grayscale
    return templates.TemplateResponse("grayscale.html", {"request": request})

@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_grayscale(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_path = save_image(img, "original")
    modified_path = save_image(gray_img, "grayscale")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/histogram/", response_class=HTMLResponse)
async def histogram_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk histogram
    return templates.TemplateResponse("histogram.html", {"request": request})

@app.post("/histogram/", response_class=HTMLResponse)
async def generate_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Pastikan gambar berhasil diimpor
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    # Buat histogram grayscale dan berwarna
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_histogram_path = save_histogram(gray_img, "grayscale")

    color_histogram_path = save_color_histogram(img)

    return templates.TemplateResponse("histogram.html", {
        "request": request,
        "grayscale_histogram_path": grayscale_histogram_path,
        "color_histogram_path": color_histogram_path
    })

@app.get("/equalize/", response_class=HTMLResponse)
async def equalize_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk equalisasi histogram
    return templates.TemplateResponse("equalize.html", {"request": request})

@app.post("/equalize/", response_class=HTMLResponse)
async def equalize_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    equalized_img = cv2.equalizeHist(img)

    original_path = save_image(img, "original")
    modified_path = save_image(equalized_img, "equalized")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/specify/", response_class=HTMLResponse)
async def specify_form(request: Request):
    # Menampilkan halaman untuk upload gambar dan referensi untuk spesifikasi histogram
    return templates.TemplateResponse("specify.html", {"request": request})

@app.post("/specify/", response_class=HTMLResponse)
async def specify_histogram(request: Request, file: UploadFile = File(...), ref_file: UploadFile = File(...)):
    # Baca gambar yang diunggah dan gambar referensi
    image_data = await file.read()
    ref_image_data = await ref_file.read()

    np_array = np.frombuffer(image_data, np.uint8)
    ref_np_array = np.frombuffer(ref_image_data, np.uint8)
		
		#jika ingin grayscale
    #img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    #ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_GRAYSCALE)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Membaca gambar dalam format BGR
    ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)  # Membaca gambar referensi dalam format BGR


    if img is None or ref_img is None:
        return HTMLResponse("Gambar utama atau gambar referensi tidak dapat dibaca.", status_code=400)

    # Spesifikasi histogram menggunakan match_histograms dari skimage #grayscale
    #specified_img = match_histograms(img, ref_img, multichannel=False)
		    # Spesifikasi histogram menggunakan match_histograms dari skimage untuk gambar berwarna
    specified_img = match_histograms(img, ref_img, channel_axis=-1)
    # Konversi kembali ke format uint8 jika diperlukan
    specified_img = np.clip(specified_img, 0, 255).astype('uint8')

    original_path = save_image(img, "original")
    modified_path = save_image(specified_img, "specified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/statistics/", response_class=HTMLResponse)
async def calculate_statistics(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    image_path = save_image(img, "statistics")

    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "mean_intensity": mean_intensity,
        "std_deviation": std_deviation,
        "image_path": image_path
    })

def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"

def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"

'''MINGGU 3'''
def apply_convolution(image, kernel_type="average"):
    """
    Menerapkan konvolusi pada gambar dengan kernel yang dipilih
    
    Args:
        image: Gambar input dalam format BGR
        kernel_type: Jenis kernel yang akan digunakan
            - "average": Kernel rata-rata 3x3
            - "sharpen": Kernel penajaman
            - "edge": Kernel deteksi tepi
    
    Returns:
        Gambar hasil konvolusi
    """
    if kernel_type == "average":
        kernel = np.ones((3, 3), np.float32) / 9
    elif kernel_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif kernel_type == "edge":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    output_img = cv2.filter2D(image, -1, kernel)
    return output_img

def apply_zero_padding(image, padding_size=10):
    """
    Menerapkan zero padding pada gambar
    
    Args:
        image: Gambar input
        padding_size: Ukuran padding (dalam pixel)
    
    Returns:
        Gambar dengan zero padding
    """
    padded_img = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img

def apply_filter(image, filter_type="low"):
    """
    Menerapkan filter pada gambar
    
    Args:
        image: Gambar input
        filter_type: Jenis filter yang akan digunakan
            - "low": Low pass filter (Gaussian Blur)
            - "high": High pass filter
            - "band": Band pass filter
    
    Returns:
        Gambar hasil filter
    """
    if filter_type == "low":
        filtered_img = cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "high":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        filtered_img = cv2.filter2D(image, -1, kernel)
    elif filter_type == "band":
        low_pass = cv2.GaussianBlur(image, (9, 9), 0)
        high_pass = image - low_pass
        filtered_img = low_pass + high_pass

    return filtered_img

def apply_fourier_transform(image):
    """
    Menerapkan transformasi Fourier pada gambar
    
    Args:
        image: Gambar input
    
    Returns:
        Magnitude spectrum dari transformasi Fourier
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

def reduce_periodic_noise(image):
    """
    Mengurangi noise periodik pada gambar menggunakan transformasi Fourier
    
    Args:
        image: Gambar input
    
    Returns:
        Gambar dengan noise periodik yang berkurang
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Create a mask to remove periodic noise
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # Radius of the mask
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

# Endpoints untuk Convolution
@app.get("/convolution/", response_class=HTMLResponse)
async def convolution_form(request: Request):
    return templates.TemplateResponse("convolution.html", {"request": request})

@app.post("/convolution/", response_class=HTMLResponse)
async def apply_convolution_endpoint(request: Request, file: UploadFile = File(...), kernel_type: str = Form(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")
    result_img = apply_convolution(img, kernel_type)
    modified_path = save_image(result_img, "convolution")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

# Endpoints untuk Zero Padding
@app.get("/zeropadding/", response_class=HTMLResponse)
async def zero_padding_form(request: Request):
    return templates.TemplateResponse("zeropadding.html", {"request": request})

@app.post("/zeropadding/", response_class=HTMLResponse)
async def apply_zero_padding_endpoint(request: Request, file: UploadFile = File(...), padding_size: int = Form(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")
    padded_img = apply_zero_padding(img, padding_size)
    modified_path = save_image(padded_img, "padded")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

# Endpoints untuk Filter
@app.get("/filter/", response_class=HTMLResponse)
async def filter_form(request: Request):
    return templates.TemplateResponse("filter.html", {"request": request})

@app.post("/filter/", response_class=HTMLResponse)
async def apply_filter_endpoint(request: Request, file: UploadFile = File(...), filter_type: str = Form(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")
    filtered_img = apply_filter(img, filter_type)
    modified_path = save_image(filtered_img, "filtered")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

# Endpoints untuk Fourier Transform
@app.get("/fourier/", response_class=HTMLResponse)
async def fourier_form(request: Request):
    return templates.TemplateResponse("fourier.html", {"request": request})

@app.post("/fourier/", response_class=HTMLResponse)
async def apply_fourier_endpoint(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")
    magnitude_spectrum =  apply_fourier_transform(img)
    modified_path = save_image(magnitude_spectrum, "fourier")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

# Endpoints untuk Reduce Periodic Noise
@app.get("/reduceperiodicnoise/", response_class=HTMLResponse)
async def reduce_periodic_noise_form(request: Request):
    return templates.TemplateResponse("reduceperiodicnoise.html", {"request": request})

@app.post("/reduceperiodicnoise/", response_class=HTMLResponse)
async def apply_reduce_periodic_noise_endpoint(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")
    reduced_img = reduce_periodic_noise(img)
    modified_path = save_image(reduced_img, "reduced")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

def detect_faces(image):
    """
    Mendeteksi wajah dalam gambar menggunakan Haar Cascade
    
    Args:
        image: Gambar input dalam format BGR
    
    Returns:
        List of tuples (x, y, w, h) yang merepresentasikan lokasi wajah
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    return faces



async def generate_frames():
    global camera, is_capturing
    while is_capturing:
        if camera is None:
            camera = cv2.VideoCapture(0)
        
        success, frame = camera.read()
        if not success:
            break
        else:
            # Deteksi wajah
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Gambar kotak di sekitar wajah
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.1)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(),
                           media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/start_capture")
async def start_capture():
    global is_capturing
    is_capturing = True
    return {"status": "capturing started"}

@app.post("/stop_capture")
async def stop_capture():
    global is_capturing, camera
    is_capturing = False
    if camera is not None:
        camera.release()
        camera = None
    return {"status": "capturing stopped"}

@app.post("/save_face")
async def save_face(person_name: str):
    global camera
    dataset_path = 'dataset'
    person_path = os.path.join(dataset_path, person_name)
    max_images = 20
    delay = 0.1

    # Cek jika kamera belum diinisialisasi
    if camera is None:
        return {"error": "Camera not initialized"}

    # Cek jika folder nama sudah ada
    if os.path.exists(person_path):
        return {"error": "Nama sudah ada di dataset. Silakan pilih nama lain atau tambahkan lebih banyak gambar."}

    # Buat folder untuk orang baru
    os.makedirs(person_path)
    num_images = 0
    saved_paths = []
    error_count = 0

    while num_images < max_images:
        success, frame = camera.read()
        if not success:
            error_count += 1
            if error_count > 5:
                break
            continue

        # Deteksi wajah
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                img_name = f"img_{num_images}.jpg"
                img_path = os.path.join(person_path, img_name)
                cv2.imwrite(img_path, face)
                saved_paths.append(img_path)
                num_images += 1
                break  # Simpan satu wajah per frame
        time.sleep(delay)

    if num_images == 0:
        return {"error": "Tidak ada wajah yang terdeteksi dari webcam."}

    return {
        "status": "success",
        "message": f"{num_images} gambar telah berhasil ditambahkan ke dataset {person_name}.",
        "paths": saved_paths
    }

@app.get("/compile/", response_class=HTMLResponse)
async def compile_form(request: Request):
    return templates.TemplateResponse("compile.html", {"request": request})

@app.post("/compile/", response_class=HTMLResponse)
async def compile_processing(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(None)):
    try:
        # Baca gambar utama
        image_data1 = await file1.read()
        np_array1 = np.frombuffer(image_data1, np.uint8)
        img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)
        
        if img1 is None:
            return HTMLResponse("Tidak dapat membaca gambar utama", status_code=400)
        
        # Baca gambar referensi jika ada
        img2 = None
        if file2:
            image_data2 = await file2.read()
            np_array2 = np.frombuffer(image_data2, np.uint8)
            img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)
            if img2 is None:
                return HTMLResponse("Tidak dapat membaca gambar referensi", status_code=400)
            
            # Resize gambar referensi agar sesuai dengan ukuran gambar utama
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        results = {}

        # --- Modul 1: Array RGB (Nilai Array) ---
        b, g, r = cv2.split(img1)
        results["red_array_str"] = np.array2string(r, separator=', ', threshold=sys.maxsize)
        results["green_array_str"] = np.array2string(g, separator=', ', threshold=sys.maxsize)
        results["blue_array_str"] = np.array2string(b, separator=', ', threshold=sys.maxsize)

        # Operasi Aritmatika
        value = 50  # nilai default untuk operasi aritmatika
        results["add"] = save_image(cv2.add(img1, np.full(img1.shape, value, dtype=np.uint8)), "add")
        results["subtract"] = save_image(cv2.subtract(img1, np.full(img1.shape, value, dtype=np.uint8)), "subtract")
        results["max"] = save_image(np.maximum(img1, np.full(img1.shape, value, dtype=np.uint8)), "max")
        results["min"] = save_image(np.minimum(img1, np.full(img1.shape, value, dtype=np.uint8)), "min")

        # Operasi Logika
        results["logic_not"] = save_image(cv2.bitwise_not(img1), "not")
        if img2 is not None:
            results["logic_and"] = save_image(cv2.bitwise_and(img1, img2), "and")
            results["logic_xor"] = save_image(cv2.bitwise_xor(img1, img2), "xor")
        else:
            results["logic_and"] = results["logic_xor"] = results["logic_not"]

        # Grayscale dan Histogram
        gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        results["grayscale"] = save_image(gray_img, "grayscale")
        results["grayscale_histogram"] = save_histogram(gray_img, "grayscale")
        results["color_histogram"] = save_color_histogram(img1)

        # Equalisasi dan Spesifikasi
        results["equalized"] = save_image(cv2.equalizeHist(gray_img), "equalized")
        if img2 is not None:
            ref_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            results["specified"] = save_image(match_histograms(gray_img, ref_gray), "specified")
        else:
            results["specified"] = results["equalized"]

        # Filter dan Transformasi
        results["convolution"] = save_image(apply_convolution(img1), "convolution")
        results["zero_padding"] = save_image(apply_zero_padding(img1), "zeropadding")
        results["filter"] = save_image(apply_filter(img1), "filter")
        results["fourier"] = save_image(apply_fourier_transform(img1), "fourier")

        # Pengurangan Noise
        results["noise_reduction"] = save_image(reduce_periodic_noise(img1), "noise_reduction")

        # --- Modul 5: Canny ---
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        low_threshold = 50
        high_threshold = 150
        canny_edges = cv2.Canny(blurred, low_threshold, high_threshold)
        results["canny"] = save_image(canny_edges, "canny")

        # --- Modul 5: Chaincode ---
        _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        chaincode_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        chaincode_text = "Tidak ada kontur ditemukan."
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(chaincode_img, [largest], -1, (0,255,0), 1)
            def generate_freeman_chain_code(contour):
                chain_code = []
                if len(contour) < 2:
                    return chain_code
                directions = {
                    (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
                    (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
                }
                for i in range(len(contour)):
                    p1 = contour[i][0]
                    p2 = contour[(i + 1) % len(contour)][0]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    norm_dx = np.sign(dx)
                    norm_dy = np.sign(dy)
                    code = directions.get((norm_dx, norm_dy))
                    if code is not None:
                        chain_code.append(code)
                return chain_code
            chaincode_result = generate_freeman_chain_code(largest)
            chaincode_text = ', '.join(map(str, chaincode_result))
        results["chaincode_img"] = save_image(chaincode_img, "chaincode")
        results["chaincode_text"] = chaincode_text

        # --- Modul 5: Proyeksi Integral ---
        _, binary_img_proj = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_norm = binary_img_proj / 255.0
        horizontal_projection = np.sum(binary_norm, axis=0)
        vertical_projection = np.sum(binary_norm, axis=1)
        height, width = binary_norm.shape
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        ax_img = fig.add_subplot(gs[1, 0])
        ax_img.imshow(binary_norm, cmap='gray')
        ax_img.set_title('Citra Biner (Objek=1)')
        ax_img.set_xlabel('Indeks Kolom')
        ax_img.set_ylabel('Indeks Baris')
        ax_hproj = fig.add_subplot(gs[0, 0], sharex=ax_img)
        ax_hproj.plot(np.arange(width), horizontal_projection)
        ax_hproj.set_title('Proyeksi Horizontal (Profil Vertikal)')
        ax_hproj.set_ylabel('Jumlah Piksel')
        plt.setp(ax_hproj.get_xticklabels(), visible=False)
        ax_vproj = fig.add_subplot(gs[1, 1], sharey=ax_img)
        ax_vproj.plot(vertical_projection, np.arange(height))
        ax_vproj.set_title('Proyeksi Vertikal')
        ax_vproj.set_xlabel('Jumlah Piksel')
        ax_vproj.invert_yaxis()
        plt.setp(ax_vproj.get_yticklabels(), visible=False)
        plt.suptitle("Analisis Proyeksi Integral", fontsize=14)
        base = f"static/uploads/{uuid4()}"
        plot_path = base + "_proyeksi.png"
        plt.savefig(plot_path)
        plt.close(fig)
        results["proyeksi_integral"] = "/" + plot_path

        # --- Modul 6: Kompresi ---
        # Simpan file asli sementara untuk perbandingan ukuran dan kualitas
        temp_orig_path = f"static/uploads/{uuid4()}_compile_orig.png"
        cv2.imwrite(temp_orig_path, img1)
        original_size_kb = get_file_size_kb(temp_orig_path)
        results["original_size"] = original_size_kb
        results["original_path"] = "/" + temp_orig_path
        
        # --- Lossy (JPEG Q=90) ---
        lossy_path = f"static/uploads/{uuid4()}_compile_lossy.jpeg"
        cv2.imwrite(lossy_path, img1, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_lossy = cv2.imread(lossy_path)
        results["lossy_path"] = "/" + lossy_path
        lossy_size_kb = get_file_size_kb(lossy_path)
        results["lossy_size"] = lossy_size_kb
        # Hitung Metrik Lossy
        min_dim = min(img1.shape[:2])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        if win_size < 3: win_size = 3
        ssim_lossy, _ = ssim(img1, img_lossy, full=True, channel_axis=2, win_size=win_size)
        results["lossy_psnr"] = f"{cv2.PSNR(img1, img_lossy):.2f}"
        results["lossy_ssim"] = f"{ssim_lossy:.4f}"
        results["lossy_ratio"] = f"{original_size_kb / lossy_size_kb if lossy_size_kb > 0 else 0:.2f} : 1"
        
        # --- Lossless (PNG L=9) ---
        lossless_path = f"static/uploads/{uuid4()}_compile_lossless.png"
        cv2.imwrite(lossless_path, img1, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        img_lossless = cv2.imread(lossless_path)
        results["lossless_path"] = "/" + lossless_path
        lossless_size_kb = get_file_size_kb(lossless_path)
        results["lossless_size"] = lossless_size_kb
        # Hitung Metrik Lossless
        psnr_lossless = cv2.PSNR(img1, img_lossless)
        ssim_lossless, _ = ssim(img1, img_lossless, full=True, channel_axis=2, win_size=win_size)
        results["lossless_psnr"] = f"{psnr_lossless:.2f}" if psnr_lossless != float('inf') else 'Infinity'
        results["lossless_ssim"] = f"{ssim_lossless:.4f}"
        results["lossless_ratio"] = f"{original_size_kb / lossless_size_kb if lossless_size_kb > 0 else 0:.2f} : 1"
        results["lossless_identical"] = "Ya" if np.array_equal(img1, img_lossless) else "Tidak"

        # os.remove(temp_orig_path) # Hapus file sementara agar tidak menumpuk, tapi kita butuh untuk ditampilkan

        # --- Modul 7: Warna ---
        image_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        base7 = f"static/uploads/{uuid4()}_compile7"
        # RGB
        rgb_path = base7 + "_rgb.png"
        cv2.imwrite(rgb_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        R, G, B = cv2.split(image_rgb)
        r_path = base7 + "_r.png"
        g_path = base7 + "_g.png"
        b_path = base7 + "_b.png"
        cv2.imwrite(r_path, R)
        cv2.imwrite(g_path, G)
        cv2.imwrite(b_path, B)
        # XYZ
        image_xyz = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2XYZ)
        xyz_path = base7 + "_xyz.png"
        cv2.imwrite(xyz_path, cv2.cvtColor(image_xyz, cv2.COLOR_RGB2BGR))
        X, Y, Z = cv2.split(image_xyz)
        x_path = base7 + "_x.png"
        y_path = base7 + "_y.png"
        z_path = base7 + "_z.png"
        cv2.imwrite(x_path, X)
        cv2.imwrite(y_path, Y)
        cv2.imwrite(z_path, Z)
        # Lab
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
        lab_path = base7 + "_lab.png"
        cv2.imwrite(lab_path, cv2.cvtColor(image_lab, cv2.COLOR_RGB2BGR))
        L, a, bb = cv2.split(image_lab)
        l_path = base7 + "_l.png"
        a_path = base7 + "_a.png"
        bb_path = base7 + "_bb.png"
        cv2.imwrite(l_path, L)
        cv2.imwrite(a_path, a)
        cv2.imwrite(bb_path, bb)
        # YCbCr
        image_ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        ycbcr_path = base7 + "_ycbcr.png"
        cv2.imwrite(ycbcr_path, cv2.cvtColor(image_ycbcr, cv2.COLOR_RGB2BGR))
        Y_ycbcr, Cr, Cb = cv2.split(image_ycbcr)
        y_ycbcr_path = base7 + "_y_ycbcr.png"
        cb_path = base7 + "_cb.png"
        cr_path = base7 + "_cr.png"
        cv2.imwrite(y_ycbcr_path, Y_ycbcr)
        cv2.imwrite(cb_path, Cb)
        cv2.imwrite(cr_path, Cr)
        # HSV
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hsv_path = base7 + "_hsv.png"
        cv2.imwrite(hsv_path, cv2.cvtColor(image_hsv, cv2.COLOR_RGB2BGR))
        H, S, V = cv2.split(image_hsv)
        h_path = base7 + "_h.png"
        s_path = base7 + "_s.png"
        v_path = base7 + "_v.png"
        cv2.imwrite(h_path, H)
        cv2.imwrite(s_path, S)
        cv2.imwrite(v_path, V)
        # YIQ
        def rgb_to_yiq(rgb):
            rgb_norm = rgb.astype(np.float32) / 255.0
            transform_matrix = np.array([
                [0.299, 0.587, 0.114],
                [0.596, -0.274, -0.322],
                [0.211, -0.523, 0.312]
            ])
            height, width, _ = rgb_norm.shape
            rgb_reshaped = rgb_norm.reshape(height * width, 3)
            yiq_reshaped = np.dot(rgb_reshaped, transform_matrix.T)
            yiq = yiq_reshaped.reshape(height, width, 3)
            return yiq
        image_yiq = rgb_to_yiq(image_rgb)
        yiq_img = np.clip(image_yiq * 255, 0, 255).astype(np.uint8)
        yiq_path = base7 + "_yiq.png"
        cv2.imwrite(yiq_path, cv2.cvtColor(yiq_img, cv2.COLOR_RGB2BGR))
        Y_yiq = image_yiq[:, :, 0]
        I = image_yiq[:, :, 1]
        Q = image_yiq[:, :, 2]
        y_yiq_path = base7 + "_y_yiq.png"
        i_path = base7 + "_i.png"
        q_path = base7 + "_q.png"
        cv2.imwrite(y_yiq_path, np.clip(Y_yiq * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(i_path, np.clip((I - I.min()) / (I.max() - I.min()) * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(q_path, np.clip((Q - Q.min()) / (Q.max() - Q.min()) * 255, 0, 255).astype(np.uint8))
        # Perbandingan luminansi
        fig = plt.figure(figsize=(12, 8))
        luminance_components = {
            'Y dari YCbCr': Y_ycbcr,
            'L dari Lab': L,
            'Y dari YIQ': (Y_yiq * 255).astype(np.uint8),
            'V dari HSV': V
        }
        i = 1
        for name, component in luminance_components.items():
            plt.subplot(2, 2, i)
            plt.imshow(component, cmap='gray')
            plt.title(name)
            plt.axis('off')
            i += 1
        plt.tight_layout()
        luminance_compare_path = base7 + "_luminance_compare.png"
        plt.savefig(luminance_compare_path)
        plt.close(fig)
        results.update({
            "warna_rgb_path": "/" + rgb_path,
            "warna_r_path": "/" + r_path,
            "warna_g_path": "/" + g_path,
            "warna_b_path": "/" + b_path,
            "warna_xyz_path": "/" + xyz_path,
            "warna_x_path": "/" + x_path,
            "warna_y_path": "/" + y_path,
            "warna_z_path": "/" + z_path,
            "warna_lab_path": "/" + lab_path,
            "warna_l_path": "/" + l_path,
            "warna_a_path": "/" + a_path,
            "warna_bb_path": "/" + bb_path,
            "warna_ycbcr_path": "/" + ycbcr_path,
            "warna_y_ycbcr_path": "/" + y_ycbcr_path,
            "warna_cb_path": "/" + cb_path,
            "warna_cr_path": "/" + cr_path,
            "warna_hsv_path": "/" + hsv_path,
            "warna_h_path": "/" + h_path,
            "warna_s_path": "/" + s_path,
            "warna_v_path": "/" + v_path,
            "warna_yiq_path": "/" + yiq_path,
            "warna_y_yiq_path": "/" + y_yiq_path,
            "warna_i_path": "/" + i_path,
            "warna_q_path": "/" + q_path,
            "warna_luminance_compare_path": "/" + luminance_compare_path
        })
        # --- Modul 7: Tekstur ---
        base7t = f"static/uploads/{uuid4()}_compile7t"
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        rgb_path_t = base7t + "_rgb.png"
        gray_path_t = base7t + "_gray.png"
        cv2.imwrite(rgb_path_t, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(gray_path_t, image_gray)
        # Statistik tekstur
        def compute_texture_statistics(image, window_size=15):
            image = np.float32(image)
            feature_maps = {}
            mean = cv2.boxFilter(image, -1, (window_size, window_size))
            feature_maps['mean'] = mean
            mean_sqr = cv2.boxFilter(image*image, -1, (window_size, window_size))
            variance = mean_sqr - mean*mean
            variance = np.maximum(variance, 0)
            feature_maps['variance'] = variance
            std_dev = np.sqrt(variance)
            feature_maps['std_dev'] = std_dev
            norm_maps = {}
            for key, value in feature_maps.items():
                norm_maps[key] = cv2.normalize(value, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return norm_maps
        stat_maps = compute_texture_statistics(image_gray)
        mean_path = base7t + "_mean.png"
        variance_path = base7t + "_variance.png"
        stddev_path = base7t + "_stddev.png"
        cv2.imwrite(mean_path, stat_maps['mean'])
        cv2.imwrite(variance_path, stat_maps['variance'])
        cv2.imwrite(stddev_path, stat_maps['std_dev'])
        # GLCM
        distances = [1]
        angles = [0]
        glcm = graycomatrix(util.img_as_ubyte(image_gray), distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        glcm_results = {prop: graycoprops(glcm, prop)[0, 0] for prop in glcm_props}
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(glcm[:, :, 0, 0], cmap='viridis')
        plt.colorbar(label='Frekuensi')
        plt.title('Matriks GLCM (jarak=1, sudut=0)')
        plt.tight_layout()
        glcm_matrix_path = base7t + "_glcm_matrix.png"
        plt.savefig(glcm_matrix_path)
        plt.close(fig)
        # LBP
        lbp_image = local_binary_pattern(image_gray, P=24, R=3, method='uniform')
        n_bins = int(lbp_image.max() + 1)
        hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        lbp_path = base7t + "_lbp.png"
        plt.imsave(lbp_path, lbp_image, cmap='jet')
        fig = plt.figure(figsize=(6, 4))
        plt.bar(range(len(hist)), hist)
        plt.title('Histogram LBP')
        plt.xlabel('Nilai LBP')
        plt.ylabel('Frekuensi')
        plt.tight_layout()
        lbp_hist_path = base7t + "_lbp_hist.png"
        plt.savefig(lbp_hist_path)
        plt.close(fig)
        # Gabor
        frequencies = [0.1, 0.2]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        fig = plt.figure(figsize=(12, 6))
        idx = 1
        for frequency in frequencies:
            for theta in orientations:
                kernel_size = int(2 * np.ceil(frequency*10) + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma=frequency*10/3, theta=theta, lambd=1/frequency, gamma=0.5, psi=0)
                filtered = cv2.filter2D(image_gray, cv2.CV_32F, kernel)
                plt.subplot(len(frequencies), len(orientations), idx)
                plt.imshow(filtered, cmap='gray')
                plt.title(f'f={frequency}, θ={np.degrees(theta):.0f}°')
                plt.axis('off')
                idx += 1
        plt.tight_layout()
        gabor_path = base7t + "_gabor.png"
        plt.savefig(gabor_path)
        plt.close(fig)
        results.update({
            "tekstur_rgb_path": "/" + rgb_path_t,
            "tekstur_gray_path": "/" + gray_path_t,
            "tekstur_mean_path": "/" + mean_path,
            "tekstur_variance_path": "/" + variance_path,
            "tekstur_stddev_path": "/" + stddev_path,
            "tekstur_glcm_matrix_path": "/" + glcm_matrix_path,
            "tekstur_glcm_contrast": f"{glcm_results['contrast']:.4f}",
            "tekstur_glcm_dissimilarity": f"{glcm_results['dissimilarity']:.4f}",
            "tekstur_glcm_homogeneity": f"{glcm_results['homogeneity']:.4f}",
            "tekstur_glcm_energy": f"{glcm_results['energy']:.4f}",
            "tekstur_glcm_correlation": f"{glcm_results['correlation']:.4f}",
            "tekstur_glcm_asm": f"{glcm_results['ASM']:.4f}",
            "tekstur_lbp_path": "/" + lbp_path,
            "tekstur_lbp_hist_path": "/" + lbp_hist_path,
            "tekstur_gabor_path": "/" + gabor_path
        })

        return templates.TemplateResponse("compile.html", {
            "request": request,
            "results": results
        })
    except Exception as e:
        print(f"Error in compile_processing: {str(e)}")
        return HTMLResponse(f"Terjadi kesalahan: {str(e)}", status_code=500)

@app.get("/tambah_dataset/", response_class=HTMLResponse)
async def tambah_dataset_form(request: Request):
    return templates.TemplateResponse("tambah_dataset.html", {"request": request})

@app.post("/tambah_dataset/", response_class=HTMLResponse)
async def tambah_dataset(request: Request, person_name: str = Form(...)):
    if not person_name:
        return templates.TemplateResponse("tambah_dataset.html", {"request": request, "error": "Silakan masukkan nama orang baru."})

    dataset_path = 'dataset'
    person_path = os.path.join(dataset_path, person_name)
    max_images = 20
    max_attempts = 60  # Batas maksimal percobaan
    delay = 0.1
    image_paths = []

    if os.path.exists(person_path):
        return templates.TemplateResponse("tambah_dataset.html", {"request": request, "error": "Nama sudah ada di dataset. Silakan pilih nama lain atau tambahkan lebih banyak gambar."})

    os.makedirs(person_path)
    num_images = 0
    attempts = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return templates.TemplateResponse("tambah_dataset.html", {"request": request, "error": "Tidak dapat membuka webcam. Pastikan webcam terhubung dan tidak digunakan oleh aplikasi lain."})
    try:
        while num_images < max_images and attempts < max_attempts:
            ret, frame = cap.read()
            if not ret:
                attempts += 1
                continue
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    img_name = f"img_{num_images}.jpg"
                    img_path = os.path.join(person_path, img_name)
                    cv2.imwrite(img_path, face)
                    image_paths.append(f"/{img_path}")
                    num_images += 1
                    break  # Simpan satu wajah per frame
            attempts += 1
            time.sleep(delay)
    finally:
        cap.release()
    if num_images == 0:
        return templates.TemplateResponse("tambah_dataset.html", {"request": request, "error": "Tidak ada wajah yang terdeteksi dari webcam."})
    if num_images < max_images:
        return templates.TemplateResponse("tambah_dataset.html", {"request": request, "error": f"Hanya berhasil menyimpan {num_images} dari {max_images} gambar. Pastikan wajah terlihat jelas di kamera dan coba lagi.", "image_paths": image_paths})
    return templates.TemplateResponse("tambah_dataset.html", {"request": request, "message": f"{num_images} gambar telah berhasil ditambahkan ke dataset {person_name}.", "image_paths": image_paths})

@app.get("/arrayrgb/", response_class=HTMLResponse)
async def arrayrgb_home(request: Request):
    return templates.TemplateResponse("array_RBG.html", {"request": request})

@app.post("/arrayrgb/upload/", response_class=HTMLResponse)
async def arrayrgb_upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    file_extension = file.filename.split(".")[-1]
    filename = f"{uuid4()}.{file_extension}"
    file_path = os.path.join("static", "uploads", filename)

    with open(file_path, "wb") as f:
        f.write(image_data)

    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    rgb_array = {"R": r.tolist(), "G": g.tolist(), "B": b.tolist()}

    return templates.TemplateResponse("display_rgb.html", {
        "request": request,
        "image_path": f"/static/uploads/{filename}",
        "rgb_array": rgb_array
    })

# --- Modul 5: Canny Edge Detection ---
@app.get("/canny/", response_class=HTMLResponse)
async def canny_form(request: Request):
    return templates.TemplateResponse("canny.html", {"request": request})

@app.post("/canny/", response_class=HTMLResponse)
async def canny_process(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        return HTMLResponse("Gagal membaca gambar", status_code=400)
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    # Simpan gambar sementara
    base = f"static/uploads/{uuid4()}"
    orig_path = base + "_orig.png"
    blur_path = base + "_blur.png"
    canny_path = base + "_canny.png"
    cv2.imwrite(orig_path, img)
    cv2.imwrite(blur_path, blurred)
    cv2.imwrite(canny_path, edges)
    return templates.TemplateResponse("canny.html", {
        "request": request,
        "original_image_path": "/" + orig_path,
        "blurred_image_path": "/" + blur_path,
        "canny_image_path": "/" + canny_path
    })

# --- Modul 5: Chaincode ---
@app.get("/chaincode/", response_class=HTMLResponse)
async def chaincode_form(request: Request):
    return templates.TemplateResponse("chaincode.html", {"request": request})

@app.post("/chaincode/", response_class=HTMLResponse)
async def chaincode_process(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return HTMLResponse("Gagal membaca gambar", status_code=400)
    # Simpan citra asli
    base = f"static/uploads/{uuid4()}"
    orig_path = base + "_orig.png"
    cv2.imwrite(orig_path, img)
    # Binarisasi
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    bin_path = base + "_bin.png"
    cv2.imwrite(bin_path, binary_img)
    # Kontur
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    chaincode_text = "Tidak ada kontur ditemukan."
    contour_path = base + "_contour.png"
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_img, [largest], -1, (0,255,0), 1)
        cv2.imwrite(contour_path, contour_img)
        # Chaincode
        def generate_freeman_chain_code(contour):
            chain_code = []
            if len(contour) < 2:
                return chain_code
            directions = {
                (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
                (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
            }
            for i in range(len(contour)):
                p1 = contour[i][0]
                p2 = contour[(i + 1) % len(contour)][0]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                norm_dx = np.sign(dx)
                norm_dy = np.sign(dy)
                code = directions.get((norm_dx, norm_dy))
                if code is not None:
                    chain_code.append(code)
            return chain_code
        chaincode_result = generate_freeman_chain_code(largest)
        chaincode_text = ', '.join(map(str, chaincode_result))
    else:
        cv2.imwrite(contour_path, contour_img)
    return templates.TemplateResponse("chaincode.html", {
        "request": request,
        "original_image_path": "/" + orig_path,
        "binary_image_path": "/" + bin_path,
        "contour_image_path": "/" + contour_path,
        "chaincode_text": chaincode_text
    })

# --- Modul 5: Proyeksi Integral ---
@app.get("/proyeksi_integral/", response_class=HTMLResponse)
async def proyeksi_integral_form(request: Request):
    return templates.TemplateResponse("proyeksi_integral.html", {"request": request})

@app.post("/proyeksi_integral/", response_class=HTMLResponse)
async def proyeksi_integral_process(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return HTMLResponse("Gagal membaca gambar", status_code=400)
    # Binarisasi Otsu
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_norm = binary_img / 255.0
    horizontal_projection = np.sum(binary_norm, axis=0)
    vertical_projection = np.sum(binary_norm, axis=1)
    height, width = binary_norm.shape
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.imshow(binary_norm, cmap='gray')
    ax_img.set_title('Citra Biner (Objek=1)')
    ax_img.set_xlabel('Indeks Kolom')
    ax_img.set_ylabel('Indeks Baris')
    ax_hproj = fig.add_subplot(gs[0, 0], sharex=ax_img)
    ax_hproj.plot(np.arange(width), horizontal_projection)
    ax_hproj.set_title('Proyeksi Horizontal (Profil Vertikal)')
    ax_hproj.set_ylabel('Jumlah Piksel')
    plt.setp(ax_hproj.get_xticklabels(), visible=False)
    ax_vproj = fig.add_subplot(gs[1, 1], sharey=ax_img)
    ax_vproj.plot(vertical_projection, np.arange(height))
    ax_vproj.set_title('Proyeksi Vertikal')
    ax_vproj.set_xlabel('Jumlah Piksel')
    ax_vproj.invert_yaxis()
    plt.setp(ax_vproj.get_yticklabels(), visible=False)
    plt.suptitle("Analisis Proyeksi Integral", fontsize=14)
    # Simpan plot ke file
    base = f"static/uploads/{uuid4()}"
    plot_path = base + "_proyeksi.png"
    plt.savefig(plot_path)
    plt.close(fig)
    return templates.TemplateResponse("proyeksi_integral.html", {
        "request": request,
        "projection_image_path": "/" + plot_path
    })

# --- Modul 6: Kompresi ---
def get_file_size_kb(path):
    return round(os.path.getsize(path) / 1024, 2)

@app.get("/modul61/", response_class=HTMLResponse)
async def modul61_form(request: Request):
    return templates.TemplateResponse("modul61.html", {"request": request})

@app.post("/modul61/", response_class=HTMLResponse)
async def modul61_process(request: Request, file: UploadFile = File(...), quality: int = Form(...)):
    # Baca gambar
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img_original = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Simpan gambar asli sementara untuk perbandingan ukuran
    temp_orig_path = f"static/uploads/{uuid4()}_orig_temp.png"
    cv2.imwrite(temp_orig_path, img_original)
    original_size_kb = get_file_size_kb(temp_orig_path)
    
    # Lakukan kompresi lossy (JPEG)
    compressed_path = f"static/uploads/{uuid4()}_lossy.jpeg"
    cv2.imwrite(compressed_path, img_original, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    compressed_size_kb = get_file_size_kb(compressed_path)

    # Baca kembali gambar terkompresi untuk analisis
    img_compressed = cv2.imread(compressed_path)

    # Hitung metrik
    psnr_value = cv2.PSNR(img_original, img_compressed)
    
    # SSIM
    min_dim = min(img_original.shape[:2])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3: win_size = 3
    ssim_value, _ = ssim(img_original, img_compressed, full=True, channel_axis=2, win_size=win_size)

    # Hitung rasio
    compression_ratio = f"{original_size_kb / compressed_size_kb if compressed_size_kb > 0 else 0:.2f} : 1"

    results = {
        "original_path": "/" + temp_orig_path,
        "compressed_path": "/" + compressed_path,
        "original_size_kb": original_size_kb,
        "compressed_size_kb": compressed_size_kb,
        "quality": quality,
        "compression_ratio": compression_ratio,
        "psnr": f"{psnr_value:.2f}",
        "ssim": f"{ssim_value:.4f}",
    }
    
    return templates.TemplateResponse("modul61.html", {"request": request, "results": results})


@app.get("/modul62/", response_class=HTMLResponse)
async def modul62_form(request: Request):
    return templates.TemplateResponse("modul62.html", {"request": request})

@app.post("/modul62/", response_class=HTMLResponse)
async def modul62_process(request: Request, file: UploadFile = File(...), level: int = Form(...)):
    # Baca gambar
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img_original = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Simpan gambar asli sementara untuk perbandingan ukuran
    temp_orig_path = f"static/uploads/{uuid4()}_orig_temp.jpeg" # Simpan sebagai jpeg agar ada kompresi awal
    cv2.imwrite(temp_orig_path, img_original, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    original_size_kb = get_file_size_kb(temp_orig_path)
    
    # Lakukan kompresi lossless (PNG)
    compressed_path = f"static/uploads/{uuid4()}_lossless.png"
    cv2.imwrite(compressed_path, img_original, [cv2.IMWRITE_PNG_COMPRESSION, level])
    compressed_size_kb = get_file_size_kb(compressed_path)

    # Baca kembali gambar terkompresi untuk analisis
    img_compressed = cv2.imread(compressed_path)

    # Hitung metrik
    psnr_value = cv2.PSNR(img_original, img_compressed)

    # SSIM
    min_dim = min(img_original.shape[:2])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3: win_size = 3
    ssim_value, _ = ssim(img_original, img_compressed, full=True, channel_axis=2, win_size=win_size)

    # Verifikasi identik
    is_identical = "Ya" if np.array_equal(img_original, img_compressed) else "Tidak"

    # Hitung rasio
    compression_ratio = f"{original_size_kb / compressed_size_kb if compressed_size_kb > 0 else 0:.2f} : 1"

    results = {
        "original_path": "/" + temp_orig_path,
        "compressed_path": "/" + compressed_path,
        "original_size_kb": original_size_kb,
        "compressed_size_kb": compressed_size_kb,
        "level": level,
        "compression_ratio": compression_ratio,
        "psnr": f"{psnr_value:.2f}" if psnr_value != float('inf') else 'Infinity',
        "ssim": f"{ssim_value:.4f}",
        "identical": is_identical
    }
    
    return templates.TemplateResponse("modul62.html", {"request": request, "results": results})

@app.get("/warna/", response_class=HTMLResponse)
async def warna_form(request: Request):
    return templates.TemplateResponse("warna.html", {"request": request})

@app.post("/warna/", response_class=HTMLResponse)
async def warna_process(request: Request, file: UploadFile = File(...)):
    import matplotlib.pyplot as plt
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Simpan RGB
    from uuid import uuid4
    base = f"static/uploads/{uuid4()}"
    rgb_path = base + "_rgb.png"
    cv2.imwrite(rgb_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    # Kanal RGB
    R, G, B = cv2.split(image_rgb)
    r_path = base + "_r.png"
    g_path = base + "_g.png"
    b_path = base + "_b.png"
    cv2.imwrite(r_path, R)
    cv2.imwrite(g_path, G)
    cv2.imwrite(b_path, B)
    # XYZ
    image_xyz = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2XYZ)
    xyz_path = base + "_xyz.png"
    cv2.imwrite(xyz_path, cv2.cvtColor(image_xyz, cv2.COLOR_RGB2BGR))
    X, Y, Z = cv2.split(image_xyz)
    x_path = base + "_x.png"
    y_path = base + "_y.png"
    z_path = base + "_z.png"
    cv2.imwrite(x_path, X)
    cv2.imwrite(y_path, Y)
    cv2.imwrite(z_path, Z)
    # Lab
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    lab_path = base + "_lab.png"
    cv2.imwrite(lab_path, cv2.cvtColor(image_lab, cv2.COLOR_RGB2BGR))
    L, a, bb = cv2.split(image_lab)
    l_path = base + "_l.png"
    a_path = base + "_a.png"
    bb_path = base + "_bb.png"
    cv2.imwrite(l_path, L)
    cv2.imwrite(a_path, a)
    cv2.imwrite(bb_path, bb)
    # YCbCr
    image_ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    ycbcr_path = base + "_ycbcr.png"
    cv2.imwrite(ycbcr_path, cv2.cvtColor(image_ycbcr, cv2.COLOR_RGB2BGR))
    Y_ycbcr, Cr, Cb = cv2.split(image_ycbcr)
    y_ycbcr_path = base + "_y_ycbcr.png"
    cb_path = base + "_cb.png"
    cr_path = base + "_cr.png"
    cv2.imwrite(y_ycbcr_path, Y_ycbcr)
    cv2.imwrite(cb_path, Cb)
    cv2.imwrite(cr_path, Cr)
    # HSV
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hsv_path = base + "_hsv.png"
    cv2.imwrite(hsv_path, cv2.cvtColor(image_hsv, cv2.COLOR_RGB2BGR))
    H, S, V = cv2.split(image_hsv)
    h_path = base + "_h.png"
    s_path = base + "_s.png"
    v_path = base + "_v.png"
    cv2.imwrite(h_path, H)
    cv2.imwrite(s_path, S)
    cv2.imwrite(v_path, V)
    # YIQ
    def rgb_to_yiq(rgb):
        rgb_norm = rgb.astype(np.float32) / 255.0
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.312]
        ])
        height, width, _ = rgb_norm.shape
        rgb_reshaped = rgb_norm.reshape(height * width, 3)
        yiq_reshaped = np.dot(rgb_reshaped, transform_matrix.T)
        yiq = yiq_reshaped.reshape(height, width, 3)
        return yiq
    image_yiq = rgb_to_yiq(image_rgb)
    # Simpan YIQ sebagai gambar (skala 0-255)
    yiq_img = np.clip(image_yiq * 255, 0, 255).astype(np.uint8)
    yiq_path = base + "_yiq.png"
    cv2.imwrite(yiq_path, cv2.cvtColor(yiq_img, cv2.COLOR_RGB2BGR))
    Y_yiq = image_yiq[:, :, 0]
    I = image_yiq[:, :, 1]
    Q = image_yiq[:, :, 2]
    y_yiq_path = base + "_y_yiq.png"
    i_path = base + "_i.png"
    q_path = base + "_q.png"
    cv2.imwrite(y_yiq_path, np.clip(Y_yiq * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(i_path, np.clip((I - I.min()) / (I.max() - I.min()) * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(q_path, np.clip((Q - Q.min()) / (Q.max() - Q.min()) * 255, 0, 255).astype(np.uint8))
    # Perbandingan luminansi
    fig = plt.figure(figsize=(12, 8))
    luminance_components = {
        'Y dari YCbCr': Y_ycbcr,
        'L dari Lab': L,
        'Y dari YIQ': (Y_yiq * 255).astype(np.uint8),
        'V dari HSV': V
    }
    i = 1
    for name, component in luminance_components.items():
        plt.subplot(2, 2, i)
        plt.imshow(component, cmap='gray')
        plt.title(name)
        plt.axis('off')
        i += 1
    plt.tight_layout()
    luminance_compare_path = base + "_luminance_compare.png"
    plt.savefig(luminance_compare_path)
    plt.close(fig)
    results = {
        "rgb_path": "/" + rgb_path,
        "r_path": "/" + r_path,
        "g_path": "/" + g_path,
        "b_path": "/" + b_path,
        "xyz_path": "/" + xyz_path,
        "x_path": "/" + x_path,
        "y_path": "/" + y_path,
        "z_path": "/" + z_path,
        "lab_path": "/" + lab_path,
        "l_path": "/" + l_path,
        "a_path": "/" + a_path,
        "bb_path": "/" + bb_path,
        "ycbcr_path": "/" + ycbcr_path,
        "y_ycbcr_path": "/" + y_ycbcr_path,
        "cb_path": "/" + cb_path,
        "cr_path": "/" + cr_path,
        "hsv_path": "/" + hsv_path,
        "h_path": "/" + h_path,
        "s_path": "/" + s_path,
        "v_path": "/" + v_path,
        "yiq_path": "/" + yiq_path,
        "y_yiq_path": "/" + y_yiq_path,
        "i_path": "/" + i_path,
        "q_path": "/" + q_path,
        "luminance_compare_path": "/" + luminance_compare_path
    }
    return templates.TemplateResponse("warna.html", {"request": request, "results": results})

@app.get("/tekstur/", response_class=HTMLResponse)
async def tekstur_form(request: Request):
    return templates.TemplateResponse("tekstur.html", {"request": request})

@app.post("/tekstur/", response_class=HTMLResponse)
async def tekstur_process(request: Request, file: UploadFile = File(...)):
    import matplotlib.pyplot as plt
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage import util
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base = f"static/uploads/{uuid4()}"
    rgb_path = base + "_rgb.png"
    cv2.imwrite(rgb_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    # Grayscale
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_path = base + "_gray.png"
    cv2.imwrite(gray_path, image_gray)
    # Statistik tekstur
    def compute_texture_statistics(image, window_size=15):
        image = np.float32(image)
        feature_maps = {}
        mean = cv2.boxFilter(image, -1, (window_size, window_size))
        feature_maps['mean'] = mean
        mean_sqr = cv2.boxFilter(image*image, -1, (window_size, window_size))
        variance = mean_sqr - mean*mean
        variance = np.maximum(variance, 0)
        feature_maps['variance'] = variance
        std_dev = np.sqrt(variance)
        feature_maps['std_dev'] = std_dev
        norm_maps = {}
        for key, value in feature_maps.items():
            norm_maps[key] = cv2.normalize(value, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return norm_maps
    stat_maps = compute_texture_statistics(image_gray)
    mean_path = base + "_mean.png"
    variance_path = base + "_variance.png"
    stddev_path = base + "_stddev.png"
    cv2.imwrite(mean_path, stat_maps['mean'])
    cv2.imwrite(variance_path, stat_maps['variance'])
    cv2.imwrite(stddev_path, stat_maps['std_dev'])
    # GLCM
    distances = [1]
    angles = [0]
    glcm = graycomatrix(util.img_as_ubyte(image_gray), distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    glcm_results = {prop: graycoprops(glcm, prop)[0, 0] for prop in glcm_props}
    # Visualisasi matriks GLCM
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(glcm[:, :, 0, 0], cmap='viridis')
    plt.colorbar(label='Frekuensi')
    plt.title('Matriks GLCM (jarak=1, sudut=0)')
    plt.tight_layout()
    glcm_matrix_path = base + "_glcm_matrix.png"
    plt.savefig(glcm_matrix_path)
    plt.close(fig)
    # LBP
    lbp_image = local_binary_pattern(image_gray, P=24, R=3, method='uniform')
    n_bins = int(lbp_image.max() + 1)
    hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    # Simpan LBP
    lbp_path = base + "_lbp.png"
    plt.imsave(lbp_path, lbp_image, cmap='jet')
    # Simpan histogram LBP
    fig = plt.figure(figsize=(6, 4))
    plt.bar(range(len(hist)), hist)
    plt.title('Histogram LBP')
    plt.xlabel('Nilai LBP')
    plt.ylabel('Frekuensi')
    plt.tight_layout()
    lbp_hist_path = base + "_lbp_hist.png"
    plt.savefig(lbp_hist_path)
    plt.close(fig)
    # Gabor
    frequencies = [0.1, 0.2]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    fig = plt.figure(figsize=(12, 6))
    idx = 1
    for frequency in frequencies:
        for theta in orientations:
            kernel_size = int(2 * np.ceil(frequency*10) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma=frequency*10/3, theta=theta, lambd=1/frequency, gamma=0.5, psi=0)
            filtered = cv2.filter2D(image_gray, cv2.CV_32F, kernel)
            plt.subplot(len(frequencies), len(orientations), idx)
            plt.imshow(filtered, cmap='gray')
            plt.title(f'f={frequency}, θ={np.degrees(theta):.0f}°')
            plt.axis('off')
            idx += 1
    plt.tight_layout()
    gabor_path = base + "_gabor.png"
    plt.savefig(gabor_path)
    plt.close(fig)
    results = {
        "rgb_path": "/" + rgb_path,
        "gray_path": "/" + gray_path,
        "mean_path": "/" + mean_path,
        "variance_path": "/" + variance_path,
        "stddev_path": "/" + stddev_path,
        "glcm_matrix_path": "/" + glcm_matrix_path,
        "glcm_contrast": f"{glcm_results['contrast']:.4f}",
        "glcm_dissimilarity": f"{glcm_results['dissimilarity']:.4f}",
        "glcm_homogeneity": f"{glcm_results['homogeneity']:.4f}",
        "glcm_energy": f"{glcm_results['energy']:.4f}",
        "glcm_correlation": f"{glcm_results['correlation']:.4f}",
        "glcm_asm": f"{glcm_results['ASM']:.4f}",
        "lbp_path": "/" + lbp_path,
        "lbp_hist_path": "/" + lbp_hist_path,
        "gabor_path": "/" + gabor_path
    }
    return templates.TemplateResponse("tekstur.html", {"request": request, "results": results})

        





