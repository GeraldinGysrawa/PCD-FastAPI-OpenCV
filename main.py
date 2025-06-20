import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms  # pastikan paket scikit-image sudah terinstal

import numpy as np
import cv2
import matplotlib.pyplot as plt
import asyncio
import time
from fastapi import status
from fastapi.responses import RedirectResponse
from typing import List

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

        





