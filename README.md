# 🎬 Video Enhancement Lab

A real-time video enhancement dashboard built with Streamlit that applies advanced image processing techniques to improve video quality. Process entire videos in batch mode with comprehensive analysis and visualization tools.

---

## 📋 Features

### **Page 1: Video Processing & Playback**
- Upload videos in MP4, AVI, MOV, or MKV format
- Process entire videos using a 5-stage enhancement pipeline
- Play original and enhanced videos side-by-side with native HTML5 players
- View average **PSNR** and **SSIM** metrics across all frames
- Real-time progress tracking with ETA

### **Page 2: Frame Analysis & Heatmap**
- Select any individual frame with an interactive slider
- See original vs. enhanced frame comparison
- Generate pixel-level difference heatmaps using COLORMAP_JET
- View per-frame PSNR and SSIM metrics
- Analyze distribution of pixel changes with histograms
- 50/50 blended overlay for spatial context

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- pip or conda package manager

### Setup

1. **Clone or download the project:**
   ```bash
   cd "Video Enhancement"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| **streamlit** | Web app framework |
| **opencv-python** | Video/image processing |
| **numpy** | Numerical computations |
| **pandas** | Data handling |
| **pillow** | Image operations |
| **imageio** | Video I/O |
| **scikit-image** | PSNR/SSIM metrics |
| **imageio-ffmpeg** | FFmpeg backend for video encoding |

---

## 🎨 Enhancement Pipeline

The app applies a **6-stage enhancement process** to each frame (operating on the L channel in LAB color space):

```
1. BGR → LAB              Color space isolation (process luminance only)
2. Median Blur            Remove impulse/salt-and-pepper noise
3. Gamma Correction       Power-law exposure adjustment (γ parameter)
4. CLAHE                  Adaptive contrast enhancement (8×8 tiles, limit 1.5)
5. Gaussian High-Pass     FFT-based edge sharpening (cutoff 5%, boost 1.0)
6. Blending               70% enhanced + 30% original (preserve PSNR/SSIM)
```

### Adjustable Parameters

| Control | Range | Effect |
|---------|-------|--------|
| **Gamma (γ)** | 0.1 – 3.0 | γ > 1 brightens shadows; γ < 1 darkens highlights |
| **Kernel Size** | 3, 5, 7 | Larger kernel = stronger noise removal (slower) |

---

## 📊 Metrics Explained

### **PSNR (Peak Signal-to-Noise Ratio)**
- Measures pixel-level difference between original and enhanced frames
- Lower PSNR reflects the **magnitude of enhancement** (algorithm changes pixel values)
- **Target:** 20–35 dB typical for heavy enhancement
- **Formula:** `PSNR = 10 × log₁₀(255² / MSE)` (in dB)

### **SSIM (Structural Similarity Index)**
- Measures preservation of edges, textures, and luminance
- Value close to **1.0** = edges and boundaries preserved faithfully
- **Target:** ≥ 0.85 (excellent)
- Robust to aggressive color and contrast changes

---

## 📖 How to Use

### **Page 1: Video Processing & Playback**

1. **Upload a video** in the sidebar (supported formats: MP4, AVI, MOV, MKV)
2. **Adjust enhancement parameters:**
   - Gamma slider (Stage 3)
   - Kernel size selector (Stage 2)
3. **Click "Process Video"** to begin batch processing
4. Monitor **real-time progress** with frame count and ETA
5. Once complete, **play both videos** side-by-side using native controls
6. Review **average PSNR/SSIM** metrics in the cards below

### **Page 2: Frame Analysis & Heatmap**

1. **Select a frame** using the frame slider (1 to total frames)
2. View **original vs. enhanced** side-by-side
3. Check **per-frame metrics** (PSNR/SSIM for the selected frame)
4. Analyze **heatmaps:**
   - Pure heatmap shows pixel change intensity
   - Overlay blends heatmap with original for spatial context
   - Warm colors (red/yellow) = large changes
   - Cool colors (blue) = minimal changes
5. Review the **histogram** to see distribution of pixel modifications

---

## 🛠️ Project Structure

```
Video Enhancement/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── app/                   # (Optional) Additional modules
```

---

## ⚙️ Technical Details

- **Framework:** Streamlit
- **Processing:** OpenCV + NumPy (FFT for sharpening)
- **Video I/O:** imageio with libx264 codec
- **Color Space:** BGR (OpenCV) ↔ LAB ↔ RGB (Streamlit)
- **Caching:** Streamlit `@st.cache_data` for LUT and GHPF precomputation
- **Session State:** Persistent storage across page navigation

---

## 💡 Tips

- **For fast processing:** Use kernel size **3** and gamma **1.1**
- **For strong enhancement:** Use kernel size **7** and gamma **0.8** (darkens) or **1.5** (brightens)
- **For large videos:** Process in batches or use a smaller resolution video first
- **For heatmap analysis:** Frame numbers with high average change indicate noise-heavy regions

---

## 📝 Example Workflow

1. Open the app and navigate to **Page 1**
2. Upload a noisy or underexposed video
3. Set gamma to `1.2` and kernel to `5`
4. Click "Process Video" and wait for completion
5. Play the enhanced video and check metrics
6. Switch to **Page 2** to analyze specific frames in detail
7. Adjust parameters and reprocess if desired

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'cv2'` | Run `pip install -r requirements.txt` |
| Video won't upload | Ensure format is MP4, AVI, MOV, or MKV |
| Processing is slow | Reduce video resolution or use smaller kernel size |
| Heatmap appears uniform | Check if enhancement parameters are too subtle |

---

## 📄 License

This project is created for educational purposes.

---

## 👤 Author

Developed for Video Enhancement coursework (438).

---

**Questions or issues?** Refer to the inline comments in `app.py` for detailed explanations of each pipeline stage.
