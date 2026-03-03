# Android App — QR Quishing Detector

Real-time QR code scanner with on-device TFLite malicious-URL detection.

## Requirements

| Tool | Minimum version |
|------|-----------------|
| Android Studio | Hedgehog (2023.1.1) |
| JDK | 17 |
| Android SDK | 34 |
| Android NDK | Not required |
| Emulator API level | 26 (Android 8.0) or higher |

---

## First-time Setup

### 1. Place ML model & vocab

Copy the best TFLite model produced by `model_training/` into the Android
assets folder:

```bash
# From the repo root:
cp model_training/exports/tflite/<best_model>.tflite \
   android_app/app/src/main/assets/model.tflite

# Copy the corresponding vocab.txt (from HuggingFace cache or training dir):
cp <path_to_vocab.txt> \
   android_app/app/src/main/assets/vocab.txt
```

### 2. Open in Android Studio

```
File → Open → select android_app/
```

Android Studio will sync Gradle automatically.

### 3. Create an AVD (emulator)

1. **Tools → Device Manager → Create Device**
2. Pick a phone (e.g. Pixel 6), API 33, x86\_64 image
3. Finish and start the emulator

> For QR scanning on the emulator, use **Extended Controls → Camera →
> Virtual scene** and set a custom image, or run on a physical device.

---

## Build & Run

```bash
# From android_app/ with Gradle wrapper:
./gradlew assembleDebug

# Install directly on connected device / running emulator:
./gradlew installDebug
```

Or press **Run ▶** in Android Studio.

---

## Running Tests

```bash
# Unit tests (JVM):
./gradlew test

# Instrumented tests (requires emulator/device):
./gradlew connectedAndroidTest
```

---

## Backend Integration

The app posts uncertain URLs to `http://10.0.2.2:8080/validate` (the
standard emulator address for `localhost`).  To change the endpoint, edit
`BACKEND_URL` in `MainActivity.kt`.

Ensure the backend server is running before testing uncertain-verdict flows:

```bash
cd ../backend_server
mvn spring-boot:run
```

---

## Architecture Notes

| Component | Detail |
|-----------|--------|
| QR scanning | ZXing `IntentIntegrator` |
| On-device model | TFLite `Interpreter`, 2 threads |
| Tokenisation | Character-level vocabulary (`vocab.txt`) |
| Threshold | 0.80 — below this confidence, verdict = UNCERTAIN |
| Network | OkHttp + coroutines (`Dispatchers.IO`) |
| Min SDK | 26 (Android 8.0 Oreo) |
