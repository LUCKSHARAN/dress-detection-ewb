---
title: Dress Code Detector
emoji: 👔
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
app_port: 7860
---

# Dress Code Detector

Real-time formal dress code detection for male using YOLOv8 + Flask.

## How to Use
1. Click **Start Detection**
2. Allow camera permission
3. Stand **1.5–2 meters** away — full body must be visible
4. Use **🔄 Flip Camera** on mobile to switch front/back camera
5. System detects **FORMAL** or **INFORMAL** in real time

## Tech Stack
- YOLOv8m (custom trained — 11 clothing classes)
- Flask + OpenCV
- Browser-based camera (zero lag)
- 12-frame temporal smoothing
- Front/Back camera support on mobile
