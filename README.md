# 🩺 USAI10K Model Fine-Tuning Project

[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://www.tensorflow.org/guide/keras/save_and_serialize)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://docs.python.org/3.9/)
[![Model](https://img.shields.io/badge/Backbone-EfficientNetB5-green.svg)]()

คลังโค้ดสำหรับการเทรนต่อ (Continuation Training) และปรับละเอียด (Fine-tuning) โมเดล **EfficientNetB5** แบบ Single-task บนชุดข้อมูลภาพอัลตราซาวด์ทางการแพทย์ **USAI10K** โดยเริ่มต้นจากโมเดล Unlearn Checkpoint ดั้งเดิมที่มีความแม่นยำสูงถึง **89%** เพื่อจำแนกโรคและสิ่งผิดปกติหลัก 15 คลาสผ่าน `prediction_layer`

---

## 📌 ภาพรวมโครงการ (Project Overview)

โครงการนี้มุ่งเน้นการเพิ่มประสิทธิภาพโมเดลจำแนกโรคตับและท่อน้ำดีจากภาพอัลตราซาวด์ โดยใช้สถาปัตยกรรม **EfficientNetB5** และใช้เทคนิคการฝึกฝนแบบ **2 เฟส (Optimized 2-Phase Training)** ร่วมกับ **Custom Data Generator** ที่ออกแบบมาเพื่อดึงข้อมูลภาพโดยตรงจากไฟล์ตารางด้วย OpenCV ซึ่งทำให้การคำนวณและประมวลผลเป็นไปอย่างแม่นยำและเสถียรที่สุด

### 🎯 จุดเด่นของสคริปต์นี้
* **Single-Task Focus:** เจาะจงทำนายโรคและสิ่งผิดปกติทางการแพทย์ 15 คลาส (15 Classes) ผ่านชั้น `prediction_layer`
* **Custom Data Pipeline:** ดึงและประมวลผลรูปภาพแบบแมนนวลผ่าน OpenCV ช่วยควบคุมขนาดและ Data Augmentation ได้ตามต้องการ
* **Strict Data Splitting:** แบ่งชุดข้อมูลอย่างแม่นยำโดยใช้คอลัมน์ `split_data` จากตาราง Master Split ป้องกันปัญหา Data Leakage
* **Monkey Patching for FixedDropout:** แก้ปัญหา `ValueError: Unknown layer: FixedDropout` ที่มักพบใน Keras เวอร์ชันเก่าอย่างเบ็ดเสร็จ

---

## 🛠️ โครงสร้างการฝึกฝนโมเดล 2 เฟส (2-Phase Training Workflow)

เพื่อรักษาค่าน้ำหนักเดิม (Weights) ที่โมเดลเคยเรียนรู้มาไม่ให้เสียหาย (ป้องกัน Catastrophic Forgetting) การเทรนจึงถูกแบ่งออกเป็น 2 ลำดับขั้น:

| เฟสการฝึกฝน | การตั้งค่าเลเยอร์ (Layer Status) | Learning Rate | จำนวน Epoch | จุดประสงค์ |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 1: Heads Only** | Freeze ส่วน Backbone ทั้งหมด / เปิดเฉพาะหัวจำแนก | `1e-3` | 5 | วอร์มอัปส่วนหัว Classification ให้ปรับตัวเข้ากับข้อมูลใหม่ |
| **Phase 2: Fine-Tuning** | Unfreeze ส่วนท้ายของ Backbone (เฉพาะ Block 4 - Block 7) | `1e-5` | 20 | ปรับแต่งน้ำหนักอย่างละเอียดเพื่อดึงความแม่นยำสูงสุด |

---

## 💻 การตั้งค่าระบบและการใช้งาน (Environment & Execution)

### 1. การเตรียมสภาพแวดล้อม (Environment Setup)

### 📋 All Installation Commands

```bash
# 1. สร้าง Environment ใหม่ด้วย Python 3.9
conda create -n usai10k-python3.9 python=3.9 -y

# 2. เปิดใช้งาน Environment
conda activate usai10k-python3.9

# 3. ติดตั้งเครื่องมือคอมไพเลอร์พื้นฐาน (C++ / GCC) สำหรับระบบ Linux
conda install -c conda-forge gxx_linux-64 gcc_linux-64 -y

# 4. อัปเดตเครื่องมือติดตั้ง pip ให้เป็นเวอร์ชันล่าสุด
pip install --upgrade pip

# 5. ติดตั้ง TensorFlow 2.10.1 และไลบรารีประมวลผลข้อมูล/รูปภาพทั้งหมด
pip install tensorflow==2.10.1 pandas numpy opencv-python matplotlib scikit-learn jupyter ipykernel

# 6. บังคับย้อนเวอร์ชัน NumPy กลับไปเป็นซีรีส์ 1.x เพื่อแก้ปัญหาโครงสร้างอะเรย์พัง (สำคัญมาก ⚠️)
pip install "numpy<2" --force-reinstall

# 7. ลงทะเบียน Environment นี้เข้าสู่ระบบของ Jupyter Kernel
python -m ipykernel install --user --name usai10k-python3.9 --display-name "usai10k-python3.9"
