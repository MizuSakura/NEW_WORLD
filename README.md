# My Project
---

## โครงสร้างโฟลเดอร์

| โฟลเดอร์ | คำอธิบาย |
|-----------|-----------|
| `config/` | เก็บไฟล์ configuration และ scaler ที่ถูกบันทึกไว้ |
| `data/` | เก็บข้อมูลสำหรับการฝึกและวิเคราะห์ แยกเป็น `raw` สำหรับข้อมูลต้นฉบับ, `processed` สำหรับข้อมูลหลัง preprocess และ `picture` สำหรับรูปภาพสัญญาณ |
| `experiments/` | สำหรับเก็บ notebook หรือสคริปต์ทดลองต่าง ๆ |
| `logs/` | บันทึกผลลัพธ์จาก simulation หรือ training |
| `notebooks/` | Jupyter notebooks สำหรับทดลองและตรวจสอบ pipeline |
| `scripts/` | สคริปต์เสริมหรือ automation ต่าง ๆ |
| `src/` | โค้ดหลักของโปรเจกต์ แยกเป็น sub-folder: `data`, `environment`, `models`, `trainer`, `utils` |
| `tests/` | ทดสอบโค้ดใน `src/` |
| `.gitignore` | ไฟล์ ignore สำหรับ git |
| `requirements.txt` | รายการ dependencies |
| `README.md` | ไฟล์อธิบายโปรเจกต์ |
