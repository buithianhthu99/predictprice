# Hướng dẫn chạy ứng dụng (Windows)
## Bước 1: Cài đặt python 3.9.4
- Tải file cài đặt từ https://www.python.org/downloads/windows/ 
- Tiến hành cài đặt python 3.9.4
	* *Chọn "Add python 3.9.4 to path"*
## Bước 2: Cài đặt các modules cần thiết
Mở Windows PowerShell và chạy lệnh:
-	pip install -r .\requirements.txt
	* *Đảm bảo việc cài đặt này thành công trước khi qua bước tiếp theo*
## Bước 3: Build model
Mở Windows PowerShell và chạy lệnh:
-	.\build_models.ps1
	* *Đảm bảo việc cài đặt này thành công trước khi qua bước tiếp theo*
## Bước 4: Chạy app
Mở Windows PowerShell và chạy lệnh:
- 	python .\stock_app.py

*Dashboard được chạy trên: http://localhost:8050/*
